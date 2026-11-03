"""Opt-in native-reasoning targets; the legacy two-section dataset stays unchanged."""

from __future__ import annotations

import hashlib
import json
import re
from bisect import bisect_right
from pathlib import Path
from xml.etree import ElementTree

from jsonschema import ValidationError

from src.lora.dataset import (
    USER_PROMPT, AnnotationSpan, MixedToken, PreparedCaption, _SRC, _unique_object,
    parse_caption, serialize_caption,
)
from src.lora.token_classes import TokenClass as K


STUDENT_REASONING_EFFORT = "xhigh"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_plan(plan):
    if not isinstance(plan, str) or not plan.strip():
        raise ValueError("A nonempty scene plan is required.")
    plan = plan.strip()
    if re.search(r"</?(?:think|scene_plan|annotation_issue|scene_draft|comprehensive_t2i_caption)\b|<\|", plan):
        raise ValueError("The plan must contain plain text, without protocol/chat markers.")
    return plan


def plan_roots(caption_root, plan_root):
    if caption_root is None or plan_root is None:
        raise ValueError("Supervised thinking requires caption_root and plan_root.")
    caption_root, plan_root = Path(caption_root).resolve(), Path(plan_root).resolve()
    if not caption_root.is_dir() or not plan_root.is_dir():
        raise ValueError("Caption and plan roots must be existing directories.")
    if caption_root.is_relative_to(plan_root) or plan_root.is_relative_to(caption_root):
        raise ValueError("Caption and plan roots must not overlap.")
    return caption_root, plan_root


class MissingScenePlan(ValueError):
    """An expected generation is absent, rather than malformed."""


class AnnotationIssue(ValueError):
    """The teacher explicitly rejected this source; never supervise the marker."""

    def __init__(self, path, reason):
        self.path, self.reason = path, reason
        super().__init__(f"Annotation issue in {path}: {reason or '(no explanation)'}")


def scene_plan_candidates(source, *, caption_root, plan_root):
    source = Path(source).resolve()
    if not source.is_relative_to(caption_root):
        raise ValueError(f"Caption is outside caption_root: {source}")
    relative = source.relative_to(caption_root)
    candidates = [plan_root / relative.with_suffix(".xml"), plan_root / (relative.as_posix() + ".plan.json")]
    for candidate in candidates:
        if not candidate.resolve().is_relative_to(plan_root):
            raise ValueError(f"Plan path escapes plan_root: {candidate}")
    return candidates


def read_scene_plan(caption_path, *, caption_root, plan_root, source_bytes=None):
    """Pair by relative path, supporting existing XML plans and hashed JSON sidecars.

    XML files from generate_thinkink.py have no source hash. Their association is
    by path; the training manifest fingerprints both files for subsequent resume.
    JSON sidecars additionally verify the original source hash before training.
    """
    caption_root, plan_root = plan_roots(caption_root, plan_root)
    source = Path(caption_path).resolve()
    candidates = scene_plan_candidates(source, caption_root=caption_root, plan_root=plan_root)
    relative = source.relative_to(caption_root)
    found = [p for p in candidates if p.is_file()]
    if not found:
        raise MissingScenePlan(f"Expected exactly one plan for {relative.as_posix()}; found 0.")
    if len(found) > 1:
        raise ValueError(f"Expected exactly one plan for {relative.as_posix()}; found {len(found)}.")
    path = found[0]
    raw = path.read_bytes()
    source_hash = sha256(source.read_bytes() if source_bytes is None else source_bytes)
    if path.suffix == ".json":
        data = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique_object)
        if not isinstance(data, dict) or type(data.get("schema_version")) is not int or data["schema_version"] != 1:
            raise ValueError(f"Unsupported plan schema: {path}")
        if data.get("source_caption") != relative.as_posix() or data.get("source_sha256") != source_hash:
            raise ValueError(f"Plan source path/hash mismatch: {path}")
        plan = validate_plan(data.get("scene_plan"))
        source_verified = True
    else:
        text = raw.decode("utf-8-sig").strip()
        if "<!" in text or "<?" in text:
            raise ValueError(f"Expected a plain scene_plan XML document: {path}")
        try:
            root = ElementTree.fromstring(text)
        except ElementTree.ParseError as error:
            raise ValueError(f"Invalid plan XML: {path}: {error}") from error
        if root.tag in {"scene_plan", "annotation_issue"}:
            issues = list(root.iter("annotation_issue"))
            if issues:
                raise AnnotationIssue(path, " ".join("".join(issue.itertext()).strip() for issue in issues).strip())
        if root.tag != "scene_plan" or root.attrib or len(root):
            raise ValueError(f"Expected one scene_plan without nested elements: {path}")
        plan = validate_plan(root.text)
        source_verified = False
    return plan, {
        "source": relative.as_posix(), "source_sha256": source_hash,
        "plan": path.relative_to(plan_root).as_posix(), "plan_sha256": sha256(raw),
        "source_hash_verified": source_verified,
    }


def load_scene_plan(caption_path, *, caption_root, plan_root):
    return read_scene_plan(caption_path, caption_root=caption_root, plan_root=plan_root)[0]


def _encode(tokenizer, text):
    return tokenizer(text, add_special_tokens=False, truncation=False, padding=False,
                     return_offsets_mapping=True, return_attention_mask=True)


def thinking_protocol(tokenizer):
    result = {}
    for name, text in (("open_id", "<think>"), ("close_id", "</think>"), ("eos_id", tokenizer.eos_token)):
        token_id = tokenizer.convert_tokens_to_ids(text)
        # Qwen's thinking markers are registered vocabulary tokens but may not
        # belong to all_special_ids. Verify the actual encoding and round trip.
        if (not text or token_id is None or tokenizer.convert_ids_to_tokens(token_id) != text
                or _encode(tokenizer, text)["input_ids"] != [token_id]):
            raise ValueError(f"Thinking requires a registered single token for {text!r}.")
        if name == "eos_id" and token_id not in tokenizer.all_special_ids:
            raise ValueError("The assistant EOS must be a registered special token.")
        result[name] = token_id
    if len(set(result.values())) != 3:
        raise ValueError("Thinking delimiters and EOS must have distinct token IDs.")
    result["answer_separator_ids"] = list(_encode(tokenizer, "\n\n")["input_ids"])
    return result


def prepare_planned_caption(caption, tokenizer, *, scene_plan, max_length, system_prompt=None,
                            user_prompt=USER_PROMPT, response_end_token=None):
    """Validate the actual template and token boundaries before assigning classes.

    The supported protocol is Qwen's native reasoning_content with xhigh effort:
    open-think in the prefix; plan, newline, close-think, two newlines, final, EOS.
    Unknown template layouts fail rather than silently masking the wrong targets.
    """
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("Use a fast tokenizer with character offset mappings.")
    plan = validate_plan(scene_plan)
    target = serialize_caption(caption)
    if system_prompt is None:
        system_prompt = (_SRC / "templates/generate_caption.txt").read_text(encoding="utf-8")
    if not user_prompt.strip():
        raise ValueError("User prompt must be nonempty.")
    protocol = thinking_protocol(tokenizer)
    if re.search(r"</?think>|<\|", target.text):
        raise ValueError("Final response contains a thinking control token.")
    if response_end_token is not None and response_end_token != tokenizer.eos_token:
        raise ValueError("Supervised thinking requires the native tokenizer EOS.")
    for token_id in tokenizer.all_special_ids:
        marker = tokenizer.convert_ids_to_tokens(token_id)
        if marker and (marker in plan or marker in target.text):
            raise ValueError("Plan/final response contains a chat control token.")
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    options = dict(tokenize=False, enable_thinking=True, reasoning_effort=STUDENT_REASONING_EFFORT,
                   preserve_thinking=True)
    prefix = tokenizer.apply_chat_template(messages, add_generation_prompt=True, **options)
    if not prefix.endswith("<think>\n"):
        raise ValueError("The generation prefix must end with an open native thinking block.")
    rendered = tokenizer.apply_chat_template(
        [*messages, {"role": "assistant", "content": target.text, "reasoning_content": plan}],
        add_generation_prompt=False, **options,
    )
    start = len(prefix)
    plan_end = start + len(plan)
    close_start = plan_end + 1
    close_end = close_start + len("</think>")
    answer_start = close_end + 2
    answer_end = answer_start + len(target.text)
    eos_end = answer_end + len(tokenizer.eos_token)
    expected = prefix + plan + "\n</think>\n\n" + target.text + tokenizer.eos_token
    if not rendered.startswith(expected) or rendered[len(expected):].strip():
        raise ValueError("Thinking template must preserve prefix, exact plan, final response and EOS.")
    encoded, prefix_encoded = _encode(tokenizer, rendered), _encode(tokenizer, prefix)
    ids = list(encoded["input_ids"])
    offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
    prefix_ids = list(prefix_encoded["input_ids"])
    if ids[:len(prefix_ids)] != prefix_ids:
        raise ValueError("Generation prefix token IDs do not match the complete training chat.")
    if len(ids) > max_length:
        raise ValueError(f"Example has {len(ids)} tokens, exceeding max_length={max_length}; no truncation applied.")
    if len(ids) != len(offsets) or len(ids) < 2:
        raise ValueError("Invalid tokenizer IDs/offsets.")
    tail = ids[len(prefix_ids):]
    if tail.count(protocol["close_id"]) != 1 or tail.count(protocol["eos_id"]) != 1 or protocol["open_id"] in tail:
        raise ValueError("Expected one thinking closure and one final EOS in the continuation.")
    close_index = len(prefix_ids) + tail.index(protocol["close_id"])
    eos_index = len(prefix_ids) + tail.index(protocol["eos_id"])
    expected_specials = {close_index: (close_start, close_end), eos_index: (answer_end, eos_end)}
    spans = [AnnotationSpan(0, start, K.EXCLUDED), AnnotationSpan(start, plan_end, K.THINKING),
             AnnotationSpan(plan_end, close_start, K.STRUCTURE),
             AnnotationSpan(close_start, close_end, K.DECISION),
             AnnotationSpan(close_end, answer_start, K.STRUCTURE)]
    spans.extend(AnnotationSpan(s.start + answer_start, s.end + answer_start, s.token_class, s.field_id)
                 for s in target.spans)
    spans.append(AnnotationSpan(answer_end, eos_end, K.DECISION))
    if eos_end < len(rendered):
        spans.append(AnnotationSpan(eos_end, len(rendered), K.EXCLUDED))
    ends = [s.end for s in spans]
    classes, fields, mixed, answer_indices = [], [], [], []
    covered = bytearray(eos_end - start)
    for index, (token_id, pair) in enumerate(zip(ids, offsets)):
        left, right = pair
        if not 0 <= left <= right <= len(rendered):
            raise ValueError("Tokenizer offsets are outside the complete chat.")
        if index in expected_specials:
            if pair != (0, 0) and pair != expected_specials[index]:
                raise ValueError("Control token offsets do not match the template.")
            left, right = expected_specials[index]
        elif left == right:
            if index < len(prefix_ids) and token_id in tokenizer.all_special_ids:
                classes.append(int(K.EXCLUDED))
                fields.append(-1)
                continue
            raise ValueError(f"Unmapped token {index} inside/after the response.")
        elif token_id in tokenizer.all_special_ids and index >= len(prefix_ids):
            raise ValueError("Unexpected chat control token in the continuation.")
        hits = []
        for n in range(bisect_right(ends, left), len(spans)):
            if spans[n].start >= right:
                break
            hits.append(spans[n])
        kinds = {s.token_class for s in hits}
        if not kinds or (K.EXCLUDED in kinds and len(kinds) > 1):
            raise ValueError("Token crosses excluded and supervised text.")
        if K.THINKING in kinds and K.CONTENT in kinds or left < answer_start < right:
            raise ValueError("Token crosses the thinking/final response boundary.")
        kind = next(k for k in (K.EXCLUDED, K.DECISION, K.CONTENT, K.THINKING, K.STRUCTURE) if k in kinds)
        content_fields = tuple(dict.fromkeys(s.field_id for s in hits if s.token_class == K.CONTENT))
        classes.append(int(kind))
        fields.append(content_fields[0] if kind == K.CONTENT else -1)
        if len(kinds) > 1 or len(content_fields) > 1:
            mixed.append(MixedToken(index, left, right, tuple(sorted(map(int, kinds))), content_fields))
        a, b = max(left, start), min(right, eos_end)
        if a < b:
            covered[a - start:b - start] = b"\x01" * (b - a)
        if left >= answer_start and left < eos_end:
            answer_indices.append(index)
    if not all(covered) or K.THINKING not in classes[1:] or K.CONTENT not in classes[1:] or not answer_indices:
        raise ValueError("Incomplete target offsets, thinking or final content.")
    if any(c != K.EXCLUDED for c in classes[:len(prefix_ids)]) or classes[len(prefix_ids)] != K.THINKING:
        raise ValueError("Token prefix and supervised plan are not aligned.")
    if ids[close_index + 1:answer_indices[0]] != protocol["answer_separator_ids"]:
        raise ValueError("Thinking/final separator tokens do not match the protocol.")
    features = dict(input_ids=ids, attention_mask=[1] * len(ids),
                    labels=[token if kind != K.EXCLUDED else -100 for token, kind in zip(ids, classes)],
                    token_classes=classes, json_field_ids=fields)
    return PreparedCaption(features, target, rendered, tuple(offsets), tuple(mixed))


def load_planned_dataset(paths, tokenizer, *, max_length, caption_root, plan_root, audit=None,
                         caption_inputs=None, pairing=None, **kwargs):
    from datasets import Dataset
    from src.lora.pairing import collect_caption_plan_pairs

    caption_root, plan_root = plan_roots(caption_root, plan_root)
    if pairing is None:
        pairing = collect_caption_plan_pairs(paths, caption_root=caption_root, plan_root=plan_root,
                                             caption_inputs=caption_inputs)
    if audit is not None:
        audit["pairing"] = pairing["report"]
    if not pairing["examples"]:
        raise ValueError("No valid plan-caption pairs remain; pairing report: " + json.dumps(pairing["report"]))
    rows, records, lengths = [], [], []
    protocol = thinking_protocol(tokenizer)
    for pair in pairing["examples"]:
        path = pair["path"]
        try:
            prepared = prepare_planned_caption(pair["caption"], tokenizer,
                                               scene_plan=pair["scene_plan"], max_length=max_length, **kwargs)
            features = prepared.features
            ids, classes = features["input_ids"], features["token_classes"]
            first = next(i for i, k in enumerate(classes) if k != K.EXCLUDED)
            close = ids.index(protocol["close_id"], first)
            end = next(i for i in range(close + 1, len(ids)) if ids[i] == protocol["eos_id"])
            lengths.append(dict(prefix=first, thinking=classes.count(K.THINKING),
                                answer=end - close - len(protocol["answer_separator_ids"]), total=len(ids),
                                mixed_tokens=len(prepared.mixed_tokens)))
            rows.append(features)
            records.append(pair["record"])
        except (ValueError, ValidationError, OSError) as error:
            raise ValueError(f"{path}: {error}") from error
    if not rows:
        raise ValueError("Cannot build an empty planned dataset.")
    if audit is not None:
        audit.update(records=records, lengths=lengths, protocol=protocol)
    return Dataset.from_list(rows)


def build_planned_dataset(examples, tokenizer, *, max_length, **kwargs):
    from datasets import Dataset

    rows = [prepare_planned_caption(item["caption"], tokenizer, scene_plan=item["scene_plan"],
                                    max_length=max_length, **kwargs).features for item in examples]
    if not rows:
        raise ValueError("Cannot build an empty planned dataset.")
    return Dataset.from_list(rows)
