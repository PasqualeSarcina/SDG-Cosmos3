"""Strict caption preparation for the position-weighted Cosmos loss.

The full chat is tokenized once, without truncation or added special tokens.
Annotations refer to the canonical serialization, not to punctuation/token-ID
heuristics. Use the checkpoint's fast tokenizer and its own chat template.
"""

from __future__ import annotations

import json
from bisect import bisect_right
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import ValidationError

from src.lora.token_classes import TokenClass
from src.validation.scene_draft import validate_scene_draft


USER_PROMPT = (
    "Generate a new, realistic, and visually diverse COCO-style scene containing "
    "at least one car. Invent the scene from scratch."
)
_SRC = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AnnotationSpan:
    start: int
    end: int
    token_class: TokenClass
    field_id: int = -1


@dataclass(frozen=True)
class AnnotatedTarget:
    text: str
    spans: tuple[AnnotationSpan, ...]
    field_paths: tuple[str, ...]


@dataclass(frozen=True)
class MixedToken:
    token_index: int
    start: int
    end: int
    classes: tuple[int, ...]
    field_ids: tuple[int, ...]


@dataclass(frozen=True)
class PreparedCaption:
    features: dict[str, list[int]]
    target: AnnotatedTarget
    # Offsets and mixed-token positions refer to the full rendered chat.
    rendered_chat: str
    offsets: tuple[tuple[int, int], ...]
    mixed_tokens: tuple[MixedToken, ...]


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}.")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}.")


def parse_caption(text: str) -> dict[str, Any]:
    """Read only the exact two-section XML/JSON caption protocol.

    JSON is decoded before looking for the scene closing tag, so tag-like text
    inside JSON strings is harmless. Narrative text is not XML-unescaped.
    Prose, Markdown fences, duplicate keys and incomplete sections are errors.
    """
    decoder = json.JSONDecoder(object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    text = text.strip()
    if not text.startswith("<scene_draft>"):
        raise ValueError("Expected <scene_draft> followed by <comprehensive_t2i_caption>.")
    opening_end = len("<scene_draft>")
    json_start = opening_end + len(text[opening_end:]) - len(text[opening_end:].lstrip())
    scene, end = decoder.raw_decode(text, json_start)
    rest = text[end:].lstrip()
    bridge = "</scene_draft>"
    if not rest.startswith(bridge):
        raise ValueError("Missing </scene_draft> immediately after the JSON object.")
    rest = rest[len(bridge):].lstrip()
    opening, closing = "<comprehensive_t2i_caption>", "</comprehensive_t2i_caption>"
    if not rest.startswith(opening) or not rest.endswith(closing):
        raise ValueError("Expected the narrative section after </scene_draft>.")
    narrative = rest[len(opening):-len(closing)].strip()
    if closing in narrative:
        raise ValueError("Repeated narrative closing tag.")
    if not isinstance(scene, dict) or "comprehensive_t2i_caption" in scene:
        raise ValueError("scene_draft must be an object without an embedded narrative.")
    return {**scene, "comprehensive_t2i_caption": narrative}


@lru_cache(maxsize=1)
def _schema():
    return json.loads((_SRC / "schema/image_description_json_schema.json").read_text(encoding="utf-8"))


def _pointer(path):
    return "/" + "/".join(str(part).replace("~", "~0").replace("/", "~1") for part in path)


class _TargetWriter:
    """Serialize validated values while retaining exact character spans."""

    def __init__(self):
        self.parts = []
        self.spans = []
        self.field_paths = []
        self.position = 0

    def write(self, text, kind=TokenClass.STRUCTURE, field_id=-1, *, decision=False):
        if decision and text:
            self.write(text[:1], TokenClass.DECISION)
            self.write(text[1:], kind, field_id)
            return
        if text:
            self.parts.append(text)
            self.spans.append(AnnotationSpan(self.position, self.position + len(text), kind, field_id))
            self.position += len(text)

    def field(self, path):
        self.field_paths.append(_pointer(path))
        return len(self.field_paths) - 1

    def scalar(self, value, field_id, *, decision=False):
        serialized = json.dumps(value, ensure_ascii=False, allow_nan=False)
        if isinstance(value, str):
            self.write('"', decision=decision)
            # The closing quote is the supervised decision to end this value,
            # including empty strings. Escapes stay in the same field.
            self.write(serialized[1:], TokenClass.CONTENT, field_id)
        else:
            self.write(serialized, TokenClass.CONTENT, field_id, decision=decision)

    def value(self, value, schema, path=(), *, decision=False):
        if isinstance(value, dict):
            dynamic = path == ("subject_details",)
            if any(not isinstance(key, str) for key in value):
                raise ValueError(f"JSON object keys must be strings at {_pointer(path)}.")
            self.write("{", decision=decision)
            keys = sorted(value) if dynamic else list(schema["properties"])
            if keys:
                self.write("\n")
            for index, key in enumerate(keys):
                self.write("  " * (len(path) + 1))
                if dynamic:
                    field_id = self.field((*path, key))
                    self.scalar(key, field_id, decision=index == 0)
                    self.write(": ")
                    self.scalar(value[key], field_id)
                else:
                    self.write(json.dumps(key, ensure_ascii=False) + ": ")
                    self.value(value[key], schema["properties"][key], (*path, key))
                if index < len(keys) - 1:
                    self.write(",", TokenClass.DECISION if dynamic else TokenClass.STRUCTURE)
                self.write("\n")
            if keys:
                self.write("  " * len(path))
            self.write("}", TokenClass.DECISION if dynamic else TokenClass.STRUCTURE)
        elif isinstance(value, list):
            self.write("[", decision=decision)
            if value:
                self.write("\n")
            for index, item in enumerate(value):
                self.write("  " * (len(path) + 1))
                self.value(item, schema["items"], (*path, index), decision=index == 0)
                if index < len(value) - 1:
                    self.write(",", TokenClass.DECISION)
                self.write("\n")
            if value:
                self.write("  " * len(path))
            self.write("]", TokenClass.DECISION)
        else:
            self.scalar(value, self.field(path), decision=decision)


def serialize_caption(caption: Mapping[str, Any]) -> AnnotatedTarget:
    """Copy, strictly validate and annotate a canonical two-section target.

    Fixed fields use schema order, dynamic keys use lexical order, and lists
    retain source order. Missing fields, unknown fields and non-string
    subject_details values are rejected.
    """
    if not isinstance(caption, Mapping):
        raise ValueError("A caption must be a mapping.")
    scene = dict(caption)
    narrative = scene.pop("comprehensive_t2i_caption", None)
    if not isinstance(narrative, str) or not narrative.strip():
        raise ValueError("A nonempty comprehensive_t2i_caption string is required.")
    if "</comprehensive_t2i_caption>" in narrative:
        raise ValueError("Narrative contains a protocol closing tag.")
    validate_scene_draft(scene)
    writer = _TargetWriter()
    writer.write("<scene_draft>\n")
    writer.value(scene, _schema())
    writer.write("\n</scene_draft>\n<comprehensive_t2i_caption>\n")
    field_id = writer.field(("comprehensive_t2i_caption",))
    writer.write(narrative.strip(), TokenClass.CONTENT, field_id)
    writer.write("\n")
    writer.write("<", TokenClass.CONTENT, field_id)
    writer.write("/comprehensive_t2i_caption>")
    return AnnotatedTarget("".join(writer.parts), tuple(writer.spans), tuple(writer.field_paths))


def prepare_caption(
    caption: Mapping[str, Any],
    tokenizer,
    *,
    max_length: int,
    system_prompt: str | None = None,
    user_prompt: str = USER_PROMPT,
    response_end_token: str | None = None,
) -> PreparedCaption:
    """Build unshifted Trainer features, retaining annotations for inspection.

    The end token defaults to tokenizer.eos_token, but must match the actual
    assistant terminator emitted by the template. Thinking is disabled. Only
    whitespace may follow the terminator; that template whitespace is excluded.
    A token crossing excluded/supervised spans is an error. Other mixed tokens
    use DECISION > CONTENT > STRUCTURE, and the first touched content field, with an audit record.
    """
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("Use a fast tokenizer with character offset mappings.")
    if system_prompt is None:
        system_prompt = (_SRC / "templates/generate_caption.txt").read_text(encoding="utf-8")
    if not user_prompt.strip():
        raise ValueError("User prompt must be nonempty.")
    target = serialize_caption(caption)
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    options = dict(tokenize=False, enable_thinking=False)
    prefix = tokenizer.apply_chat_template(messages, add_generation_prompt=True, **options)
    rendered = tokenizer.apply_chat_template(
        [*messages, {"role": "assistant", "content": target.text}], add_generation_prompt=False, **options,
    )
    if not rendered.startswith(prefix + target.text):
        raise ValueError("Training chat does not match the non-thinking generation prefix and exact target.")
    start, end = len(prefix), len(prefix) + len(target.text)
    terminator = response_end_token if response_end_token is not None else tokenizer.eos_token
    suffix = rendered[end:]
    if not terminator or not suffix.startswith(terminator) or suffix[len(terminator):].strip():
        raise ValueError("response_end_token must match the template's assistant terminator.")
    terminator_id = tokenizer.convert_tokens_to_ids(terminator)
    if terminator_id not in tokenizer.all_special_ids or tokenizer.convert_ids_to_tokens(terminator_id) != terminator:
        raise ValueError("The assistant terminator must be a registered single special token.")
    encoded = tokenizer(rendered, add_special_tokens=False, truncation=False, padding=False,
                        return_offsets_mapping=True, return_attention_mask=True)
    ids = encoded["input_ids"]
    offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
    if len(ids) > max_length:
        raise ValueError(f"Example has {len(ids)} tokens, exceeding max_length={max_length}; no truncation applied.")
    if len(ids) != len(offsets) or len(ids) < 2:
        raise ValueError("Invalid tokenizer IDs/offsets.")
    for left, right in offsets:
        if not 0 <= left <= right <= len(rendered):
            raise ValueError("Tokenizer offsets must refer to characters in the complete chat.")
    body_indices = [i for i, (left, right) in enumerate(offsets) if left < end and right > start]
    if not body_indices:
        raise ValueError("Tokenizer returned no offsets for the target.")
    end_indices = [i for i, token_id in enumerate(ids) if token_id == terminator_id and i > body_indices[-1]]
    if len(end_indices) != 1:
        raise ValueError("Expected exactly one assistant terminator after the target.")
    end_index = end_indices[0]
    body_index_set = set(body_indices)
    special_ids = set(tokenizer.all_special_ids)
    spans = [AnnotationSpan(0, start, TokenClass.EXCLUDED)]
    spans.extend(AnnotationSpan(span.start + start, span.end + start, span.token_class, span.field_id)
                 for span in target.spans)
    spans.append(AnnotationSpan(end, end + len(terminator), TokenClass.DECISION))
    if end + len(terminator) < len(rendered):
        spans.append(AnnotationSpan(end + len(terminator), len(rendered), TokenClass.EXCLUDED))
    span_ends = [span.end for span in spans]
    classes, fields, mixed = [], [], []
    covered = bytearray(len(target.text))
    for index, (token_id, (left, right)) in enumerate(zip(ids, offsets)):
        if left == right:
            # Some fast tokenizers use (0, 0) for inserted special tokens.
            if index == end_index:
                classes.append(int(TokenClass.DECISION))
            elif index < body_indices[0]:
                classes.append(int(TokenClass.EXCLUDED))
            else:
                raise ValueError(f"Unmapped token {index} inside/after the response.")
            fields.append(-1)
            continue
        hits = []
        for span_index in range(bisect_right(span_ends, left), len(spans)):
            span = spans[span_index]
            if span.start >= right:
                break
            hits.append(span)
        kinds = {span.token_class for span in hits}
        if not kinds or (TokenClass.EXCLUDED in kinds and len(kinds) > 1):
            raise ValueError(f"Token {index} crosses excluded and supervised text; adjust serialization/tokenizer.")
        if index == end_index and (left, right) != (end, end + len(terminator)):
            raise ValueError("Assistant terminator offsets do not match its template span.")
        if index in body_index_set and token_id in special_ids:
            raise ValueError(f"Target contains a chat control token at position {index}.")
        kind = next(kind for kind in (TokenClass.EXCLUDED, TokenClass.DECISION, TokenClass.CONTENT, TokenClass.STRUCTURE) if kind in kinds)
        content_fields = tuple(dict.fromkeys(span.field_id for span in hits if span.token_class == TokenClass.CONTENT))
        classes.append(int(kind))
        fields.append(content_fields[0] if kind == TokenClass.CONTENT else -1)
        if len(kinds) > 1 or len(content_fields) > 1:
            mixed.append(MixedToken(index, left, right, tuple(sorted(map(int, kinds))), content_fields))
        a, b = max(left, start), min(right, end)
        if a < b:
            covered[a - start:b - start] = b"\x01" * (b - a)
    if not all(covered) or TokenClass.CONTENT not in classes[1:]:
        raise ValueError("Incomplete target offsets or no content tokens after the causal shift.")
    features = {
        "input_ids": list(ids),
        "attention_mask": [1] * len(ids),
        "labels": [token_id if kind != TokenClass.EXCLUDED else -100 for token_id, kind in zip(ids, classes)],
        "token_classes": classes,
        "json_field_ids": fields,
    }
    return PreparedCaption(features, target, rendered, tuple(offsets), tuple(mixed))


def build_dataset(captions: Iterable[Mapping[str, Any]], tokenizer, *, max_length: int, **kwargs):
    """Build a Hugging Face Dataset with exactly the five collator columns.

    Call separately on train/validation sources already split by origin/group.
    prepare_caption exposes the detailed token audit for representative examples.
    """
    from datasets import Dataset

    rows = []
    for index, caption in enumerate(captions):
        try:
            rows.append(prepare_caption(caption, tokenizer, max_length=max_length, **kwargs).features)
        except (ValueError, ValidationError) as error:
            raise ValueError(f"Caption {index}: {error}") from error
    if not rows:
        raise ValueError("Cannot build an empty caption dataset.")
    return Dataset.from_list(rows)


def load_caption_dataset(paths: Iterable[str | Path], tokenizer, *, max_length: int, **kwargs):
    """Load explicit two-section caption files; never modify sources or silently skip errors.

    Pass separate file lists for train/validation. No automatic random split is
    performed because near-duplicates and related images must stay together.
    """
    def examples():
        for value in paths:
            path = Path(value)
            try:
                caption = parse_caption(path.read_text(encoding="utf-8-sig"))
                # Include the source path in preprocessing errors as well.
                yield prepare_caption(caption, tokenizer, max_length=max_length, **kwargs).features
            except (ValueError, ValidationError, OSError) as error:
                raise ValueError(f"{path}: {error}") from error

    from datasets import Dataset

    rows = list(examples())
    return Dataset.from_list(rows)
