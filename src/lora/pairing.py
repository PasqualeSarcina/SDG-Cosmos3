"""Audit the intersection of generated captions and scene plans before loading a model."""

import logging
from pathlib import Path

from jsonschema import ValidationError

from src.lora.dataset import parse_caption, serialize_caption
from src.lora.thinking_dataset import (
    AnnotationIssue, MissingScenePlan, plan_roots, read_scene_plan, scene_plan_candidates,
)


log = logging.getLogger(__name__)
CAPTION_EXTENSIONS = {".xml", ".json", ".txt"}


def _plan_inventory(paths, caption_inputs, caption_root, plan_root):
    # Explicit file selections don't claim unrelated neighbouring plans. For
    # directory selections inspect the corresponding subtree, including orphans.
    scopes = [Path(p).resolve() for p in caption_inputs] if caption_inputs is not None else sorted({p.parent for p in paths})
    inventory = set()
    for source in scopes:
        if not source.is_relative_to(caption_root):
            raise ValueError(f"Caption input is outside caption_root: {source}")
        if source.is_dir():
            directory = plan_root / source.relative_to(caption_root)
            if not directory.resolve().is_relative_to(plan_root):
                raise ValueError(f"Plan directory escapes plan_root: {directory}")
            candidates = directory.rglob("*") if directory.is_dir() else []
        else:
            candidates = scene_plan_candidates(source, caption_root=caption_root, plan_root=plan_root)
        for candidate in candidates:
            if "debug" in {part.casefold() for part in candidate.relative_to(plan_root).parts[:-1]}:
                continue
            if candidate.suffix.lower() != ".xml" and not candidate.name.endswith(".plan.json"):
                continue
            if not candidate.resolve().is_relative_to(plan_root):
                raise ValueError(f"Plan path escapes plan_root: {candidate}")
            if candidate.is_file():
                inventory.add(candidate)
    return sorted(inventory)


def collect_caption_plan_pairs(paths, *, caption_root, plan_root, caption_inputs=None):
    """Keep complete pairs and count discarded inputs without classifying them.

    Malformed files, stale hashed sidecars, ambiguous identifiers and unsafe
    paths remain fatal. No fallback to a neighbouring filename or list position.
    The loaded caption/plan snapshots are reused in tokenization and fingerprinting.
    """
    caption_root, plan_root = plan_roots(caption_root, plan_root)
    paths = [Path(p).resolve() for p in paths]
    inventory = _plan_inventory(paths, caption_inputs, caption_root, plan_root)
    discarded = 0
    examples, seen_sources, handled_plans, pair_keys = [], set(), set(), {}
    for path in paths:
        candidates = scene_plan_candidates(path, caption_root=caption_root, plan_root=plan_root)
        if path in seen_sources:
            raise ValueError(f"Repeated caption path: {path}")
        seen_sources.add(path)
        try:
            source = path.read_bytes()
        except FileNotFoundError:
            found = [p for p in candidates if p.is_file()]
            discarded += max(1, len(found))
            handled_plans.update(found)
            continue
        # XML plans intentionally replace the source extension. Two source
        # extensions with the same stem therefore cannot silently share one plan.
        key = path.relative_to(caption_root).with_suffix("").as_posix().casefold()
        if key in pair_keys:
            raise ValueError(f"Ambiguous caption identifier: {pair_keys[key]} and {path}")
        pair_keys[key] = path
        try:
            plan, record = read_scene_plan(path, caption_root=caption_root, plan_root=plan_root, source_bytes=source)
        except MissingScenePlan:
            discarded += 1
            continue
        except AnnotationIssue as issue:
            discarded += 1
            handled_plans.add(issue.path)
            continue
        except (ValueError, OSError) as error:
            raise ValueError(f"{path}: {error}") from error
        try:
            caption = parse_caption(source.decode("utf-8-sig"))
            serialize_caption(caption)  # Validate both sections before a model/tokenizer is loaded.
        except (ValueError, ValidationError) as error:
            raise ValueError(f"{path}: {error}") from error
        handled_plans.add(plan_root / record["plan"])
        examples.append(dict(path=path, caption=caption, scene_plan=plan, record=record))

    # Inventory unused plans in the selected split only. An existing caption
    # outside an explicit subset is unselected, not a failed generation.
    directories = {}
    for path in inventory:
        if path in handled_plans:
            continue
        relative = path.relative_to(plan_root)
        if path.name.endswith(".plan.json"):
            expected = caption_root / relative.as_posix().removesuffix(".plan.json")
            exists = expected.is_file()
        else:
            base = caption_root / relative.with_suffix("")
            if base.parent not in directories:
                directories[base.parent] = {p.stem.casefold() for p in base.parent.iterdir()
                                            if p.is_file() and p.suffix.lower() in CAPTION_EXTENSIONS} if base.parent.is_dir() else set()
            exists = base.name.casefold() in directories[base.parent]
        if not exists:
            discarded += 1
    if discarded:
        log.warning("Plan/caption pairing: %d discarded.", discarded)
    return {"examples": examples, "report": {"discarded": discarded}}
