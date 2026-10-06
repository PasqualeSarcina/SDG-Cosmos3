"""Apply weather/time presets to both parts of an image caption response."""

from copy import deepcopy

from src.augmentation.augm_phrases import AUGM_PHRASES
from src.caption.structured_caption import parse_caption_fragments


def _merge_fields(target: dict, replacements: dict) -> None:
    """Replace preset leaves while preserving unrelated generated fields."""
    for key, value in replacements.items():
        if key == "background_setting":
            existing = target.get(key) or ""
            if not isinstance(existing, str) or not isinstance(value, str):
                raise ValueError("background_setting must be a string")
            target[key] = " ".join(part.strip() for part in (existing, value) if part.strip())
        elif isinstance(value, dict):
            if not isinstance(target.get(key), dict):
                target[key] = {}
            _merge_fields(target[key], value)
        else:
            target[key] = deepcopy(value)


def _text_fragments(value):
    """Yield nonempty text leaves in JSON insertion order, without field labels."""
    if isinstance(value, str):
        if value.strip():
            yield value.strip()
    elif isinstance(value, dict):
        for child in value.values():
            yield from _text_fragments(child)
    elif isinstance(value, list):
        for child in value:
            yield from _text_fragments(child)
    elif value is not None:
        raise ValueError("Phase-2 caption fields must contain text, objects, or lists of text")


def augment_caption(
    scene_draft: dict,
    comprehensive_t2i_caption: str,
    weather: str = "rainy",
    time_of_day: str = "day",
) -> tuple[dict, str]:
    """Apply json_fields to Phase 1 and caption_fragments to Phase 2.

    Inputs and AUGM_PHRASES are never mutated. Phase 2 must contain a JSON
    object. After replacement, Phase-1 palette and intensity become a single
    color_scheme string, and all Phase-2 text leaves become a single narrative.
    """
    try:
        preset = AUGM_PHRASES[(weather, time_of_day)]
    except KeyError as exc:
        raise ValueError(f"Unknown augmentation preset: {weather!r}, {time_of_day!r}") from exc

    phase2 = parse_caption_fragments(comprehensive_t2i_caption)

    phase1 = deepcopy(scene_draft)
    for target, fields, intensity_path in (
        (phase1, preset["json_fields"], ("aesthetics",)),
        (phase2, preset["caption_fragments"], ("aesthetics", "color_scheme")),
    ):
        replacements = deepcopy(fields)
        if "intensity" in replacements:
            intensity = replacements.pop("intensity")
            nested = target
            for key in intensity_path:
                if not isinstance(nested.get(key), dict):
                    nested[key] = {}
                nested = nested[key]
            nested["intensity"] = intensity
        _merge_fields(target, replacements)

    aesthetics = phase1["aesthetics"]
    palette = aesthetics.pop("palette", "")
    intensity = aesthetics.pop("intensity", "")
    if not isinstance(palette, str) or not isinstance(intensity, str):
        raise ValueError("Phase-1 aesthetics.palette and aesthetics.intensity must be strings")
    aesthetics["color_scheme"] = " ".join(_text_fragments([palette, intensity]))

    return phase1, " ".join(_text_fragments(phase2))
