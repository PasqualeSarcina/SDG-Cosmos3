import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _CaptionModel(BaseModel):
    # Preserve image-specific attributes rather than silently dropping them.
    model_config = ConfigDict(extra="allow")


class Subject(_CaptionModel):
    description: str | None = None
    appearance_details: str | None = None
    relationship: str | None = None
    location: str | None = None
    relative_size: str | None = None
    orientation: str | None = None
    pose: str | None = None
    clothing: str | None = None
    expression: str | None = None
    gender: str | None = None
    age: str | None = None
    skin_tone_and_texture: str | None = None
    facial_features: str | None = None
    number_of_subjects: int | None = Field(default=None, ge=0)
    number_of_arms: int | None = Field(default=None, ge=0)
    number_of_legs: int | None = Field(default=None, ge=0)
    number_of_hands: int | None = Field(default=None, ge=0)
    number_of_fingers: int | None = Field(default=None, ge=0)


class Lighting(_CaptionModel):
    conditions: str | None = None
    direction: str | None = None
    shadows: str | None = None
    illumination_effect: str | None = None


class Aesthetics(_CaptionModel):
    composition: str | None = None
    color_scheme: str | None = None
    mood_atmosphere: str | None = None
    patterns: str | None = None


class Cinematography(_CaptionModel):
    framing: str | None = None
    camera_angle: str | None = None
    depth_of_field: str | None = None
    focus: str | None = None
    lens_focal_length: str | None = None


class TextElement(_CaptionModel):
    text: str | None = None
    category: str | None = None
    appearance: str | None = None
    spatial: str | None = None
    context: str | None = None


class QuadrantScan(_CaptionModel):
    top_left: str | None = None
    top_right: str | None = None
    bottom_left: str | None = None
    bottom_right: str | None = None
    absolute_center: str | None = None


class Resolution(_CaptionModel):
    H: int = Field(gt=0)
    W: int = Field(gt=0)


class StructuredCaption(_CaptionModel):
    """Validate the image-caption template without generation-size constraints."""

    comprehensive_t2i_caption: str = Field(min_length=1)
    subjects: list[Subject] | None = None
    subject_details: dict[str, Any] | None = None
    background_setting: str | None = None
    lighting: Lighting | None = None
    aesthetics: Aesthetics | None = None
    cinematography: Cinematography | None = None
    style_medium: str | None = None
    artistic_style: str | None = None
    context: str | None = None
    text_and_signage_elements: list[TextElement] | None = None
    quadrant_scan: QuadrantScan | None = None
    resolution: Resolution | None = None
    aspect_ratio: str | None = None


def extract_xml_tag(text: str, tag: str) -> str | None:
    """Return the inner text of ``<tag>...</tag>`` (DOTALL), or ``None``."""
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.DOTALL)
    return match.group(1).strip() if match else None

def _strip_code_fences(text: str) -> str:
    """Strip a leading ```json / ``` fence and trailing ``` if present."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    return cleaned.strip()


def parse_caption_fragments(text: str) -> dict:
    """Parse the Phase-2 JSON object used as input to offline augmentation."""
    try:
        parsed = json.loads(_strip_code_fences(text))
    except json.JSONDecodeError as exc:
        raise ValueError("Expected valid Phase-2 JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Expected a Phase-2 JSON object")
    return parsed


def _first_json_object(text: str) -> str | None:
    """Return the first balanced ``{...}`` block in ``text``, or ``None``.

    Brace-counting fallback for when the model wraps the JSON in prose without
    fences/tags.  Ignores braces inside double-quoted strings.
    """
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None

def parse_structured_caption(text: str) -> dict | None:
    """Extract the Phase-1 ``<scene_draft>`` JSON object from a VLM response.

    Resolution order, each tolerant of ```` ```json ```` fences:

    1. The ``<scene_draft>`` XML block.
    2. The whole response (if it is itself a JSON object).
    3. The first balanced ``{...}`` block anywhere in the response.

    Returns the parsed ``dict`` on success, or ``None`` if no valid JSON object
    can be recovered (the caller should retry).
    """
    candidates: list[str] = []
    tagged = extract_xml_tag(text, "scene_draft")
    if tagged is not None:
        candidates.append(tagged)
    candidates.append(text)

    for candidate in candidates:
        cleaned = _strip_code_fences(candidate)
        for blob in (cleaned, _first_json_object(cleaned)):
            if not blob:
                continue
            try:
                parsed = json.loads(blob)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
    return None

def aspect_ratio_str(width: int, height: int) -> str:
    """Reduce ``width``/``height`` to a ``"W,H"`` ratio string (e.g. ``"1,1"``)."""
    from math import gcd

    if width <= 0 or height <= 0:
        return ""
    g = gcd(int(width), int(height)) or 1
    return f"{int(width) // g},{int(height) // g}"

def media_fields_from_metadata(meta: tuple[int, int]) -> dict:
    """Build the caption's media fields from :func:`probe_video_metadata` output.

    Uses the image's *actual* values (not the canonical generation enums): the
    enums constrain the upsampler's generation params, not ground-truth captions.
    """
    width, height = int(meta[0]), int(meta[1])
    return {
        "resolution": {"H": height, "W": width},
        "aspect_ratio": aspect_ratio_str(width, height),
    }

def assemble_caption_json(scene_draft: dict, comprehensive_t2i_caption: str, media: dict) -> dict:
    """Assemble the final caption object and validate it.

    * ``comprehensive_t2i_caption`` contains the polished Phase-2 narrative,
      equal to ``caption.txt``, overriding any draft value from Phase 1.
    * ``media`` (from :func:`media_fields_from_metadata`) is merged in.

    Returns a normalised ``dict`` (None-valued fields dropped, types coerced).
    Raises ``pydantic.ValidationError`` if the structure is unusable.
    """
    data: dict[str, Any] = dict(scene_draft)
    data.update(media)
    data.pop("temporal_caption", None)
    data["comprehensive_t2i_caption"] = (comprehensive_t2i_caption or "").strip()
    model = StructuredCaption.model_validate(data)
    return model.model_dump(exclude_none=True, mode="json")
