import json
import re


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

def assemble_caption_json(scene_draft: dict, final_prompt: str, media: dict) -> dict:
    """Assemble the final caption object and validate it.

    * ``temporal_caption`` is set to the polished Phase-2 ``final_prompt`` (this
      is what keeps the dense narrative available *inside* the JSON and equal to
      ``caption.txt``), overriding any draft value from Phase 1.
    * ``media`` (from :func:`media_fields_from_metadata`) is merged in.

    Returns a normalised ``dict`` (None-valued fields dropped, types coerced).
    Raises ``pydantic.ValidationError`` if the structure is unusable.
    """
    data: dict[str, Any] = dict(scene_draft)
    data["temporal_caption"] = (final_prompt or "").strip()
    data.update(media)
    model = StructuredCaption.model_validate(data)
    return model.model_dump(exclude_none=True, mode="json")