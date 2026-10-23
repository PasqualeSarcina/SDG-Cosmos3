"""Validate image scene drafts before adding captions and image metadata."""

import json
from functools import lru_cache
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError


@lru_cache(maxsize=1)
def _scene_validator() -> Draft202012Validator:
    path = Path(__file__).resolve().parents[1] / "schema/image_description_json_schema.json"
    schema = json.loads(path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def validate_scene_draft(scene: dict, *, allow_caption_field: bool = False) -> None:
    """Validate the raw draft without coercion; preserve the JSON-only fallback.

    The JSON-only response carries its narrative alongside the scene fields.
    Exclude that field only for this legacy format, without changing the input.
    """
    if allow_caption_field:
        scene = {key: value for key, value in scene.items() if key != "comprehensive_t2i_caption"}
    error = next(_scene_validator().iter_errors(scene), None)
    if error is not None:
        # Keep retry diagnostics useful without dumping the entire scene/schema.
        raise ValidationError(f"Invalid scene_draft at {error.json_path}: {error.message}") from error
