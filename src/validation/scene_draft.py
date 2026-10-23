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


def cosmos_structural_tag() -> dict:
    """Constrain both Cosmos sections in order, using the validation schema.

    The JSON region enforces field types, required properties and enum values.
    The narrative remains free text; nonempty content, normal completion and
    semantic correctness still require checks after generation.
    """
    return {
        "type": "structural_tag",
        "format": {
            "type": "sequence",
            "elements": [
                {
                    "type": "tag",
                    "begin": "<scene_draft>",
                    "content": {"type": "json_schema", "json_schema": _scene_validator().schema},
                    "end": "</scene_draft>",
                },
                {
                    "type": "tag",
                    "begin": "<comprehensive_t2i_caption>",
                    "content": {"type": "any_text"},
                    "end": "</comprehensive_t2i_caption>",
                },
            ],
        },
    }
