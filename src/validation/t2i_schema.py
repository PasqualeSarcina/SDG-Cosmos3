import re
from typing import Any


def t2i_template_to_json_schema(template: dict[str, Any]) -> dict[str, Any]:
    """Convert the descriptive T2I template into an OpenAI strict JSON schema.

    All declared fields are required. Free-form ``subject_details`` keys are
    represented as key/value entries, since strict schemas forbid open objects.
    """
    def convert(value: Any, path: tuple[str, ...]) -> dict[str, Any]:
        if path == ("subject_details",):
            return {
                "type": "array",
                "description": "Image-specific attributes as key/value entries; an empty list if inapplicable.",
                "items": {
                    "type": "object",
                    "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
                    "required": ["key", "value"],
                    "additionalProperties": False,
                },
            }
        if isinstance(value, dict):
            return {
                "type": "object",
                "properties": {key: convert(item, (*path, key)) for key, item in value.items()},
                "required": list(value),
                "additionalProperties": False,
            }
        if isinstance(value, list):
            if len(value) != 1:
                raise ValueError(f"T2I template array {'.'.join(path)} must contain one example item.")
            return {"type": "array", "items": convert(value[0], (*path, "items"))}
        if not isinstance(value, str):
            raise ValueError(f"Expected a field description at {'.'.join(path)}.")
        if path in (("resolution", "H"), ("resolution", "W")):
            return {"type": "integer", "minimum": 1, "description": value}
        if value.startswith("int;"):
            return {"type": "integer", "minimum": 0, "description": value}
        schema: dict[str, Any] = {"type": "string", "description": value}
        if re.match(r"^(?:must be )?one of\b", value) or (
            value.startswith("'") and " or '" in value
        ):
            choices = re.findall(r"'([^']*)'", value)
            if choices:
                schema["enum"] = choices
        if path == ("comprehensive_t2i_caption",):
            schema["minLength"] = 1
        return schema

    return convert(template, ())
