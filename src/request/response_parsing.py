"""Shared parsing of JSON chat responses, including Markdown code blocks."""

import json
import re
from typing import Any


def _strip_code_fences(text: str) -> str:
    """Remove a complete Markdown JSON code block around the response."""
    cleaned = text.strip()
    match = re.fullmatch(r"```(?:json)?[ \t]*\r?\n?(.*?)\r?\n?```", cleaned, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else cleaned


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

def parse_json_response(text: str, *, allow_prose: bool = False) -> Any:
    """Parse JSON returned directly or inside a Markdown code block.

    With allow_prose, also recover the first balanced object from surrounding
    text. Invalid or incomplete JSON raises json.JSONDecodeError.
    """
    cleaned = _strip_code_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        if allow_prose:
            blob = _first_json_object(cleaned)
            if blob is not None:
                return json.loads(blob)
        raise
