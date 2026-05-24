"""Helpers for extracting JSON objects from LLM text output."""

from __future__ import annotations

import json
import re
from typing import Any


class JSONOutputError(ValueError):
    """Raised when an LLM output cannot be read as one JSON object."""


def extract_json_text(raw_output: str) -> str:
    """Return the most likely JSON object text from a raw LLM response."""
    text = (raw_output or "").strip()
    if not text:
        raise JSONOutputError("empty output")

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()

    start = text.find("{")
    if start < 0:
        return text

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:idx + 1]

    return text[start:]


def load_json_object(raw_output: str) -> dict[str, Any]:
    """Parse one JSON object from raw LLM output."""
    try:
        data = json.loads(extract_json_text(raw_output))
    except (json.JSONDecodeError, TypeError) as exc:
        raise JSONOutputError(f"JSON decode failed: {exc}") from exc

    if not isinstance(data, dict):
        raise JSONOutputError(f"expected JSON object, got {type(data).__name__}")
    return data
