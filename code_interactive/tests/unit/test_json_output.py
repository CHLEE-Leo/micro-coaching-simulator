from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from code_interactive.agents.json_output import JSONOutputError, load_json_object


def test_load_json_object_from_plain_json():
    assert load_json_object('{"action": "inquire"}') == {"action": "inquire"}


def test_load_json_object_from_markdown_fence():
    raw = 'Here is the decision:\n```json\n{"action": "assess"}\n```'
    assert load_json_object(raw) == {"action": "assess"}


def test_load_json_object_handles_nested_object():
    raw = 'prefix {"outer": {"inner": "ok"}, "items": [1, 2]} suffix'
    assert load_json_object(raw) == {
        "outer": {"inner": "ok"},
        "items": [1, 2],
    }


def test_load_json_object_requires_object():
    with pytest.raises(JSONOutputError):
        load_json_object('["not", "an", "object"]')
