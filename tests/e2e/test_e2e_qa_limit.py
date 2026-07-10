#!/usr/bin/env python3
"""E2E regression for repeated user questions and the respond action limit."""

from __future__ import annotations

from typing import Any, Dict

import requests

BASE = "http://localhost:8000"
TIMEOUT = 180


def _post(path: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    response = requests.post(f"{BASE}{path}", json=payload or {}, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def _delete(path: str) -> None:
    try:
        requests.delete(f"{BASE}{path}", timeout=30)
    except requests.RequestException:
        pass


def test_repeated_user_questions_do_not_trigger_unbounded_respond_loop():
    """The dialogue planner may answer questions, but should resume coaching after two."""

    session = _post(
        "/api/session/start",
        {
            "nutrition_goal": "lean_protein",
            "meal_description": "",
            "meal_ingredient": "",
            "meal_type": "lunch",
            "mode": "custom",
            "alignment_enabled": True,
            "uncertainty_tracking": True,
            "context_tracking": True,
        },
    )
    sid = session["session_id"]

    user_turns = [
        "I'm having a turkey sandwich on whole wheat with a small side salad.",
        "Is turkey breast better than deli turkey?",
        "How much protein is in that?",
        "What if I add avocado, would that be too much fat?",
        "Should I eat before or after a workout?",
    ]

    actions: list[str] = []
    try:
        for text in user_turns:
            result = _post(f"/api/session/{sid}/turn", {"user_reply": text})
            decision = result.get("dialogue_plan") or {}
            action = decision.get("action") or ""
            if action:
                actions.append(action)
            if result.get("status") != "active":
                break
    finally:
        _delete(f"/api/session/{sid}")

    max_respond_streak = 0
    current_streak = 0
    for action in actions:
        if action == "respond":
            current_streak += 1
            max_respond_streak = max(max_respond_streak, current_streak)
        else:
            current_streak = 0

    assert max_respond_streak <= 2, f"actions={actions}"
