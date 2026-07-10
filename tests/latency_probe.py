"""Probe FastAPI turn latency against a running code_interactive server.

This script is intentionally not a pytest test. Use it for local or deployment
latency checks:

    python tests/latency_probe.py --base-url http://127.0.0.1:8091
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_TURNS = [
    "I am thinking of having grilled chicken with rice and a small salad.",
    "Breast, skinless, about one palm-sized piece.",
]


def _post(base_url: str, path: str, payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=240) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body), time.perf_counter() - started


def run_probe(base_url: str, turns: list[str]) -> dict[str, Any]:
    start_payload = {
        "mode": "custom",
        "alignment_enabled": True,
        "nutrition_goal": "lean_protein",
        "meal_type": "dinner",
        "meal_description": "",
        "meal_ingredient": "",
        "context_tracking": True,
        "uncertainty_tracking": True,
    }
    start_response, start_elapsed = _post(base_url, "/api/session/start", start_payload)
    session_id = start_response["session_id"]

    turn_results = []
    for text in turns:
        turn_response, client_elapsed = _post(
            base_url,
            f"/api/session/{session_id}/turn",
            {"user_reply": text},
        )
        turn_results.append(
            {
                "user_reply": text,
                "client_elapsed_seconds": round(client_elapsed, 3),
                "status": turn_response.get("status"),
                "coach_question": turn_response.get("coach_question"),
                "latency": turn_response.get("latency"),
            }
        )

    return {
        "base_url": base_url,
        "session_id": session_id,
        "start_elapsed_seconds": round(start_elapsed, 3),
        "first_question": start_response.get("first_question"),
        "turns": turn_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", default="")
    parser.add_argument("--turn", action="append", dest="turns")
    args = parser.parse_args()

    result = run_probe(args.base_url, args.turns or DEFAULT_TURNS)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
