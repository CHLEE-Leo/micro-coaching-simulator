from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from metrics.rule_metrics import evaluate_probe


def test_metrics_detect_confirmation_loop_and_user_repair():
    probe = {
        "test_name": "unit",
        "scenario_set": "stress",
        "results": [
                {
                    "id": "loop_case",
                    "stopped_by": "max_turns",
                    "user_done_signal_turns": [0],
                    "turns": [
                        {
                            "turn_idx": 0,
                            "user_reply": "Sounds good.",
                        "coach_text": "Does this look right?",
                        "status": "active",
                        "effective_action": "confirm",
                        "coach_elapsed_seconds": 1.0,
                    },
                    {
                        "turn_idx": 1,
                        "user_reply": "I already told you yes.",
                        "coach_text": "Are you fully set?",
                        "status": "active",
                            "effective_action": "confirm",
                            "coach_elapsed_seconds": 2.0,
                        },
                        {
                            "turn_idx": 2,
                            "user_reply": "This is the same question again.",
                            "coach_text": "Are you fully set now?",
                            "status": "active",
                            "effective_action": "confirm",
                            "coach_elapsed_seconds": 3.0,
                        },
                    ],
                }
            ],
        }

    report = evaluate_probe(probe)

    assert report["aggregate"]["scenario_count"] == 1
    assert report["aggregate"]["max_turns_count"] == 1
    assert report["aggregate"]["confirmation_loop_count"] == 1
    assert report["aggregate"]["post_done_continuation_count"] == 1
    assert report["aggregate"]["user_repair_signal_count"] == 2
    assert report["latency"]["coach_latency"]["mean"] == 2.0


def test_metrics_treat_clean_completion_as_unflagged():
    probe = {
        "test_name": "unit",
        "scenario_set": "stress",
        "results": [
            {
                "id": "clean_case",
                "stopped_by": "coach",
                "user_done_signal_turns": [0],
                "turns": [
                    {
                        "turn_idx": 0,
                        "user_reply": "That works.",
                        "coach_text": "You are set with the tofu scramble.",
                        "status": "terminated",
                        "effective_action": "close",
                        "coach_elapsed_seconds": 1.0,
                    }
                ],
            }
        ],
    }

    scenario = evaluate_probe(probe)["scenarios"][0]

    assert scenario["confirmation_loop"] is False
    assert scenario["post_done_continuation"] is False
    assert scenario["final_question"] is False
    assert scenario["single_turn_close"] is True
