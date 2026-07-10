"""Run LLM-as-a-user closed-loop scenarios against a FastAPI chatbot server.

This is a manual simulated experiment, not a pytest test. The coach is the
running FastAPI app. The user is a separate LLM actor that reads the latest
coach reply and generates the next user reply.

Example:

    python tests/closed_loop_probe.py \
      --base-url http://127.0.0.1:8121 \
      --output code_interactive/docs/chat_seeded_closed_loop_completion_probe/run.json

    INTERACTIVE_TEST_PACKAGE=code_interactive_v2 python tests/closed_loop_probe.py \
      --base-url http://127.0.0.1:8122 \
      --output code_interactive_v2/docs/chat_seeded_closed_loop_completion_probe/run.json

    python tests/closed_loop_probe.py \
      --base-url http://127.0.0.1:8143 \
      --scenario-set complex_stress \
      --scenario-id complex_finalization_reopening_new_option \
      --scenario-id complex_variant_milk_allergy_smoothie_request
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

_SCRIPT_PATH = Path(__file__).resolve()
_REPO_DIR = _SCRIPT_PATH.parents[1]
if str(_REPO_DIR) not in sys.path:
    sys.path.insert(0, str(_REPO_DIR))


def _target_package() -> str:
    package = os.environ.get("INTERACTIVE_TEST_PACKAGE", "code_interactive").strip()
    if package not in {"code_interactive", "code_interactive_v2"}:
        raise RuntimeError(
            "INTERACTIVE_TEST_PACKAGE must be 'code_interactive' or "
            f"'code_interactive_v2', got {package!r}."
        )
    return package


_PACKAGE = _target_package()
load_model = importlib.import_module(f"{_PACKAGE}.agents.openai_client").load_model
_user_model = importlib.import_module(f"{_PACKAGE}.user_model")
_scenario_module = importlib.import_module(f"{_PACKAGE}.user_model.scenarios")
DialogueTurn = _user_model.DialogueTurn
LLMUserSimulator = _user_model.LLMUserSimulator
SimulatedUserTurnRequest = _user_model.SimulatedUserTurnRequest
UserModelConfig = _user_model.UserModelConfig
CHAT_SEEDED_CLOSED_LOOP_SCENARIOS = getattr(
    _scenario_module,
    "CHAT_SEEDED_CLOSED_LOOP_SCENARIOS",
    [],
)
STRESS_CLOSED_LOOP_SCENARIOS = getattr(
    _scenario_module,
    "STRESS_CLOSED_LOOP_SCENARIOS",
    [],
)
COMPLEX_STRESS_CLOSED_LOOP_SCENARIOS = getattr(
    _scenario_module,
    "COMPLEX_STRESS_CLOSED_LOOP_SCENARIOS",
    [],
)
HUMAN_LIKE_STRESS_CLOSED_LOOP_SCENARIOS = getattr(
    _scenario_module,
    "HUMAN_LIKE_STRESS_CLOSED_LOOP_SCENARIOS",
    [],
)
PHASE_TRANSITION_CLOSED_LOOP_SCENARIOS = getattr(
    _scenario_module,
    "PHASE_TRANSITION_CLOSED_LOOP_SCENARIOS",
    [],
)
WORKFLOW_REGRESSION_CLOSED_LOOP_SCENARIOS = getattr(
    _scenario_module,
    "WORKFLOW_REGRESSION_CLOSED_LOOP_SCENARIOS",
    [],
)


SCENARIO_SETS = {
    "chat_seeded": CHAT_SEEDED_CLOSED_LOOP_SCENARIOS,
    "stress": STRESS_CLOSED_LOOP_SCENARIOS,
    "complex_stress": COMPLEX_STRESS_CLOSED_LOOP_SCENARIOS,
    "human_like_stress": HUMAN_LIKE_STRESS_CLOSED_LOOP_SCENARIOS,
    "phase_transition": PHASE_TRANSITION_CLOSED_LOOP_SCENARIOS,
    "workflow_regression": WORKFLOW_REGRESSION_CLOSED_LOOP_SCENARIOS,
    "all": (
        CHAT_SEEDED_CLOSED_LOOP_SCENARIOS
        + STRESS_CLOSED_LOOP_SCENARIOS
        + COMPLEX_STRESS_CLOSED_LOOP_SCENARIOS
        + HUMAN_LIKE_STRESS_CLOSED_LOOP_SCENARIOS
        + PHASE_TRANSITION_CLOSED_LOOP_SCENARIOS
        + WORKFLOW_REGRESSION_CLOSED_LOOP_SCENARIOS
    ),
}


def _post(base_url: str, path: str, payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body), time.perf_counter() - started


def _error_record(exc: BaseException) -> dict[str, Any]:
    if isinstance(exc, urllib.error.HTTPError):
        try:
            body = exc.read().decode("utf-8")
        except Exception:
            body = ""
        return {
            "type": type(exc).__name__,
            "status_code": exc.code,
            "reason": exc.reason,
            "body": body,
        }
    return {"type": type(exc).__name__, "message": str(exc)}


def _coach_text(turn_response: dict[str, Any]) -> str:
    messages = turn_response.get("coach_messages") or []
    if messages:
        return "\n\n".join(str(message) for message in messages if str(message).strip())
    return str(turn_response.get("coach_question") or "")


def _compact_turn(
    *,
    turn_idx: int,
    user_reply: str,
    turn_response: dict[str, Any],
    coach_elapsed: float,
    user_sim: dict[str, Any] | None,
    user_elapsed: float | None,
) -> dict[str, Any]:
    metadata = turn_response.get("engine_metadata") or {}
    latency = turn_response.get("latency") or {}
    decision = turn_response.get("dialogue_plan") or {}
    planning_policy = metadata.get("planning_policy") or metadata.get("intent_policy") or {}
    post_assessment = metadata.get("post_assessment_decision") or {}
    commitment_gate = metadata.get("commitment_gate") or {}
    post_assessment_gate = metadata.get("post_assessment_gate") or {}
    return {
        "turn_idx": turn_idx,
        "user_reply": user_reply,
        "coach_text": _coach_text(turn_response),
        "coach_messages": turn_response.get("coach_messages"),
        "status": turn_response.get("status"),
        "phase": turn_response.get("phase"),
        "planned_action": planning_policy.get("planned_action") or decision.get("action"),
        "effective_action": planning_policy.get("effective_action"),
        "planning_override": planning_policy.get("override"),
        "commitment_gate": commitment_gate.get("gate"),
        "post_assessment_gate": post_assessment_gate.get("gate"),
        "finalization_style": metadata.get("finalization_style"),
        "post_assessment_action": post_assessment.get("action"),
        "user_intent": decision.get("user_intent"),
        "actionability": decision.get("actionability") or planning_policy.get("actionability"),
        "intent_summary": decision.get("intent_summary"),
        "planner_reasoning": decision.get("reasoning"),
        "interaction_state": metadata.get("interaction_state"),
        "coach_elapsed_seconds": round(coach_elapsed, 3),
        "engine_total_seconds": latency.get("total_seconds"),
        "module_call_count": latency.get("module_call_count"),
        "module_totals": latency.get("module_totals"),
        "next_user_simulation": user_sim,
        "user_sim_elapsed_seconds": (
            round(user_elapsed, 3) if isinstance(user_elapsed, (int, float)) else None
        ),
    }


def _to_dialogue_turns(
    turns: list[dict[str, Any]],
    *,
    current_user_reply: str,
    current_coach_text: str,
) -> list[DialogueTurn]:
    """Convert runner turn records into the simulated-user dialogue contract."""
    dialogue_turns = [
        DialogueTurn(
            user_reply=str(turn.get("user_reply", "")),
            coach_text=str(turn.get("coach_text", "")),
        )
        for turn in turns
    ]
    dialogue_turns.append(
        DialogueTurn(user_reply=current_user_reply, coach_text=current_coach_text)
    )
    return dialogue_turns


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    coach_elapsed: list[float] = []
    user_elapsed: list[float] = []
    final_status_counts: dict[str, int] = {}
    stopped_by_counts: dict[str, int] = {}
    finalization_styles: dict[str, int] = {}
    scenario_summaries: list[dict[str, Any]] = []
    for result in results:
        turns = result.get("turns", [])
        final_status = turns[-1].get("status") if turns else "no_turns"
        if result.get("error"):
            final_status = "error"
        final_status_counts[final_status] = final_status_counts.get(final_status, 0) + 1
        stopped_by = str(result.get("stopped_by") or "unknown")
        stopped_by_counts[stopped_by] = stopped_by_counts.get(stopped_by, 0) + 1
        action_flow = []
        for turn in turns:
            if isinstance(turn.get("coach_elapsed_seconds"), (int, float)):
                coach_elapsed.append(float(turn["coach_elapsed_seconds"]))
            if isinstance(turn.get("user_sim_elapsed_seconds"), (int, float)):
                user_elapsed.append(float(turn["user_sim_elapsed_seconds"]))
            action_flow.append(turn.get("effective_action") or "")
            style = turn.get("finalization_style")
            if style:
                finalization_styles[style] = finalization_styles.get(style, 0) + 1
        scenario_summaries.append(
            {
                "id": result.get("id"),
                "source_chats": result.get("source_chats", []),
                "turns_completed": len(turns),
                "final_status": final_status,
                "stopped_by": stopped_by,
                "user_done_signal_turns": result.get("user_done_signal_turns", []),
                "effective_action_flow": action_flow,
                "final_coach_text": turns[-1].get("coach_text") if turns else "",
            }
        )

    def latency(values: list[float]) -> dict[str, float | int | None]:
        return {
            "turn_count": len(values),
            "mean": round(statistics.mean(values), 3) if values else None,
            "median": round(statistics.median(values), 3) if values else None,
            "min": round(min(values), 3) if values else None,
            "max": round(max(values), 3) if values else None,
            "p90": (
                round(statistics.quantiles(values, n=10)[8], 3)
                if len(values) >= 10
                else None
            ),
        }

    return {
        "coach_latency": latency(coach_elapsed),
        "user_sim_latency": latency(user_elapsed),
        "final_status_counts": final_status_counts,
        "stopped_by_counts": stopped_by_counts,
        "finalization_style_counts": finalization_styles,
        "scenario_summaries": scenario_summaries,
    }


def _write_checkpoint(
    *,
    output_path: Path | None,
    base_url: str,
    started_at: str,
    test_name: str,
    user_model: str,
    max_turns: int,
    scenario_set: str,
    scenario_count: int,
    selected_scenario_ids: list[str],
    results: list[dict[str, Any]],
) -> None:
    """Persist partial closed-loop results after every completed turn."""
    if output_path is None:
        return
    completed = sum(
        1
        for result in results
        if result.get("stopped_by") not in {"in_progress", None}
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "base_url": base_url,
                "started_at": started_at,
                "test_name": test_name,
                "user_model": user_model,
                "max_turns_per_scenario": max_turns,
                "scenario_set": scenario_set,
                "scenario_count": scenario_count,
                "selected_scenario_ids": selected_scenario_ids,
                "recorded_scenario_count": len(results),
                "completed_scenario_count": completed,
                "results": results,
                "summary": _summarize(results),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def run_closed_loop(
    *,
    base_url: str,
    max_turns: int,
    user_model: str,
    user_max_tokens: int,
    scenario_set: str = "chat_seeded",
    scenario_ids: list[str] | None = None,
    scenario_limit: int = 0,
    output_path: Path | None = None,
) -> dict[str, Any]:
    scenarios = list(SCENARIO_SETS[scenario_set])
    selected_scenario_ids = list(scenario_ids or [])
    if selected_scenario_ids:
        selected = set(selected_scenario_ids)
        available = {scenario.id for scenario in scenarios}
        missing = sorted(selected - available)
        if missing:
            raise ValueError(
                "Unknown scenario id(s) for "
                f"{scenario_set!r}: {', '.join(missing)}"
            )
        scenarios = [
            scenario for scenario in scenarios if scenario.id in selected
        ]
    if scenario_limit > 0:
        scenarios = scenarios[:scenario_limit]
    test_name = f"{scenario_set}_closed_loop_probe"
    user_client = load_model(user_model)
    user_simulator = LLMUserSimulator(
        user_client,
        config=UserModelConfig(model=user_model, max_new_tokens=user_max_tokens),
    )
    results: list[dict[str, Any]] = []
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    for scenario in scenarios:
        start_payload = scenario.session_config.to_start_payload()
        try:
            start_response, start_elapsed = _post(
                base_url,
                "/api/session/start",
                start_payload,
            )
        except Exception as exc:
            results.append(
                {
                    "id": scenario.id,
                    "source_chats": list(scenario.source_chats),
                    "goal": scenario.nutrition_goal,
                    "meal_type": scenario.meal_type,
                    "turns": [],
                    "error": {"stage": "start", "details": _error_record(exc)},
                }
            )
            continue
        session_id = start_response["session_id"]
        turns: list[dict[str, Any]] = []
        next_user_reply = scenario.initial_reply
        stopped_by = "max_turns"
        user_done_signal_turns: list[int] = []
        for turn_idx in range(max_turns):
            sent_user_reply = next_user_reply
            try:
                turn_response, coach_elapsed = _post(
                    base_url,
                    f"/api/session/{session_id}/turn",
                    {"user_reply": sent_user_reply},
                )
            except Exception as exc:
                turns.append(
                    {
                        "turn_idx": turn_idx,
                        "user_reply": sent_user_reply,
                        "status": "error",
                        "error": _error_record(exc),
                    }
                )
                stopped_by = "error"
                break

            user_sim = None
            user_elapsed = None
            status = turn_response.get("status")
            if status == "active":
                started = time.perf_counter()
                user_turn = user_simulator.generate_turn(
                    SimulatedUserTurnRequest(
                        scenario=scenario,
                        turns=_to_dialogue_turns(
                            turns,
                            current_user_reply=sent_user_reply,
                            current_coach_text=_coach_text(turn_response),
                        ),
                        latest_coach_text=_coach_text(turn_response),
                    )
                )
                user_elapsed = time.perf_counter() - started
                user_sim = user_turn.to_dict()
                next_user_reply = user_turn.reply
                if user_turn.done:
                    user_done_signal_turns.append(turn_idx)

            turns.append(
                _compact_turn(
                    turn_idx=turn_idx,
                    user_reply=sent_user_reply,
                    turn_response=turn_response,
                    coach_elapsed=coach_elapsed,
                    user_sim=user_sim,
                    user_elapsed=user_elapsed,
                )
            )
            partial_result = {
                "id": scenario.id,
                "source_chats": list(scenario.source_chats),
                "goal": scenario.nutrition_goal,
                "meal_type": scenario.meal_type,
                "session_id": session_id,
                "start_elapsed_seconds": round(start_elapsed, 3),
                "first_question": start_response.get("first_question"),
                "user_profile": scenario.prompt_profile(),
                "success_condition": scenario.success_condition,
                "stopped_by": "in_progress" if status == "active" else "coach",
                "user_done_signal_turns": list(user_done_signal_turns),
                "turns": list(turns),
            }
            _write_checkpoint(
                output_path=output_path,
                base_url=base_url,
                started_at=started_at,
                test_name=test_name,
                user_model=user_model,
                max_turns=max_turns,
                scenario_set=scenario_set,
                scenario_count=len(scenarios),
                selected_scenario_ids=selected_scenario_ids,
                results=results + [partial_result],
            )

            if status != "active":
                stopped_by = "coach"
                break

        result = {
            "id": scenario.id,
            "source_chats": list(scenario.source_chats),
            "goal": scenario.nutrition_goal,
            "meal_type": scenario.meal_type,
            "session_id": session_id,
            "start_elapsed_seconds": round(start_elapsed, 3),
            "first_question": start_response.get("first_question"),
            "user_profile": scenario.prompt_profile(),
            "success_condition": scenario.success_condition,
            "stopped_by": stopped_by,
            "user_done_signal_turns": user_done_signal_turns,
            "turns": turns,
        }
        results.append(result)
        _write_checkpoint(
            output_path=output_path,
            base_url=base_url,
            started_at=started_at,
            test_name=test_name,
            user_model=user_model,
            max_turns=max_turns,
            scenario_set=scenario_set,
            scenario_count=len(scenarios),
            selected_scenario_ids=selected_scenario_ids,
            results=results,
        )
    return {
        "base_url": base_url,
        "started_at": started_at,
        "test_name": test_name,
        "user_model": user_model,
        "max_turns_per_scenario": max_turns,
        "scenario_set": scenario_set,
        "scenario_count": len(scenarios),
        "selected_scenario_ids": selected_scenario_ids,
        "completed_scenario_count": len(results),
        "results": results,
        "summary": _summarize(results),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--user-model", default="gpt-5.4-mini")
    parser.add_argument("--user-max-tokens", type=int, default=180)
    parser.add_argument(
        "--scenario-set",
        choices=sorted(SCENARIO_SETS),
        default="chat_seeded",
    )
    parser.add_argument(
        "--scenario-id",
        action="append",
        default=[],
        help="Run only the selected scenario id. Can be passed multiple times.",
    )
    parser.add_argument(
        "--scenario-limit",
        type=int,
        default=0,
        help="Run only the first N scenarios after optional id filtering.",
    )
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    output_path = Path(args.output) if args.output else None
    result = run_closed_loop(
        base_url=args.base_url,
        max_turns=args.max_turns,
        user_model=args.user_model,
        user_max_tokens=args.user_max_tokens,
        scenario_set=args.scenario_set,
        scenario_ids=args.scenario_id,
        scenario_limit=args.scenario_limit,
        output_path=output_path,
    )
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
