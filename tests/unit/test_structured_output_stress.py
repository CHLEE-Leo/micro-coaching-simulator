from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from code_interactive.agents.agent_config import AgentConfig
from code_interactive.agents.json_output import JSONOutputError, load_json_object
from code_interactive.agents.modules.dialogue_planner import DialoguePlanner
from code_interactive.agents.modules.information_seeker import InformationSeeker
from code_interactive.agents.modules.interaction_state_tracker import InteractionStateTracker
from code_interactive.agents.modules.meal_assessor import MealAssessor
from code_interactive.agents.modules.meal_recommender import MealRecommender


def test_json_loader_survives_prose_fences_and_nested_braces_in_strings():
    raw = (
        "I will explain first.\n"
        "```json\n"
        '{"action":"assess","reasoning":"keep literal braces like } inside text",'
        '"nested":{"value":"ok"}}\n'
        "```\n"
        "A second object should be ignored: {\"action\":\"bad\"}"
    )

    parsed = load_json_object(raw)

    assert parsed["action"] == "assess"
    assert parsed["nested"] == {"value": "ok"}
    assert "}" in parsed["reasoning"]


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "not json",
        "[1, 2, 3]",
        '{"action": "assess"',
    ],
)
def test_json_loader_rejects_non_object_or_broken_outputs(raw):
    with pytest.raises(JSONOutputError):
        load_json_object(raw)


def test_dialogue_planner_recovers_truncated_json_without_losing_action():
    planner = DialoguePlanner("lean_protein", AgentConfig())
    raw = (
        '{"intent_summary":"The user accepted the plan.",'
        '"user_intent":"accepting",'
        '"phase":"confirmation",'
        '"actionability":"settled",'
        '"action":"confirm",'
        '"closure_readiness":"ready_to_close",'
        '"reasoning":"The meal is settled but the JSON is truncated",'
    )

    parsed = planner.parse_output(raw, fallback_phase="exploration")

    assert parsed["action"] == "confirm"
    assert parsed["user_intent"] == "accepting"
    assert parsed["actionability"] == "settled"
    assert parsed["confidence"] == 0.0
    assert "partial planner JSON recovery" in parsed["parse_warning"]


def test_dialogue_planner_normalizes_invalid_followup_phase_and_confidence():
    planner = DialoguePlanner("lean_protein", AgentConfig())
    raw = """
    Here is the plan:
    {
      "intent_summary": "Need to assess before recommending.",
      "user_intent": "informing",
      "phase": "assessment",
      "actionability": "workable",
      "action": "assess",
      "closure_readiness": "actionable",
      "reasoning": "A recommendation should be grounded in assessment.",
      "instruction": "Assess the current meal.",
      "assessment_followup_action": "recommend",
      "assessment_followup_phase": "negotiation",
      "assessment_followup_instruction": "Recommend a compact bundle.",
      "confidence": 7.5
    }
    trailing prose
    """

    parsed = planner.parse_output(raw, fallback_phase="exploration")

    assert parsed["action"] == "assess"
    assert parsed["assessment_followup_action"] == "recommend"
    assert parsed["assessment_followup_phase"] == "recommendation"
    assert parsed["confidence"] == 1.0


def test_meal_assessor_marks_malformed_outputs_as_degraded_not_semantic_summary():
    assessor = MealAssessor("lean_protein", AgentConfig())

    parsed = assessor.parse('{"summary":"ok","strengths":[],"limitations":[],"overall":"perfect"}')

    assert parsed["summary"] == ""
    assert parsed["overall"] == "partially_aligned"
    assert parsed["_degraded"] is True
    assert "invalid overall" in parsed["_parse_error"]
    assert "parse error" not in parsed["summary"].lower()


def test_meal_assessor_retry_recovery_records_telemetry():
    assessor = MealAssessor("lean_protein", AgentConfig())

    parsed = assessor.parse_with_retry(
        base_msgs=[],
        raw_output="not json",
        reinvoke_fn=lambda messages: (
            '{"summary":"Chicken sandwich is workable.",'
            '"strengths":["lean protein"],'
            '"limitations":["vegetables unclear"],'
            '"overall":"partially_aligned"}'
        ),
    )

    assert parsed["summary"] == "Chicken sandwich is workable."
    assert parsed["_degraded"] is False
    assert parsed["_retry_attempted"] is True
    assert parsed["_retry_recovered"] is True


def test_meal_recommender_normalizes_enum_like_fields_under_stress():
    recommender = MealRecommender("lean_protein", AgentConfig())
    raw = """
    ```json
    {
      "recommendation_type": "OPTIMIZE",
      "target_food": "jajangmyeon",
      "suggestion": "Use lean chicken in the sauce.",
      "reasoning": "Improves lean protein.",
      "expected_impact": "EXTREME",
      "options": [
        {
          "option_id": "a",
          "target_food": "sauce",
          "suggestion": "Use lean chicken.",
          "reasoning": "Adds lean protein.",
          "expected_impact": "HIGH"
        },
        {
          "option_id": "b",
          "target_food": "noodles",
          "suggestion": "Keep noodles moderate.",
          "reasoning": "Reduces carb load.",
          "expected_impact": "unknown"
        }
      ]
    }
    ```
    """

    parsed = recommender.parse_output(raw, turn_idx=3)

    assert parsed["recommendation_type"] == "modify"
    assert parsed["expected_impact"] == "low"
    assert parsed["options"][0]["expected_impact"] == "high"
    assert parsed["options"][1]["expected_impact"] == "low"
    assert recommender.recommendation_history[-1]["turn_idx"] == 3


def test_meal_recommender_retry_fallback_remains_non_empty_and_degraded():
    recommender = MealRecommender("lean_protein", AgentConfig())

    parsed = recommender.parse_with_retry(
        base_msgs=[],
        raw_output='{"recommendation_type":"modify","options":[]}',
        reinvoke_fn=lambda messages: "still not json",
        turn_idx=4,
    )

    assert parsed["structured_output_degraded"] is True
    assert parsed["suggestion"]
    assert parsed["options"][0]["suggestion"]
    assert "Parser detail" in parsed["reasoning"]


def test_interaction_state_tracker_ignores_wrong_types_and_preserves_fallback():
    tracker = InteractionStateTracker()
    malformed = "not json"
    fallback = "Answered facts:\n- chicken accepted"

    assert tracker.parse_output(malformed, fallback=fallback) == fallback

    raw = """
    {
      "answered_facts": "scalar should not become a fake list item",
      "open_questions": ["side dish unresolved"],
      "rejected_options": [null, "egg"],
      "unavailable_options": [],
      "safety_conflicted_options": ["egg-fried rice"],
      "user_requested_conflicted_options": [],
      "candidate_options": ["plain shrimp"],
      "accepted_options": ["chicken sandwich"],
      "meal_slots": ["main accepted"],
      "active_issue": "side dish",
      "latest_user_position": "The user wants no egg."
    }
    """
    parsed = tracker.parse_output(raw)

    assert "scalar should not become" not in parsed
    assert "Open questions:\n- side dish unresolved" in parsed
    assert "Rejected options:\n- egg" in parsed
    assert "Safety-conflicted options:\n- egg-fried rice" in parsed
    assert "Latest user position:\n- The user wants no egg." in parsed


def test_information_seeker_parser_keeps_question_template_non_empty_under_garbage():
    seeker = InformationSeeker(
        model=None,
        nutrition_goal="lean_protein",
        meal_type="dinner",
        config=AgentConfig(),
    )

    parsed = seeker._parse_template("What protein options are available?")

    assert parsed["question_type"] == "fallback"
    assert parsed["question_template"] == "What protein options are available?"
