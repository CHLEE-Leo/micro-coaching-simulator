"""Regression tests for router guidance and meal tracking state boundaries."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from code_interactive.agents.agent_config import AgentConfig
from code_interactive.agents.memory.conversation_memory import SharedConversationHistory
from code_interactive.agents.modules.information_seeker import InformationSeeker
import code_interactive.agents.modules.meal_tracker as meal_tracker_module
from code_interactive.agents.modules.meal_assessor import MealAssessor
from code_interactive.agents.modules.meal_tracker import MealTrackerModel


def _config() -> AgentConfig:
    config = AgentConfig()
    config.coach_use_template_guidance = False
    config.max_new_tokens = 64
    config.summarize_max_new_tokens = 256
    config.sampling = "greedy"
    return config


def test_information_seeker_includes_router_instruction_block():
    history = SharedConversationHistory(context_window=5)
    history.update_context_base("User has not decided on lunch yet.")
    seeker = InformationSeeker(
        model=None,
        nutrition_goal="lean_protein",
        meal_type="lunch",
        config=_config(),
    )

    messages = seeker.get_messages(
        history,
        phase="exploration",
        instruction="Ask only one focused question about available cooking equipment.",
    )

    assert "[Router instruction for this turn]" in messages[0]["content"]
    assert "available cooking equipment" in messages[0]["content"]


def test_information_seeker_omits_router_instruction_when_blank():
    history = SharedConversationHistory(context_window=5)
    seeker = InformationSeeker(
        model=None,
        nutrition_goal="lean_protein",
        meal_type="lunch",
        config=_config(),
    )

    messages = seeker.get_messages(history, phase="exploration", instruction="")

    assert "[Router instruction for this turn]" not in messages[0]["content"]


def test_meal_tracker_incremental_messages_use_previous_tracking_state():
    tracker = MealTrackerModel(model=None, config=_config())

    messages = tracker.get_messages(
        "Coach: How about soup?\nUser: I have a microwave",
        prev_tracker_state="[Tracking State]\n- Tentative food items: tomato soup",
    )

    assert "Previous tracking state:" in messages[1]["content"]
    assert "Tentative food items: tomato soup" in messages[1]["content"]
    assert "Previous meal_base" not in messages[1]["content"]


def test_meal_tracker_parses_tracking_state_and_published_meal_base():
    tracker = MealTrackerModel(model=None, config=_config())
    raw_output = """[Tracking State]
- Confirmed food items: chicken breast
- Tentative food items: tomato soup
- Rejected food items: none
- Decision context: microwave available

[Published Meal Base]
- Food items: chicken breast
- Ingredients: chicken breast
- Preparation methods: grilled
- Portions/amounts: 6 oz
- Beverages: none mentioned
- Additional notes: none
"""

    parsed = tracker.parse_tracking_output(raw_output)

    assert "tomato soup" in parsed["tracker_state"]
    assert "microwave available" in parsed["tracker_state"]
    assert "tomato soup" not in parsed["meal_base"]
    assert "Food items: chicken breast" in parsed["meal_base"]


def test_meal_tracker_empty_output_returns_safe_fallback():
    tracker = MealTrackerModel(model=None, config=_config())

    parsed = tracker.parse_tracking_output("")

    assert "Confirmed food items: none" in parsed["tracker_state"]
    assert "Food items: not yet mentioned" in parsed["meal_base"]


def test_meal_tracker_extract_returns_published_meal_base_only(monkeypatch):
    tracker = MealTrackerModel(model=None, config=_config())
    raw_output = """[Tracking State]
- Confirmed food items: chicken breast
- Tentative food items: tomato soup
- Rejected food items: none
- Decision context: microwave available

[Published Meal Base]
- Food items: chicken breast
- Ingredients: chicken breast
- Preparation methods: grilled
- Portions/amounts: 6 oz
- Beverages: none mentioned
- Additional notes: none
"""

    monkeypatch.setattr(
        meal_tracker_module,
        "generate_response",
        lambda *args, **kwargs: raw_output,
    )

    meal_base = tracker.extract("Coach: ...\nUser: ...")

    assert "Food items: chicken breast" in meal_base
    assert "tomato soup" not in meal_base
    assert "microwave available" not in meal_base


def test_tracking_state_can_keep_tentative_food_out_of_alignment_context():
    history = SharedConversationHistory(context_window=5)
    history.update_tracker_state(
        "[Tracking State]\n"
        "- Confirmed food items: none\n"
        "- Tentative food items: tomato soup\n"
        "- Rejected food items: none\n"
        "- Decision context: microwave available"
    )
    history.update_meal_base(
        "- Food items: not yet mentioned\n"
        "- Ingredients: not yet mentioned\n"
        "- Preparation methods: not yet mentioned\n"
        "- Portions/amounts: not yet mentioned\n"
        "- Beverages: none mentioned\n"
        "- Additional notes: not yet mentioned"
    )

    assert "tomato soup" in history.tracker_state
    assert "microwave available" in history.tracker_state
    assert "tomato soup" not in history.to_alignment_context()


def test_assessment_prompt_uses_current_meal_evidence_beyond_published_base():
    history = SharedConversationHistory(context_window=5)
    history.add_turn(0, "What are you thinking of having?", "I'm trying jajangmian and egg-fried rice.")
    history.update_meal_base(
        "- Food items: not yet mentioned\n"
        "- Ingredients: not yet mentioned\n"
        "- Preparation methods: not yet mentioned"
    )
    history.update_tracker_state(
        "[Tracking State]\n"
        "- Confirmed food items: none\n"
        "- Tentative food items: jajangmian, egg-fried rice\n"
        "- Rejected food items: none\n"
        "- Decision context: user is considering dinner"
    )
    history.update_context_base(
        "[Personal Context]\nThe user has an egg allergy and diabetes/prediabetes."
    )
    history.update_interaction_state(
        "Candidate options:\n- jajangmian\n- egg-fried rice\n"
        "Known profile constraints:\n- Allergy constraint: Eggs"
    )

    assessor = MealAssessor("lean_protein", _config())
    messages = assessor.get_messages(
        history=history,
        user_preferences="Allergies: Eggs\nHealth Concerns: Diabetes / Prediabetes",
        recommendation_history=[
            {"turn_idx": 1, "suggestion": "use lean turkey", "target_food": "jajang sauce"}
        ],
    )
    user_content = messages[1]["content"]
    system_content = messages[0]["content"]

    assert "Food items: not yet mentioned" in user_content
    assert "Tentative food items: jajangmian, egg-fried rice" in user_content
    assert "Candidate options:" in user_content
    assert "Allergies: Eggs" in user_content
    assert "Do NOT say no meal items were provided" in system_content


@pytest.mark.parametrize(
    ("raw_output", "expected_present", "expected_absent"),
    [
        (
            """[Tracking State]
- Confirmed food items: chicken breast
- Tentative food items: none
- Rejected food items: none
- Decision context: none

[Published Meal Base]
- Food items: chicken breast
- Ingredients: chicken breast
- Preparation methods: grilled
- Portions/amounts: 6 oz
- Beverages: none mentioned
- Additional notes: none
""",
            "Food items: chicken breast",
            None,
        ),
        (
            """[Tracking State]
- Confirmed food items: none
- Tentative food items: tomato soup
- Rejected food items: none
- Decision context: microwave available

[Published Meal Base]
- Food items: not yet mentioned
- Ingredients: not yet mentioned
- Preparation methods: not yet mentioned
- Portions/amounts: not yet mentioned
- Beverages: none mentioned
- Additional notes: not yet mentioned
""",
            "Food items: not yet mentioned",
            "tomato soup",
        ),
        (
            """[Tracking State]
- Confirmed food items: none
- Tentative food items: none
- Rejected food items: tomato soup
- Decision context: microwave available

[Published Meal Base]
- Food items: not yet mentioned
- Ingredients: not yet mentioned
- Preparation methods: not yet mentioned
- Portions/amounts: not yet mentioned
- Beverages: none mentioned
- Additional notes: user declined tomato soup
""",
            "Additional notes: user declined tomato soup",
            "Food items: tomato soup",
        ),
    ],
)
def test_published_meal_base_tracks_only_confirmed_foods(
    raw_output: str,
    expected_present: str,
    expected_absent: str | None,
):
    tracker = MealTrackerModel(model=None, config=_config())

    parsed = tracker.parse_tracking_output(raw_output)

    assert expected_present in parsed["meal_base"]
    if expected_absent is not None:
        assert expected_absent not in parsed["meal_base"]
