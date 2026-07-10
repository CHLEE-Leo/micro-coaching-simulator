from __future__ import annotations

from code_interactive.user_model import (
    DialogueTurn,
    FrontendSessionConfig,
    SimulatedUserProfile,
    SimulatedUserScenario,
    SimulatedUserTurnRequest,
)
from code_interactive.user_model.prompts.roles.simulated_user import (
    render_simulated_user_prompt,
)
from code_interactive.user_model.scenarios import CHAT_SEEDED_CLOSED_LOOP_SCENARIOS
from code_interactive.user_model.scenarios import COMPLEX_STRESS_CLOSED_LOOP_SCENARIOS
from code_interactive.user_model.scenarios import PHASE_TRANSITION_CLOSED_LOOP_SCENARIOS
from code_interactive.user_model.scenarios import STRESS_CLOSED_LOOP_SCENARIOS
from code_interactive.user_model.scenarios import WORKFLOW_REGRESSION_CLOSED_LOOP_SCENARIOS


def test_frontend_session_config_exports_real_start_payload_fields():
    config = FrontendSessionConfig(
        nutrition_goal="lean_protein",
        meal_type="dinner",
        profile=SimulatedUserProfile(
            activity_level="moderate",
            diet_preferences=["vegetarian"],
            allergies=["egg"],
            health_concerns=["diabetes"],
        ),
    )

    payload = config.to_start_payload()

    assert payload["nutrition_goal"] == "lean_protein"
    assert payload["meal_type"] == "dinner"
    assert payload["persona_activity_level"] == "moderate"
    assert payload["persona_diet_preferences"] == ["vegetarian"]
    assert payload["persona_allergies"] == ["egg"]
    assert payload["persona_health_concerns"] == ["diabetes"]
    assert payload["alignment_enabled"] is False
    assert payload["context_tracking"] is True
    assert payload["uncertainty_tracking"] is False


def test_simulated_user_prompt_includes_frontend_profile_and_latest_coach_message():
    scenario = SimulatedUserScenario(
        id="example",
        session_config=FrontendSessionConfig(
            nutrition_goal="one_fourth_carbs",
            meal_type="dinner",
            profile=SimulatedUserProfile(
                allergies=["egg"],
                health_concerns=["diabetes"],
                freeform="The user wants a light dinner.",
            ),
        ),
        initial_reply="I want noodles.",
        user_goal="Keep dinner light while respecting allergies.",
        success_condition="The user should end with a safe feasible dinner.",
    )
    request = SimulatedUserTurnRequest(
        scenario=scenario,
        turns=[
            DialogueTurn(
                user_reply="I want noodles.",
                coach_text="Do you want protein with that?",
            )
        ],
        latest_coach_text="Do you want protein with that?",
    )

    prompt = render_simulated_user_prompt(request)

    assert "nutrition goal: one_fourth_carbs" in prompt
    assert "meal type: dinner" in prompt
    assert "allergies: egg" in prompt
    assert "health concerns: diabetes" in prompt
    assert "The user wants a light dinner." in prompt
    assert "Do you want protein with that?" in prompt
    assert "Keep dinner light while respecting allergies." in prompt


def test_chat_seeded_scenarios_are_typed_and_api_ready():
    assert len(CHAT_SEEDED_CLOSED_LOOP_SCENARIOS) == 5
    for scenario in CHAT_SEEDED_CLOSED_LOOP_SCENARIOS:
        payload = scenario.session_config.to_start_payload()
        assert scenario.id
        assert scenario.initial_reply
        assert scenario.success_condition
        assert payload["nutrition_goal"] == scenario.nutrition_goal
        assert payload["meal_type"] == scenario.meal_type


def test_stress_scenarios_are_diverse_typed_and_api_ready():
    assert len(STRESS_CLOSED_LOOP_SCENARIOS) == 6
    goals = {scenario.nutrition_goal for scenario in STRESS_CLOSED_LOOP_SCENARIOS}
    assert {"lean_protein", "half_fruits_vegetables", "one_fourth_carbs", "drink_water"} <= goals

    for scenario in STRESS_CLOSED_LOOP_SCENARIOS:
        payload = scenario.session_config.to_start_payload()
        assert scenario.id.startswith("stress_")
        assert scenario.initial_reply
        assert scenario.user_goal
        assert scenario.success_condition
        assert payload["nutrition_goal"] == scenario.nutrition_goal
        assert payload["meal_type"] == scenario.meal_type


def test_phase_transition_scenarios_are_typed_and_api_ready():
    assert len(PHASE_TRANSITION_CLOSED_LOOP_SCENARIOS) == 6
    goals = {
        scenario.nutrition_goal
        for scenario in PHASE_TRANSITION_CLOSED_LOOP_SCENARIOS
    }
    assert goals == {"lean_protein", "half_fruits_vegetables", "one_fourth_carbs"}

    for scenario in PHASE_TRANSITION_CLOSED_LOOP_SCENARIOS:
        payload = scenario.session_config.to_start_payload()
        assert scenario.id.startswith("phase_")
        assert scenario.initial_reply
        assert scenario.user_goal
        assert scenario.success_condition
        assert payload["nutrition_goal"] == scenario.nutrition_goal
        assert payload["meal_type"] == scenario.meal_type


def test_workflow_regression_scenarios_are_typed_and_api_ready():
    assert len(WORKFLOW_REGRESSION_CLOSED_LOOP_SCENARIOS) == 6
    goals = {
        scenario.nutrition_goal
        for scenario in WORKFLOW_REGRESSION_CLOSED_LOOP_SCENARIOS
    }
    assert {
        "lean_protein",
        "half_fruits_vegetables",
        "one_fourth_carbs",
    } <= goals

    for scenario in WORKFLOW_REGRESSION_CLOSED_LOOP_SCENARIOS:
        payload = scenario.session_config.to_start_payload()
        assert scenario.id.startswith("workflow_")
        assert scenario.initial_reply
        assert scenario.user_goal
        assert scenario.success_condition
        assert payload["nutrition_goal"] == scenario.nutrition_goal
        assert payload["meal_type"] == scenario.meal_type


def test_complex_stress_scenarios_are_diverse_typed_and_api_ready():
    assert len(COMPLEX_STRESS_CLOSED_LOOP_SCENARIOS) >= 40
    goals = {
        scenario.nutrition_goal
        for scenario in COMPLEX_STRESS_CLOSED_LOOP_SCENARIOS
    }
    assert {
        "lean_protein",
        "half_fruits_vegetables",
        "one_fourth_carbs",
    } <= goals

    profile_allergy_count = 0
    profile_health_count = 0
    fatigue_sensitive_count = 0
    for scenario in COMPLEX_STRESS_CLOSED_LOOP_SCENARIOS:
        payload = scenario.session_config.to_start_payload()
        profile_text = scenario.prompt_profile().lower()
        assert scenario.id.startswith("complex_")
        assert scenario.initial_reply
        assert scenario.user_goal
        assert scenario.success_condition
        assert payload["nutrition_goal"] == scenario.nutrition_goal
        assert payload["meal_type"] == scenario.meal_type
        profile_allergy_count += bool(scenario.session_config.profile.allergies)
        profile_health_count += bool(scenario.session_config.profile.health_concerns)
        fatigue_sensitive_count += any(
            marker in profile_text
            for marker in ("fatigue", "tired", "thinking burden", "long answers")
        )

    assert profile_allergy_count >= 3
    assert profile_health_count >= 4
    assert fatigue_sensitive_count >= 4
