from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from code_interactive.agents import (
        AssistantReply,
        ChatMessage,
        CoachingState,
        CoachingTurnRequest,
        CoachingTurnResult,
        ConversationEngine,
        UserProfileContext,
        build_opening_message,
    )
    from code_interactive.agents.agent_config import AgentConfig
    from code_interactive.agents.history_adapter import build_shared_history
except ModuleNotFoundError:
    from app.agents import (
        AssistantReply,
        ChatMessage,
        CoachingState,
        CoachingTurnRequest,
        CoachingTurnResult,
        ConversationEngine,
        UserProfileContext,
        build_opening_message,
    )
    from app.agents.agent_config import AgentConfig
    from app.agents.history_adapter import build_shared_history


def test_opening_message_uses_profile_name():
    assert (
        build_opening_message(UserProfileContext(name="Alice"))
        == "Hi, Alice. How can I help you with your meal today?"
    )


def test_history_adapter_injects_opening_for_user_first_chat():
    adapted = build_shared_history(
        [],
        "I had chicken and rice.",
        context_window=10,
        opening_message="Hi, Alice. How can I help you with your meal today?",
    )

    assert adapted.turn_idx == 0
    assert adapted.opening_used == "Hi, Alice. How can I help you with your meal today?"
    assert adapted.history.to_plain_text() == (
        "Coach: Hi, Alice. How can I help you with your meal today?\n"
        "User: I had chicken and rice."
    )


def test_history_adapter_folds_consecutive_assistant_bubbles():
    adapted = build_shared_history(
        [
            ChatMessage("assistant", "Your meal has a good protein base."),
            ChatMessage("assistant", "Could you tell me about vegetables?"),
        ],
        "There was a small salad.",
        context_window=10,
    )

    assert adapted.history.to_plain_text() == (
        "Coach: Your meal has a good protein base.\n\n"
        "Could you tell me about vegetables?\n"
        "User: There was a small salad."
    )


def test_turn_result_primary_message_is_last_bubble():
    result = CoachingTurnResult(
        assistant_messages=[
            AssistantReply("First bubble", kind="assessment"),
            AssistantReply("Second bubble", kind="question"),
        ],
        state=CoachingState(),
    )

    assert result.primary_message is not None
    assert result.primary_message.content == "Second bubble"


def test_turn_request_keeps_meal_app_style_flat_history():
    request = CoachingTurnRequest(
        current_message="What should I change?",
        history=[
            ChatMessage("assistant", "Hi. How can I help you with your meal today?"),
            ChatMessage("user", "I had fried chicken."),
        ],
        profile=UserProfileContext(name="Alice"),
    )

    assert request.history[0].role == "assistant"
    assert request.current_message == "What should I change?"


def test_structured_decision_modules_do_not_request_api_reasoning_summary():
    try:
        from code_interactive.web_app_config import WebAppConfig
    except ModuleNotFoundError:
        from app.web_app_config import WebAppConfig

    config = WebAppConfig()

    for module in ("dialogue_planner", "alignment_estimator", "certainty_estimator"):
        assert config.resolve_reasoning_effort(module) == "none"
        assert config.resolve_reasoning_summary(module) is None


def test_openai_client_extracts_text_from_nested_response_output():
    try:
        from code_interactive.agents.openai_client import OpenAIClient
    except ModuleNotFoundError:
        from app.agents.openai_client import OpenAIClient

    class Response:
        output_text = ""
        output = [
            {
                "content": [
                    {
                        "type": "output_text",
                        "text": '{"summary":"ok"}',
                    }
                ]
            }
        ]

    assert OpenAIClient._extract_output_text(Response()) == '{"summary":"ok"}'


def test_post_assessment_gate_closes_after_explicit_stop_boundary():
    interaction_state = (
        "Accepted options:\n"
        "- chicken sandwich\n\n"
        "Open questions:\n"
        "- none\n\n"
        "Latest user position:\n"
        "- No, I don't want anything more."
    )

    metadata = {"planning_policy": {"actionability": "workable"}}
    decision = ConversationEngine._gate_post_assessment_decision(
        post_decision={
            "action": "confirm",
            "accepted_phase": "confirmation",
            "instruction": "Confirm before closing.",
        },
        initial_decision={"actionability": "workable"},
        prior_state=CoachingState(),
        interaction_state=interaction_state,
        user_intent="rejecting",
        metadata=metadata,
    )

    assert decision["action"] == "close"
    assert decision["accepted_phase"] == "finalization"
    assert (
        metadata["post_assessment_gate"]["gate"]
        == "post_assessment_stop_boundary_confirm_to_close"
    )


def test_post_assessment_gate_closes_after_answered_confirmation():
    interaction_state = (
        "Accepted options:\n"
        "- tofu scramble\n\n"
        "Latest user position:\n"
        "- Yes, that plan is set.\n\n"
        "Active issue:\n"
        "- Assess the user's current commitment and avoid repeating settled choices."
    )

    metadata = {"planning_policy": {"actionability": "settled"}}
    decision = ConversationEngine._gate_post_assessment_decision(
        post_decision={
            "action": "confirm",
            "accepted_phase": "confirmation",
            "instruction": "Confirm before closing.",
        },
        initial_decision={"actionability": "settled"},
        prior_state=CoachingState(phase="confirmation"),
        interaction_state=interaction_state,
        user_intent="accepting",
        metadata=metadata,
    )

    assert decision["action"] == "close"
    assert decision["accepted_phase"] == "finalization"
    assert (
        metadata["post_assessment_gate"]["gate"]
        == "post_assessment_confirmed_plan_closed"
    )


def test_low_effort_meal_fatigue_is_not_dialogue_fatigue():
    assert not ConversationEngine._is_dialogue_fatigue_or_repetition_complaint(
        "i'm tired and might just do instant ramen with leftover rice."
    )
    assert ConversationEngine._is_dialogue_fatigue_or_repetition_complaint(
        "i'm tired of answering the same question again."
    )


def test_latest_safe_replacement_resolves_historical_allergy_conflict():
    profile = UserProfileContext(allergies=["egg"])
    request = CoachingTurnRequest(
        current_message=(
            "Please finalize the egg-free tofu omelet with spinach and black beans."
        ),
        profile=profile,
    )

    conflict = ConversationEngine._detect_profile_constraint_conflict(
        request=request,
        current_message=request.current_message,
        meal_base=(
            "Earlier topic: the user initially asked for an omelet-style breakfast. "
            "Current plan: egg-free tofu omelet with vegetables."
        ),
    )

    assert conflict is None


def test_explicit_conflicted_food_request_still_triggers_allergy_conflict():
    profile = UserProfileContext(allergies=["egg"])
    request = CoachingTurnRequest(
        current_message="I know about the allergy, but I still want eggs.",
        profile=profile,
    )

    conflict = ConversationEngine._detect_profile_constraint_conflict(
        request=request,
        current_message=request.current_message,
        meal_base="Current topic: breakfast.",
    )

    assert conflict is not None
    assert conflict["constraint"] == "egg"


def test_health_concern_mentions_do_not_trigger_hard_safety_conflict():
    profile = UserProfileContext(extra={"health_concerns": ["high cholesterol"]})
    request = CoachingTurnRequest(
        current_message=(
            "Please keep this cholesterol-conscious with low-fat cottage cheese "
            "and minimal oil."
        ),
        profile=profile,
    )

    conflict = ConversationEngine._detect_profile_constraint_conflict(
        request=request,
        current_message=request.current_message,
        meal_base="Current plan: tofu breakfast optimized for high cholesterol.",
    )

    assert conflict is None


def test_broad_cuisine_is_not_assessable_meal_anchor():
    assert not ConversationEngine._has_assessable_meal_anchor(
        meal_base="- Food items: French food",
        interaction_data={"candidate_options": ["French food"]},
        current_message="Something French food.",
    )


def test_filled_sandwich_is_assessable_meal_anchor():
    assert ConversationEngine._has_assessable_meal_anchor(
        meal_base="- Food items: ham and cheese sandwich, water",
        interaction_data={"accepted_options": ["ham and cheese sandwich"]},
        current_message="ham and cheese",
    )


def test_exploration_sufficiency_does_not_redirect_broad_cuisine():
    engine = ConversationEngine(lambda **_: "")
    gate = engine._apply_exploration_sufficiency_gate(
        action="inquire",
        phase="exploration",
        decision={
            "action": "inquire",
            "actionability": "workable",
            "instruction": "Ask what French dish they are considering.",
        },
        request=CoachingTurnRequest(current_message="Something French food."),
        prior_state=CoachingState(),
        current_message="Something French food.",
        meal_base="- Food items: French food",
        interaction_state=(
            "Answered facts:\n"
            "- User is interested in French food.\n"
            "Open questions:\n"
            "- Which French dish the user will choose.\n"
            "Candidate options:\n"
            "- French food\n"
            "Active issue:\n"
            "- Clarify the specific French dish."
        ),
        user_intent="informing",
    )

    assert gate["applied"] is False
    assert gate["action"] == "inquire"


def test_exploration_sufficiency_rewrites_recommend_followup_instruction():
    engine = ConversationEngine(lambda **_: "")
    gate = engine._apply_exploration_sufficiency_gate(
        action="inquire",
        phase="exploration",
        decision={
            "action": "inquire",
            "actionability": "workable",
            "closure_readiness": "actionable",
            "instruction": "Ask about condiments.",
            "assessment_followup_action": "recommend",
            "assessment_followup_phase": "recommendation",
            "assessment_followup_instruction": "Ask which changes they want.",
        },
        request=CoachingTurnRequest(
            current_message="Turkey sandwich with lean ham and water."
        ),
        prior_state=CoachingState(),
        current_message="Turkey sandwich with lean ham and water.",
        meal_base="- Food items: turkey sandwich with lean ham, water",
        interaction_state=(
            "Answered facts:\n"
            "- User has a turkey sandwich with lean ham and water.\n"
            "Open questions:\n"
            "- Optional condiment detail.\n"
            "Accepted options:\n"
            "- turkey sandwich with lean ham\n"
            "Active issue:\n"
            "- Make the sandwich fit the lean protein goal."
        ),
        user_intent="informing",
    )

    assert gate["applied"] is True
    assert gate["action"] == "assess"
    instruction = gate["decision"]["assessment_followup_instruction"]
    assert "Recommend one concise default bundle" in instruction
    assert "Ask which changes" not in instruction


def test_profile_seeded_suggestion_request_skips_mood_question():
    engine = ConversationEngine(lambda **_: "")
    request = CoachingTurnRequest(
        current_message="Please give me breakfast suggestions.",
        nutrition_goal="lean_protein",
        meal_type="breakfast",
        profile=UserProfileContext(
            allergies=["egg"],
            extra={"health_concerns": ["high cholesterol"]},
        ),
    )

    gate = engine._apply_exploration_sufficiency_gate(
        action="inquire",
        phase="exploration",
        decision={
            "action": "inquire",
            "actionability": "insufficient",
            "instruction": "Ask what breakfast the user is in the mood for.",
        },
        request=request,
        prior_state=CoachingState(),
        current_message=request.current_message,
        meal_base="- Food items: not yet mentioned",
        interaction_state=(
            "Answered facts:\n"
            "- Allergy constraint: egg\n"
            "Open questions:\n"
            "- What breakfast the user wants.\n"
            "Latest user position:\n"
            "- Please give me breakfast suggestions.\n"
            "Known profile constraints:\n"
            "- Allergy constraint: egg\n"
            "- Health concern: high cholesterol"
        ),
        user_intent="inquiring",
    )

    assert gate["applied"] is True
    assert gate["action"] == "assess"
    assert gate["decision"]["assessment_followup_action"] == "recommend"
    assert "exactly one recommendation option" in gate["decision"][
        "assessment_followup_instruction"
    ]


def test_assessment_saturation_closes_deferred_final_decision():
    engine = ConversationEngine(lambda **_: "")
    decision = {
        "action": "assess",
        "accepted_phase": "assessment",
        "actionability": "workable",
        "closure_readiness": "actionable",
    }

    gate = engine._apply_assessment_saturation_gate(
        action="assess",
        phase="assessment",
        decision=decision,
        prior_state=CoachingState(
            phase="recommendation",
            recommendation_history=({"suggestion": "produce base"},),
        ),
        current_message=(
            "That base works, but I'm still comparing soup or chicken and "
            "I'm not ready to finalize the full dinner yet."
        ),
        interaction_state=(
            "Accepted options:\n"
            "- lettuce, carrots, and apple base\n"
            "Latest user position:\n"
            "- That base works, but I'm still comparing soup or chicken and "
            "I'm not ready to finalize the full dinner yet."
        ),
        user_intent="informing",
        actionability="workable",
        closure_readiness="actionable",
    )

    assert gate["applied"] is True
    assert gate["action"] == "close"
    assert gate["metadata"]["gate"] == (
        "assessment_saturation_to_deferred_decision_close"
    )


def test_response_generator_repairs_perspective_and_decision_menu_tail():
    from code_interactive.agents.modules.response_generator import ResponseGenerator

    text = (
        "I'd keep the sandwich simple:\n"
        "- Use lean ham.\n"
        "- Keep cheese modest.\n"
        "Which of these adjustments do you want to keep, skip, or change?"
    )

    repaired = ResponseGenerator._repair_perspective_and_register(text)

    assert repaired.startswith("Keep the sandwich simple")
    assert "Which of these" not in repaired
    assert "keep, skip, or change" not in repaired

    handoff_text = (
        "Makes sense — you want to keep the main open for now.\n\n"
        "What would you like to do next?\n"
        "- Compare chicken, pasta, and soup briefly\n"
        "- Get one concrete main suggestion\n"
        "- Pause here and keep the produce base"
    )
    handoff_repaired = ResponseGenerator._repair_perspective_and_register(
        handoff_text
    )
    assert "What would you like to do next" not in handoff_repaired
    assert "Get one concrete main suggestion" not in handoff_repaired

    answer_text = (
        "Try plain nonfat Greek yogurt with berries; it is quick, egg-free, "
        "and gives you solid lean protein. Want to go with that?"
    )
    answer_repaired = ResponseGenerator._repair_perspective_and_register(answer_text)
    assert "Want to go with that" not in answer_repaired


def test_response_generator_repairs_confirmation_add_change_tail():
    from code_interactive.agents.modules.response_generator import ResponseGenerator

    text = (
        "Your plan is noodles with chicken and vegetables. "
        "Does that look right, or is there anything you want to add or change before we wrap up?"
    )

    repaired = ResponseGenerator._repair_perspective_and_register(text)

    assert "add or change" not in repaired
    assert not repaired.endswith("?")
    assert repaired.endswith("If that is accurate, I can wrap it there.")


def test_dialogue_state_estimators_are_opt_in_by_default():
    request = CoachingTurnRequest(current_message="I had chicken and rice.")

    assert request.enable_alignment is False
    assert request.enable_certainty is False

    try:
        from code_interactive.app import StartSessionRequest
    except ModuleNotFoundError:
        from app.app import StartSessionRequest

    api_request = StartSessionRequest(nutrition_goal="lean_protein")
    assert api_request.alignment_enabled is False
    assert api_request.uncertainty_tracking is None


def test_conversation_engine_smoke_with_fake_llm():
    class FakeLLM:
        def generate(self, *, module, messages, mode):
            if module == "meal_tracker":
                return (
                    "[Tracking State]\n"
                    "- Confirmed food items: chicken and rice\n"
                    "- Tentative food items: none\n"
                    "- Rejected food items: none\n"
                    "- Decision context: none\n\n"
                    "[Published Meal Base]\n"
                    "- Food items: chicken and rice\n"
                    "- Ingredients: chicken, rice\n"
                    "- Preparation methods: not yet mentioned\n"
                    "- Portions/amounts: not yet mentioned\n"
                    "- Beverages: none mentioned\n"
                    "- Additional notes: not yet mentioned"
                )
            if module == "context_tracker":
                return "The user is discussing a chicken and rice meal."
            if module == "interaction_tracker":
                return (
                    '{"answered_facts": ["the user had chicken and rice"], '
                    '"open_questions": ["chicken preparation"], '
                    '"rejected_options": [], '
                    '"unavailable_options": [], '
                    '"accepted_options": [], '
                    '"latest_user_position": "The user described chicken and rice."}'
                )
            if module == "alignment_estimator":
                return '{"answer": "0", "reasoning": "More details are needed."}'
            if module == "dialogue_planner":
                return (
                    '{"intent_summary": "The user shared their meal.", '
                    '"user_intent": "informing", '
                    '"phase": "exploration", '
                    '"action": "inquire", '
                    '"reasoning": "Need preparation details.", '
                    '"instruction": "Ask about preparation.", '
                    '"assessment_followup_action": "", '
                    '"assessment_followup_phase": "", '
                    '"assessment_followup_instruction": "", '
                    '"confidence": 0.82}'
                )
            if module == "info_seeker":
                return (
                    '{"question_type": "preparation", '
                    '"target": "chicken", '
                    '"reasoning": "Preparation affects lean protein alignment.", '
                    '"question_template": "How was the chicken prepared?"}'
                )
            if module == "response_generator":
                return "How was the chicken prepared?"
            raise AssertionError(f"Unexpected module: {module}")

    engine = ConversationEngine(FakeLLM().generate)
    result = engine.generate_chat_replies(
        CoachingTurnRequest(
            current_message="I had chicken and rice.",
            profile=UserProfileContext(name="Alice"),
            enable_guardrail=False,
        )
    )

    assert [m.content for m in result.assistant_messages] == [
        "How was the chicken prepared?"
    ]
    assert result.state.phase == "exploration"
    assert result.state.status == "active"
    assert "chicken and rice" in result.state.meal_base
    assert "chicken preparation" in result.state.interaction_state
    assert result.state.last_user_intent == "informing"
    assert result.state.stall_count == 0
    assert result.state.recommendation_rejection_count == 0
    assert result.metadata["intent_policy"]["user_intent"] == "informing"
    assert result.metadata["latency"]["module_call_count"] > 0
    assert "response_generator" in result.metadata["latency"]["module_totals"]
    assert result.primary_message is result.assistant_messages[-1]


def test_interaction_tracker_can_be_disabled():
    class FakeLLM:
        def generate(self, *, module, messages, mode):
            if module == "interaction_tracker":
                raise AssertionError("interaction_tracker should be disabled")
            if module == "meal_tracker":
                return (
                    "[Tracking State]\n"
                    "- Confirmed food items: chicken and rice\n\n"
                    "[Published Meal Base]\n"
                    "- Food items: chicken and rice"
                )
            if module == "context_tracker":
                return "The user is discussing chicken and rice."
            if module == "alignment_estimator":
                return '{"answer": "0", "reasoning": "More detail needed."}'
            if module == "dialogue_planner":
                return (
                    '{"intent_summary": "The user shared a meal.", '
                    '"user_intent": "informing", '
                    '"phase": "exploration", '
                    '"action": "inquire", '
                    '"reasoning": "Need preparation details.", '
                    '"instruction": "Ask preparation.", '
                    '"assessment_followup_action": "", '
                    '"assessment_followup_phase": "", '
                    '"assessment_followup_instruction": "", '
                    '"confidence": 0.8}'
                )
            if module == "info_seeker":
                return (
                    '{"question_type": "preparation", '
                    '"target": "chicken", '
                    '"reasoning": "Preparation matters.", '
                    '"question_template": "How was the chicken prepared?"}'
                )
            if module == "response_generator":
                return "How was the chicken prepared?"
            raise AssertionError(f"Unexpected module: {module}")

    result = ConversationEngine(
        FakeLLM().generate,
        config=AgentConfig(use_interaction_tracker=False),
    ).generate_chat_replies(
        CoachingTurnRequest(
            current_message="I had chicken and rice.",
            enable_guardrail=False,
        )
    )

    assert result.state.interaction_state == ""
    assert "interaction_tracker_output" not in result.metadata


def test_alignment_estimator_can_be_disabled():
    class FakeLLM:
        def generate(self, *, module, messages, mode):
            if module == "alignment_estimator":
                raise AssertionError("alignment_estimator should be disabled")
            if module == "meal_tracker":
                return (
                    "[Tracking State]\n"
                    "- Confirmed food items: chicken and rice\n\n"
                    "[Published Meal Base]\n"
                    "- Food items: chicken and rice"
                )
            if module == "context_tracker":
                return "The user is discussing chicken and rice."
            if module == "interaction_tracker":
                return "{}"
            if module == "dialogue_planner":
                return (
                    '{"intent_summary": "The user shared a meal.", '
                    '"user_intent": "informing", '
                    '"phase": "exploration", '
                    '"actionability": "insufficient", '
                    '"action": "inquire", '
                    '"reasoning": "Need one useful detail.", '
                    '"instruction": "Ask preparation.", '
                    '"assessment_followup_action": "", '
                    '"assessment_followup_phase": "", '
                    '"assessment_followup_instruction": "", '
                    '"confidence": 0.8}'
                )
            if module == "info_seeker":
                return (
                    '{"question_type": "preparation", '
                    '"target": "chicken", '
                    '"reasoning": "Preparation matters.", '
                    '"question_template": "How was the chicken prepared?"}'
                )
            if module == "response_generator":
                return "How was the chicken prepared?"
            raise AssertionError(f"Unexpected module: {module}")

    result = ConversationEngine(FakeLLM().generate).generate_chat_replies(
        CoachingTurnRequest(
            current_message="I had chicken and rice.",
            enable_guardrail=False,
            enable_alignment=False,
        )
    )

    assert result.state.last_alignment_score is None
    assert result.state.last_alignment_reasoning is None
    assert result.metadata["alignment_enabled"] is False
    assert "alignment_raw_output" not in result.metadata


def test_workable_inquiry_is_redirected_to_assessment():
    class FakeLLM:
        def generate(self, *, module, messages, mode):
            if module == "meal_tracker":
                return (
                    "[Tracking State]\n"
                    "- Confirmed food items: turkey sandwich with lettuce and water\n\n"
                    "[Published Meal Base]\n"
                    "- Food items: turkey sandwich, lettuce, water\n"
                    "- Ingredients: turkey, whole wheat bread, lettuce\n"
                    "- Portions/amounts: one sandwich"
                )
            if module == "context_tracker":
                return "The user gave a workable meal description."
            if module == "interaction_tracker":
                return (
                    '{"answered_facts": ["turkey sandwich with lettuce and water"], '
                    '"open_questions": ["optional condiment detail"], '
                    '"rejected_options": [], '
                    '"unavailable_options": [], '
                    '"accepted_options": [], '
                    '"latest_user_position": "The user gave enough information to assess."}'
                )
            if module == "dialogue_planner":
                return (
                    '{"intent_summary": "The user gave a workable meal description.", '
                    '"user_intent": "informing", '
                    '"phase": "exploration", '
                    '"actionability": "workable", '
                    '"action": "inquire", '
                    '"closure_readiness": "actionable", '
                    '"reasoning": "Only optional detail is missing.", '
                    '"instruction": "Ask about condiments.", '
                    '"assessment_followup_action": "", '
                    '"assessment_followup_phase": "", '
                    '"assessment_followup_instruction": "", '
                    '"confidence": 0.78}'
                )
            if module == "meal_assessor" and mode == "assessment":
                return (
                    '{"summary": "A turkey sandwich with lettuce and water.", '
                    '"strengths": ["lean protein"], '
                    '"limitations": ["condiment detail is unresolved"], '
                    '"overall": "aligned"}'
                )
            if module == "info_seeker":
                return (
                    '{"question_type": "detail", '
                    '"target": "condiments", '
                    '"reasoning": "Condiments could affect leanness.", '
                    '"question_template": "What condiment, if any, will you use?"}'
                )
            if module == "response_generator":
                system = messages[0]["content"]
                if "feedback message" in system:
                    return "Your sandwich has a solid lean protein base."
                return "What condiment, if any, will you use?"
            raise AssertionError(f"Unexpected module: {module}, mode={mode}")

    result = ConversationEngine(FakeLLM().generate).generate_chat_replies(
        CoachingTurnRequest(
            current_message="Turkey sandwich with lettuce and water.",
            enable_guardrail=False,
        )
    )

    assert result.metadata["planning_policy"]["planned_action"] == "inquire"
    assert result.metadata["planning_policy"]["effective_action"] == "assess"
    assert (
        result.metadata["planning_policy"]["override"]
        == "exploration_sufficiency_to_assessment"
    )
    assert result.metadata["post_assessment_decision"]["action"] == "confirm"
    assert [message.kind for message in result.assistant_messages] == [
        "assessment",
        "confirmation",
    ]


def test_exploration_sufficiency_gate_stops_burdensome_availability_inquiry():
    class FakeLLM:
        def __init__(self):
            self.response_calls = 0

        def generate(self, *, module, messages, mode):
            if module == "meal_tracker":
                return (
                    "[Tracking State]\n"
                    "- Confirmed food items: pita, hummus, salad, stuffed grape leaves, brie\n\n"
                    "[Published Meal Base]\n"
                    "- Food items: pita, hummus, salad, stuffed grape leaves, brie\n"
                    "- Ingredients: pita, hummus, salad, stuffed grape leaves, brie"
                )
            if module == "context_tracker":
                return (
                    "The user is at a buffet and says brie is the only visible "
                    "protein option."
                )
            if module == "interaction_tracker":
                return (
                    '{"answered_facts": ["The user is at a buffet.", '
                    '"Brie is the only visible protein option."], '
                    '"open_questions": ["Whether any other protein is available"], '
                    '"rejected_options": [], '
                    '"unavailable_options": ["No other protein options are available"], '
                    '"candidate_options": ["brie", "pita", "hummus", "salad"], '
                    '"accepted_options": [], '
                    '"latest_user_position": "The user says they only see brie; no other protein options are available."}'
                )
            if module == "dialogue_planner":
                return (
                    '{"intent_summary": "The user set a strong buffet availability boundary.", '
                    '"user_intent": "informing", '
                    '"phase": "exploration", '
                    '"actionability": "insufficient", '
                    '"action": "inquire", '
                    '"closure_readiness": "not_ready", '
                    '"reasoning": "Ask them to check for other protein.", '
                    '"instruction": "Ask whether another protein is available.", '
                    '"assessment_followup_action": "", '
                    '"assessment_followup_phase": "", '
                    '"assessment_followup_instruction": "", '
                    '"confidence": 0.71}'
                )
            if module == "meal_assessor" and mode == "assessment":
                return (
                    '{"summary": "The buffet meal has some food items but limited lean protein.", '
                    '"strengths": ["The user has identified the available foods."], '
                    '"limitations": ["Brie is not a lean protein anchor."], '
                    '"overall": "partially_aligned"}'
                )
            if module == "recommender":
                return (
                    '{"recommendation_type": "modify", '
                    '"target_food": "buffet plate", '
                    '"suggestion": "Keep the hummus and salad as the main workable add-ons and keep the brie modest.", '
                    '"reasoning": "This works within the stated buffet boundary.", '
                    '"expected_impact": "medium", '
                    '"options": ['
                    '{"option_id": "use_available_plate", '
                    '"target_food": "buffet plate", '
                    '"suggestion": "Use the hummus and salad as the practical add-ons and keep brie modest.", '
                    '"reasoning": "No other protein is available.", '
                    '"expected_impact": "medium"}]}'
                )
            if module == "info_seeker":
                raise AssertionError("Exploration sufficiency gate should suppress another inquiry")
            if module == "response_generator":
                self.response_calls += 1
                if self.response_calls == 1:
                    return "You have a workable buffet plate, but lean protein is limited."
                return "Given what is available, keep hummus and salad central and keep brie modest."
            raise AssertionError(f"Unexpected module: {module}, mode={mode}")

    result = ConversationEngine(FakeLLM().generate).generate_chat_replies(
        CoachingTurnRequest(
            current_message=(
                "I'm at the buffet now, I only see brie; no other protein "
                "options are available."
            ),
            enable_guardrail=False,
            nutrition_goal="lean_protein",
        )
    )

    assert result.metadata["planning_policy"]["planned_action"] == "inquire"
    assert result.metadata["planning_policy"]["effective_action"] == "assess"
    assert (
        result.metadata["exploration_sufficiency_gate"]["gate"]
        == "exploration_sufficiency_to_assessment"
    )
    assert result.metadata["exploration_sufficiency_gate"]["burden_or_boundary_signaled"]
    assert [message.kind for message in result.assistant_messages] == [
        "assessment",
        "recommendation",
    ]


def test_exploration_sufficiency_gate_keeps_inquiry_without_meal_anchor():
    class FakeLLM:
        def generate(self, *, module, messages, mode):
            if module == "meal_tracker":
                return (
                    "[Tracking State]\n"
                    "- Confirmed food items: none\n\n"
                    "[Published Meal Base]\n"
                    "- Food items: not yet mentioned"
                )
            if module == "context_tracker":
                return "The user has not provided a concrete meal yet."
            if module == "interaction_tracker":
                return (
                    '{"answered_facts": [], '
                    '"open_questions": ["What meal option is the user considering?"], '
                    '"rejected_options": [], '
                    '"unavailable_options": [], '
                    '"candidate_options": [], '
                    '"accepted_options": [], '
                    '"latest_user_position": "The user has not decided on a meal."}'
                )
            if module == "dialogue_planner":
                return (
                    '{"intent_summary": "The user has no concrete meal anchor yet.", '
                    '"user_intent": "informing", '
                    '"phase": "exploration", '
                    '"actionability": "insufficient", '
                    '"action": "inquire", '
                    '"closure_readiness": "not_ready", '
                    '"reasoning": "A concrete meal anchor is needed.", '
                    '"instruction": "Ask what kind of meal they are considering.", '
                    '"assessment_followup_action": "", '
                    '"assessment_followup_phase": "", '
                    '"assessment_followup_instruction": "", '
                    '"confidence": 0.76}'
                )
            if module == "info_seeker":
                return (
                    '{"question_type": "meal_anchor", '
                    '"target": "meal", '
                    '"reasoning": "No meal has been named.", '
                    '"question_template": "What kind of meal are you leaning toward?"}'
                )
            if module == "response_generator":
                return "What kind of meal are you leaning toward?"
            if module in {"meal_assessor", "recommender"}:
                raise AssertionError(f"{module} should not run without a meal anchor")
            raise AssertionError(f"Unexpected module: {module}, mode={mode}")

    result = ConversationEngine(FakeLLM().generate).generate_chat_replies(
        CoachingTurnRequest(
            current_message="I haven't decided yet.",
            enable_guardrail=False,
            nutrition_goal="lean_protein",
        )
    )

    assert result.metadata["planning_policy"]["effective_action"] == "inquire"
    assert "exploration_sufficiency_gate" not in result.metadata
    assert [message.kind for message in result.assistant_messages] == ["question"]


def test_assessment_saturation_gate_answers_simplification_without_reassessment():
    class FakeLLM:
        def generate(self, *, module, messages, mode):
            if module == "meal_tracker":
                return (
                    "[Tracking State]\n"
                    "- Confirmed food items: turkey sandwich, chips, water\n\n"
                    "[Published Meal Base]\n"
                    "- Food items: turkey sandwich, chips, water"
                )
            if module == "context_tracker":
                return "The user wants a simple answer about an already discussed meal."
            if module == "interaction_tracker":
                return (
                    '{"answered_facts": ["turkey sandwich with chips and water"], '
                    '"open_questions": [], '
                    '"rejected_options": [], '
                    '"unavailable_options": [], '
                    '"accepted_options": [], '
                    '"latest_user_position": "The user already listed the meal and wants the simplest guidance now."}'
                )
            if module == "dialogue_planner":
                return (
                    '{"intent_summary": "The user asks for the simplest guidance on the same meal.", '
                    '"user_intent": "inquiring", '
                    '"phase": "recommendation", '
                    '"actionability": "workable", '
                    '"action": "assess", '
                    '"closure_readiness": "actionable", '
                    '"reasoning": "Reassess before answering.", '
                    '"instruction": "Reassess the turkey sandwich.", '
                    '"assessment_followup_action": "recommend", '
                    '"assessment_followup_phase": "recommendation", '
                    '"assessment_followup_instruction": "Recommend the simplest adjustment.", '
                    '"confidence": 0.76}'
                )
            if module in {"meal_assessor", "recommender", "info_seeker"}:
                raise AssertionError(
                    f"{module} should not run when assessment is saturated"
                )
            if module == "response_generator":
                return "The simplest step is to keep the turkey lean and leave the rest as-is."
            raise AssertionError(f"Unexpected module: {module}, mode={mode}")

    result = ConversationEngine(FakeLLM().generate).generate_chat_replies(
        CoachingTurnRequest(
            current_message=(
                "I already listed the meal. Please just tell me the simplest "
                "adjustment, if any."
            ),
            state=CoachingState(
                phase="recommendation",
                recommendation_history=(
                    {
                        "turn_idx": 0,
                        "suggestion": "use lean turkey and keep chips small",
                        "target_food": "turkey sandwich",
                    },
                ),
            ),
            enable_guardrail=False,
            nutrition_goal="lean_protein",
        )
    )

    assert result.metadata["planning_policy"]["planned_action"] == "assess"
    assert result.metadata["planning_policy"]["effective_action"] == "respond"
    assert (
        result.metadata["assessment_saturation_gate"]["gate"]
        == "assessment_saturation_to_response"
    )
    assert [message.kind for message in result.assistant_messages] == ["answer"]


def test_assessment_saturation_gate_routes_refinement_to_recommendation():
    class FakeLLM:
        def generate(self, *, module, messages, mode):
            if module == "meal_tracker":
                return (
                    "[Tracking State]\n"
                    "- Confirmed food items: egg-free omelet, turkey filling\n\n"
                    "[Published Meal Base]\n"
                    "- Food items: egg-free omelet, turkey filling"
                )
            if module == "context_tracker":
                return "The user wants another safe side recommendation."
            if module == "interaction_tracker":
                return (
                    '{"answered_facts": ["egg-free omelet style", "turkey filling works"], '
                    '"open_questions": ["safe side"], '
                    '"rejected_options": ["Greek yogurt"], '
                    '"unavailable_options": [], '
                    '"accepted_options": ["turkey filling"], '
                    '"latest_user_position": "Turkey works for the filling, but the user wants a safe side recommendation."}'
                )
            if module == "dialogue_planner":
                return (
                    '{"intent_summary": "The user asks for a new safe side recommendation.", '
                    '"user_intent": "inquiring", '
                    '"phase": "negotiation", '
                    '"actionability": "workable", '
                    '"action": "assess", '
                    '"closure_readiness": "actionable", '
                    '"reasoning": "Assess before recommending.", '
                    '"instruction": "Assess the updated breakfast.", '
                    '"assessment_followup_action": "recommend", '
                    '"assessment_followup_phase": "recommendation", '
                    '"assessment_followup_instruction": "Recommend a safe side.", '
                    '"confidence": 0.77}'
                )
            if module == "meal_assessor":
                raise AssertionError("repeated refinement should skip assessment")
            if module == "recommender":
                return (
                    '{"recommendation_type": "add", '
                    '"target_food": "breakfast side", '
                    '"suggestion": "add plain firm tofu as the side", '
                    '"reasoning": "It adds lean protein without egg or dairy.", '
                    '"expected_impact": "medium", '
                    '"options": ['
                    '{"option_id": "opt1", "target_food": "breakfast side", '
                    '"suggestion": "add plain firm tofu as the side", '
                    '"reasoning": "It adds lean protein without egg or dairy.", '
                    '"expected_impact": "medium"}]}'
                )
            if module == "response_generator":
                return "Add plain firm tofu as the side. It keeps the plan egg-free and protein-focused."
            raise AssertionError(f"Unexpected module: {module}, mode={mode}")

    result = ConversationEngine(FakeLLM().generate).generate_chat_replies(
        CoachingTurnRequest(
            current_message=(
                "Turkey works for the filling, but skip Greek yogurt. "
                "Can you suggest one safer egg-free side?"
            ),
            state=CoachingState(
                phase="recommendation",
                recommendation_history=(
                    {
                        "turn_idx": 0,
                        "suggestion": "use turkey filling and Greek yogurt side",
                        "target_food": "breakfast",
                    },
                ),
            ),
            enable_guardrail=False,
            nutrition_goal="lean_protein",
        )
    )

    assert result.metadata["planning_policy"]["planned_action"] == "assess"
    assert result.metadata["planning_policy"]["effective_action"] == "recommend"
    assert (
        result.metadata["assessment_saturation_gate"]["gate"]
        == "assessment_saturation_to_recommendation_refinement"
    )
    assert [message.kind for message in result.assistant_messages] == ["recommendation"]


def test_post_assessment_gate_confirms_preserved_plan_after_rejected_addon():
    class FakeLLM:
        def generate(self, *, module, messages, mode):
            if module == "meal_tracker":
                return (
                    "[Tracking State]\n"
                    "- Confirmed food items: grilled chicken, salad, fries\n\n"
                    "[Published Meal Base]\n"
                    "- Food items: grilled chicken, salad, fries"
                )
            if module == "context_tracker":
                return "The user wants to preserve the current meal and reject fruit."
            if module == "interaction_tracker":
                return (
                    '{"answered_facts": ["grilled chicken, salad, and fries are the meal"], '
                    '"open_questions": [], '
                    '"rejected_options": ["adding fruit to the salad"], '
                    '"unavailable_options": [], '
                    '"accepted_options": ["grilled chicken", "salad", "fries"], '
                    '"latest_user_position": "The user wants to keep grilled chicken, salad, and fries with just that and does not want to add fruit."}'
                )
            if module == "dialogue_planner":
                return (
                    '{"intent_summary": "The user asks to reassess the preserved meal without fruit.", '
                    '"user_intent": "rejecting", '
                    '"phase": "negotiation", '
                    '"actionability": "workable", '
                    '"action": "assess", '
                    '"closure_readiness": "actionable", '
                    '"reasoning": "Assess then recommend another adjustment.", '
                    '"instruction": "Assess the updated meal.", '
                    '"assessment_followup_action": "recommend", '
                    '"assessment_followup_phase": "recommendation", '
                    '"assessment_followup_instruction": "Recommend another produce adjustment.", '
                    '"confidence": 0.79}'
                )
            if module == "meal_assessor" and mode == "assessment":
                return (
                    '{"summary": "The meal has chicken, salad, and fries, but no fruit.", '
                    '"strengths": ["salad contributes vegetables"], '
                    '"limitations": ["fruit is missing"], '
                    '"overall": "partially_aligned"}'
                )
            if module == "recommender":
                raise AssertionError(
                    "post-assessment gate should prevent another recommendation"
                )
            if module == "response_generator":
                if "feedback message" in messages[0]["content"]:
                    return "This plan has vegetables from the salad, while fruit remains missing."
                return "You’re keeping grilled chicken, salad, and fries. Does that look right to finalize?"
            raise AssertionError(f"Unexpected module: {module}, mode={mode}")

    result = ConversationEngine(FakeLLM().generate).generate_chat_replies(
        CoachingTurnRequest(
            current_message=(
                "Let’s keep the grilled chicken, salad, and fries, but I don’t "
                "want to add fruit. Can you reassess it with just that?"
            ),
            state=CoachingState(
                phase="recommendation",
                recommendation_history=(
                    {
                        "turn_idx": 0,
                        "suggestion": "add fruit to the salad",
                        "target_food": "salad",
                    },
                ),
            ),
            enable_guardrail=False,
            nutrition_goal="half_fruits_vegetables",
        )
    )

    assert result.metadata["planning_policy"]["effective_action"] == "assess"
    assert result.metadata["post_assessment_decision"]["action"] == "confirm"
    assert (
        result.metadata["post_assessment_gate"]["gate"]
        == "post_assessment_preserved_plan_confirmed"
    )
    assert [message.kind for message in result.assistant_messages] == [
        "assessment",
        "confirmation",
    ]


def test_accepted_commitment_is_gated_to_assessment():
    class FakeLLM:
        def generate(self, *, module, messages, mode):
            if module == "recommender":
                raise AssertionError("accepted commitment should not call recommender")
            if module == "meal_tracker":
                return (
                    "[Tracking State]\n"
                    "- Confirmed food items: turkey sandwich and apple\n\n"
                    "[Published Meal Base]\n"
                    "- Food items: turkey sandwich, apple\n"
                    "- Ingredients: turkey, bread, apple"
                )
            if module == "context_tracker":
                return "The user accepted an apple as an easy produce option."
            if module == "interaction_tracker":
                return (
                    '{"answered_facts": ["apple is easiest"], '
                    '"open_questions": [], '
                    '"rejected_options": [], '
                    '"unavailable_options": [], '
                    '"accepted_options": ["apple"], '
                    '"latest_user_position": "The user accepted apple as good enough."}'
                )
            if module == "dialogue_planner":
                return (
                    '{"intent_summary": "The user accepted the apple.", '
                    '"user_intent": "accepting", '
                    '"phase": "recommendation", '
                    '"actionability": "workable", '
                    '"action": "recommend", '
                    '"closure_readiness": "actionable", '
                    '"reasoning": "Another produce recommendation may help.", '
                    '"instruction": "Suggest another produce option.", '
                    '"assessment_followup_action": "", '
                    '"assessment_followup_phase": "", '
                    '"assessment_followup_instruction": "", '
                    '"confidence": 0.72}'
                )
            if module == "meal_assessor" and mode == "assessment":
                return (
                    '{"summary": "A turkey sandwich with an apple.", '
                    '"strengths": ["added fruit"], '
                    '"limitations": [], '
                    '"overall": "aligned"}'
                )
            if module == "response_generator":
                system = messages[0]["content"]
                if "feedback message" in system:
                    return "The apple is a useful produce add-on here."
                return "Nice, the apple is an easy way to move this meal closer to your goal."
            raise AssertionError(f"Unexpected module: {module}, mode={mode}")

    result = ConversationEngine(FakeLLM().generate).generate_chat_replies(
        CoachingTurnRequest(
            current_message="Sounds good, apple is easiest.",
            enable_guardrail=False,
        )
    )

    assert result.metadata["planning_policy"]["planned_action"] == "recommend"
    assert result.metadata["planning_policy"]["effective_action"] == "assess"
    assert result.metadata["commitment_gate"]["gate"] == (
        "accepted_commitment_to_assessment"
    )
    assert result.metadata["post_assessment_decision"]["action"] == "confirm"
    assert result.state.status == "active"
    assert result.state.phase == "confirmation"


def test_mentioned_option_without_accepting_intent_does_not_trigger_commitment_gate():
    class FakeLLM:
        def generate(self, *, module, messages, mode):
            if module == "meal_tracker":
                return (
                    "[Tracking State]\n"
                    "- Confirmed food items: spicy noodles\n\n"
                    "[Published Meal Base]\n"
                    "- Food items: spicy noodles"
                )
            if module == "context_tracker":
                return "The user mentioned spicy noodles."
            if module == "interaction_tracker":
                return (
                    '{"answered_facts": ["the user is having spicy noodles"], '
                    '"open_questions": [], '
                    '"rejected_options": [], '
                    '"unavailable_options": [], '
                    '"accepted_options": ["spicy noodles"], '
                    '"latest_user_position": "The user is having spicy noodles."}'
                )
            if module == "dialogue_planner":
                return (
                    '{"intent_summary": "The user mentions spicy noodles.", '
                    '"user_intent": "informing", '
                    '"phase": "exploration", '
                    '"actionability": "workable", '
                    '"action": "recommend", '
                    '"closure_readiness": "actionable", '
                    '"reasoning": "Water would support the hydration goal.", '
                    '"instruction": "Suggest water with noodles.", '
                    '"assessment_followup_action": "", '
                    '"assessment_followup_phase": "", '
                    '"assessment_followup_instruction": "", '
                    '"confidence": 0.75}'
                )
            if module == "recommender":
                return (
                    '{"recommendation_type":"add", '
                    '"target_food":"drink", '
                    '"suggestion":"have water with the noodles", '
                    '"reasoning":"This supports hydration.", '
                    '"expected_impact":"high"}'
                )
            if module == "meal_assessor" and mode == "assessment":
                return (
                    '{"summary": "Spicy noodles without a drink.", '
                    '"strengths": ["clear meal item"], '
                    '"limitations": ["drink not yet included"], '
                    '"overall": "partially_aligned"}'
                )
            if module == "response_generator":
                if "feedback message" in messages[0]["content"]:
                    return "Your noodles are clear, but the drink part is still missing."
                return "How about water with the noodles? It supports your hydration goal."
            raise AssertionError(f"Unexpected module: {module}, mode={mode}")

    result = ConversationEngine(FakeLLM().generate).generate_chat_replies(
        CoachingTurnRequest(
            current_message="I'm having spicy noodles.",
            nutrition_goal="drink_water",
            enable_guardrail=False,
        )
    )

    assert "commitment_gate" not in result.metadata
    assert result.metadata["planning_policy"]["effective_action"] == "assess"
    assert result.metadata["planning_policy"]["override"] == (
        "recommendation_grounded_by_assessment"
    )
    assert [m.kind for m in result.assistant_messages] == [
        "assessment",
        "recommendation",
    ]


def test_candidate_option_list_does_not_trigger_commitment_gate():
    class FakeLLM:
        def generate(self, *, module, messages, mode):
            if module == "meal_tracker":
                return (
                    "[Tracking State]\n"
                    "- Candidate add-ons: lettuce, cucumber, apple\n\n"
                    "[Published Meal Base]\n"
                    "- Food items: turkey sandwich, chips\n"
                    "- Possible add-ons: lettuce, cucumber, apple"
                )
            if module == "context_tracker":
                return "The user named several possible produce add-ons."
            if module == "interaction_tracker":
                return (
                    '{"answered_facts": ["lettuce, cucumber, or apple are possible"], '
                    '"open_questions": [], '
                    '"rejected_options": [], '
                    '"unavailable_options": [], '
                    '"candidate_options": ["lettuce", "cucumber", "apple"], '
                    '"accepted_options": [], '
                    '"latest_user_position": "The user can add lettuce, cucumber, or apple."}'
                )
            if module == "dialogue_planner":
                return (
                    '{"intent_summary": "The user named possible produce add-ons.", '
                    '"user_intent": "accepting", '
                    '"phase": "recommendation", '
                    '"actionability": "workable", '
                    '"action": "recommend", '
                    '"closure_readiness": "actionable", '
                    '"reasoning": "A specific produce recommendation is still needed.", '
                    '"instruction": "Recommend one produce combination.", '
                    '"assessment_followup_action": "", '
                    '"assessment_followup_phase": "", '
                    '"assessment_followup_instruction": "", '
                    '"confidence": 0.72}'
                )
            if module == "recommender":
                return (
                    '{"recommendation_type":"add", '
                    '"target_food":"apple and cucumber", '
                    '"suggestion":"add apple and cucumber", '
                    '"reasoning":"This uses the available options.", '
                    '"expected_impact":"high"}'
                )
            if module == "meal_assessor" and mode == "assessment":
                return (
                    '{"summary": "Turkey sandwich and chips with candidate produce add-ons.", '
                    '"strengths": ["candidate produce options are available"], '
                    '"limitations": ["no produce option has been selected yet"], '
                    '"overall": "partially_aligned"}'
                )
            if module == "response_generator":
                if "feedback message" in messages[0]["content"]:
                    return "You have useful produce candidates, but none is selected yet."
                return "How about adding the apple plus cucumber slices?"
            raise AssertionError(f"Unexpected module: {module}, mode={mode}")

    result = ConversationEngine(FakeLLM().generate).generate_chat_replies(
        CoachingTurnRequest(
            current_message="I can add lettuce, cucumber, or an apple.",
            nutrition_goal="half_fruits_vegetables",
            enable_guardrail=False,
        )
    )

    assert "commitment_gate" not in result.metadata
    assert result.metadata["planning_policy"]["planned_action"] == "recommend"
    assert result.metadata["planning_policy"]["effective_action"] == "assess"
    assert result.metadata["planning_policy"]["override"] == (
        "recommendation_grounded_by_assessment"
    )
    assert [m.kind for m in result.assistant_messages] == [
        "assessment",
        "recommendation",
    ]
    assert "Candidate options:" in result.state.interaction_state
    assert "Accepted options:" not in result.state.interaction_state


def test_recommendation_result_is_persisted_in_state_history():
    class FakeLLM:
        def generate(self, *, module, messages, mode):
            if module == "meal_tracker":
                return (
                    "[Tracking State]\n"
                    "- Confirmed food items: turkey sandwich and chips\n\n"
                    "[Published Meal Base]\n"
                    "- Food items: turkey sandwich, chips"
                )
            if module == "context_tracker":
                return "The user can add produce."
            if module == "interaction_tracker":
                return (
                    '{"answered_facts": ["lettuce, cucumber, or apple are possible"], '
                    '"open_questions": [], '
                    '"rejected_options": [], '
                    '"unavailable_options": [], '
                    '"candidate_options": ["lettuce", "cucumber", "apple"], '
                    '"accepted_options": [], '
                    '"latest_user_position": "The user has candidate produce options."}'
                )
            if module == "dialogue_planner":
                return (
                    '{"intent_summary": "The user named candidate add-ons.", '
                    '"user_intent": "informing", '
                    '"phase": "recommendation", '
                    '"actionability": "workable", '
                    '"action": "recommend", '
                    '"closure_readiness": "actionable", '
                    '"reasoning": "A concrete candidate can be recommended.", '
                    '"instruction": "Recommend one candidate produce add-on.", '
                    '"assessment_followup_action": "", '
                    '"assessment_followup_phase": "", '
                    '"assessment_followup_instruction": "", '
                    '"confidence": 0.72}'
                )
            if module == "recommender":
                return (
                    '{"recommendation_type":"add", '
                    '"target_food":"apple and cucumber", '
                    '"suggestion":"add apple and cucumber", '
                    '"reasoning":"This uses the candidate options.", '
                    '"expected_impact":"high"}'
                )
            if module == "meal_assessor" and mode == "assessment":
                return (
                    '{"summary": "Turkey sandwich and chips with candidate produce add-ons.", '
                    '"strengths": ["candidate options are known"], '
                    '"limitations": ["produce has not been added yet"], '
                    '"overall": "partially_aligned"}'
                )
            if module == "response_generator":
                if "feedback message" in messages[0]["content"]:
                    return "You have candidate produce options available."
                return "How about apple plus cucumber? That uses what you already have."
            raise AssertionError(f"Unexpected module: {module}, mode={mode}")

    result = ConversationEngine(FakeLLM().generate).generate_chat_replies(
        CoachingTurnRequest(
            current_message="I can add lettuce, cucumber, or an apple.",
            nutrition_goal="half_fruits_vegetables",
            enable_guardrail=False,
        )
    )

    assert len(result.state.recommendation_history) == 1
    assert result.state.recommendation_history[0]["target_food"] == "apple and cucumber"
    assert result.state.recommendation_history[0]["suggestion"] == "add apple and cucumber"


def test_contracted_commitment_can_trigger_commitment_gate():
    class FakeLLM:
        def generate(self, *, module, messages, mode):
            if module == "recommender":
                raise AssertionError("contracted commitment should not call recommender")
            if module == "meal_tracker":
                return (
                    "[Tracking State]\n"
                    "- Confirmed food items: tofu, broccoli, mushrooms\n\n"
                    "[Published Meal Base]\n"
                    "- Food items: tofu, broccoli, mushrooms"
                )
            if module == "context_tracker":
                return "The user committed to tofu with broccoli and mushrooms."
            if module == "interaction_tracker":
                return (
                    '{"answered_facts": ["tofu with broccoli and mushrooms"], '
                    '"open_questions": [], '
                    '"rejected_options": ["starch"], '
                    '"unavailable_options": [], '
                    '"accepted_options": ["tofu with broccoli and mushrooms"], '
                    '"latest_user_position": "The user will go with tofu and vegetables."}'
                )
            if module == "dialogue_planner":
                return (
                    '{"intent_summary": "The user accepted the tofu plan.", '
                    '"user_intent": "accepting", '
                    '"phase": "recommendation", '
                    '"actionability": "workable", '
                    '"action": "recommend", '
                    '"closure_readiness": "actionable", '
                    '"reasoning": "Another suggestion might help.", '
                    '"instruction": "Suggest another non-starchy vegetable.", '
                    '"assessment_followup_action": "", '
                    '"assessment_followup_phase": "", '
                    '"assessment_followup_instruction": "", '
                    '"confidence": 0.72}'
                )
            if module == "meal_assessor" and mode == "assessment":
                return (
                    '{"summary": "Tofu with broccoli and mushrooms.", '
                    '"strengths": ["non-starchy vegetables"], '
                    '"limitations": [], '
                    '"overall": "aligned"}'
                )
            if module == "response_generator":
                system = messages[0]["content"]
                if "feedback message" in system:
                    return "The tofu and vegetables plan fits your direction."
                return "Great, let's finish with tofu, broccoli, and mushrooms."
            raise AssertionError(f"Unexpected module: {module}, mode={mode}")

    result = ConversationEngine(FakeLLM().generate).generate_chat_replies(
        CoachingTurnRequest(
            current_message="I'll go with tofu and broccoli.",
            nutrition_goal="one_fourth_carbs",
            enable_guardrail=False,
        )
    )

    assert result.metadata["commitment_gate"]["gate"] == (
        "accepted_commitment_to_assessment"
    )
    assert result.metadata["planning_policy"]["effective_action"] == "assess"


def test_dialogue_planner_handoff_generates_default_bridge_bubble():
    class FakeLLM:
        def generate(self, *, module, messages, mode):
            if module in {"recommender", "meal_assessor", "info_seeker"}:
                raise AssertionError(f"handoff should not call {module}")
            if module == "meal_tracker":
                return (
                    "[Tracking State]\n"
                    "- Confirmed food items: turkey sandwich and chips\n\n"
                    "[Published Meal Base]\n"
                    "- Food items: turkey sandwich, chips"
                )
            if module == "context_tracker":
                return "The user rejected the last suggested adjustment."
            if module == "interaction_tracker":
                return (
                    '{"answered_facts": ["turkey sandwich and chips"], '
                    '"open_questions": [], '
                    '"rejected_options": ["add cottage cheese"], '
                    '"unavailable_options": [], '
                    '"accepted_options": ["turkey sandwich"], '
                    '"latest_user_position": "The user does not want the suggested add-on."}'
                )
            if module == "dialogue_planner":
                return (
                    '{"intent_summary": "The user rejected the suggested add-on.", '
                    '"user_intent": "rejecting", '
                    '"phase": "negotiation", '
                    '"actionability": "workable", '
                    '"action": "handoff", '
                    '"closure_readiness": "actionable", '
                    '"reasoning": "Several next directions are possible.", '
                    '"instruction": "Default to preserving the current plan and briefly naming the tradeoff.", '
                    '"assessment_followup_action": "", '
                    '"assessment_followup_phase": "", '
                    '"assessment_followup_instruction": "", '
                    '"confidence": 0.82}'
                )
            if module == "response_generator":
                assert "control-handoff message" in messages[0]["content"]
                return (
                    "Let’s keep your turkey sandwich plan for now and avoid adding another decision.\n"
                    "- Cottage cheese is off the table.\n"
                    "- The main tradeoff is that the meal may stay lighter on lean protein."
                )
            raise AssertionError(f"Unexpected module: {module}, mode={mode}")

    result = ConversationEngine(FakeLLM().generate).generate_chat_replies(
        CoachingTurnRequest(
            current_message="No, I don't want cottage cheese.",
            state=CoachingState(
                phase="negotiation",
                recommendation_history=(
                    {
                        "turn_idx": 1,
                        "recommendation_type": "add",
                        "target_food": "protein add-on",
                        "suggestion": "add cottage cheese",
                    },
                ),
            ),
            enable_guardrail=False,
        )
    )

    assert result.metadata["planning_policy"]["effective_action"] == "handoff"
    assert result.state.phase == "negotiation"
    assert result.state.status == "active"
    assert [message.kind for message in result.assistant_messages] == ["handoff"]


def test_repeated_rejection_hands_control_to_user_instead_of_closing():
    class FakeLLM:
        def generate(self, *, module, messages, mode):
            if module in {"recommender", "meal_assessor", "info_seeker"}:
                raise AssertionError(f"rejection handoff should not call {module}")
            if module == "meal_tracker":
                return (
                    "[Tracking State]\n"
                    "- Confirmed food items: ramen\n\n"
                    "[Published Meal Base]\n"
                    "- Food items: ramen"
                )
            if module == "context_tracker":
                return "The user rejected another refinement but did not ask to stop."
            if module == "interaction_tracker":
                return (
                    '{"answered_facts": ["ramen"], '
                    '"open_questions": [], '
                    '"rejected_options": ["add vegetables", "reduce noodles"], '
                    '"unavailable_options": [], '
                    '"accepted_options": ["ramen"], '
                    '"latest_user_position": "The user rejects another refinement but has not asked to stop."}'
                )
            if module == "dialogue_planner":
                return (
                    '{"intent_summary": "The user rejects another refinement.", '
                    '"user_intent": "rejecting", '
                    '"phase": "negotiation", '
                    '"actionability": "workable", '
                    '"action": "recommend", '
                    '"closure_readiness": "actionable", '
                    '"reasoning": "Another recommendation might help.", '
                    '"instruction": "Suggest one more refinement.", '
                    '"assessment_followup_action": "", '
                    '"assessment_followup_phase": "", '
                    '"assessment_followup_instruction": "", '
                    '"confidence": 0.72}'
                )
            if module == "response_generator":
                assert "control-handoff message" in messages[0]["content"]
                return (
                    "Let’s keep the ramen as-is for now and avoid another round of swaps. "
                    "The tradeoff is that it stays higher in starch."
                )
            raise AssertionError(f"Unexpected module: {module}, mode={mode}")

    result = ConversationEngine(FakeLLM().generate).generate_chat_replies(
        CoachingTurnRequest(
            current_message="No, I don't want to reduce the noodles either.",
            state=CoachingState(
                phase="negotiation",
                recommendation_rejection_count=2,
                recommendation_history=(
                    {
                        "turn_idx": 1,
                        "recommendation_type": "modify",
                        "target_food": "ramen",
                        "suggestion": "reduce noodles",
                    },
                ),
            ),
            enable_guardrail=False,
        )
    )

    assert result.metadata["planning_policy"]["override"] == (
        "resistance_threshold_handoff"
    )
    assert result.metadata["planning_policy"]["effective_action"] == "handoff"
    assert result.state.status == "active"
    assert result.state.phase == "negotiation"


def test_safety_conflict_gets_one_clarification():
    class FakeLLM:
        def generate(self, *, module, messages, mode):
            if module in {"recommender", "meal_assessor", "info_seeker"}:
                raise AssertionError(f"safety gate should not call {module}")
            if module == "meal_tracker":
                return (
                    "[Tracking State]\n"
                    "- Confirmed food items: cheese pizza\n\n"
                    "[Published Meal Base]\n"
                    "- Food items: cheese pizza\n"
                    "- Ingredients: cheese, pizza crust"
                )
            if module == "context_tracker":
                return "The user has a cheese allergy and mentioned cheese pizza."
            if module == "interaction_tracker":
                return (
                    '{"answered_facts": ["the user wants cheese pizza"], '
                    '"open_questions": [], '
                    '"rejected_options": [], '
                    '"unavailable_options": [], '
                    '"accepted_options": ["cheese pizza"], '
                    '"latest_user_position": "The user wants cheese pizza."}'
                )
            if module == "dialogue_planner":
                return (
                    '{"intent_summary": "The user wants cheese pizza.", '
                    '"user_intent": "accepting", '
                    '"phase": "recommendation", '
                    '"actionability": "settled", '
                    '"action": "recommend", '
                    '"closure_readiness": "ready_to_close", '
                    '"reasoning": "The user chose pizza.", '
                    '"instruction": "Recommend the pizza.", '
                    '"assessment_followup_action": "", '
                    '"assessment_followup_phase": "", '
                    '"assessment_followup_instruction": "", '
                    '"confidence": 0.8}'
                )
            if module == "response_generator":
                content = messages[1]["content"]
                assert "stored allergy constraint (cheese)" in content
                assert "ask one clarifying question" in content
                return "I have cheese listed as an allergy. Did you mean dairy-free cheese pizza?"
            raise AssertionError(f"Unexpected module: {module}, mode={mode}")

    result = ConversationEngine(FakeLLM().generate).generate_chat_replies(
        CoachingTurnRequest(
            current_message="I still want cheese pizza.",
            profile=UserProfileContext(allergies=["cheese"]),
            enable_guardrail=False,
        )
    )

    assert result.metadata["planning_policy"]["effective_action"] == "respond"
    assert result.metadata["planning_policy"]["effective_phase"] == "exploration"
    assert result.metadata["commitment_gate"]["gate"] == "safety_conflict_clarification"
    assert result.metadata["commitment_gate"]["constraint"] == "cheese"
    assert result.state.safety_clarification_counts["allergy:cheese"] == 1
    assert result.state.phase == "exploration"
    assert result.assistant_messages[-1].kind == "answer"


def test_repeated_safety_conflict_continues_with_safe_plan_unless_user_stops():
    class FakeLLM:
        def generate(self, *, module, messages, mode):
            if module == "meal_tracker":
                return (
                    "[Tracking State]\n"
                    "- Confirmed food items: cheese pizza\n\n"
                    "[Published Meal Base]\n"
                    "- Food items: cheese pizza\n"
                    "- Ingredients: cheese, pizza crust"
                )
            if module == "context_tracker":
                return "The user repeated cheese pizza despite the allergy conflict."
            if module == "interaction_tracker":
                return (
                    '{"answered_facts": ["the user repeated cheese pizza"], '
                    '"open_questions": [], '
                    '"rejected_options": [], '
                    '"unavailable_options": [], '
                    '"accepted_options": ["cheese pizza"], '
                    '"latest_user_position": "The user repeated cheese pizza."}'
                )
            if module == "dialogue_planner":
                return (
                    '{"intent_summary": "The user repeats the cheese pizza choice.", '
                    '"user_intent": "accepting", '
                    '"phase": "negotiation", '
                    '"actionability": "settled", '
                    '"action": "close", '
                    '"closure_readiness": "ready_to_close", '
                    '"reasoning": "The user is settled.", '
                    '"instruction": "Close.", '
                    '"assessment_followup_action": "", '
                    '"assessment_followup_phase": "", '
                    '"assessment_followup_instruction": "", '
                    '"confidence": 0.8}'
                )
            if module == "meal_assessor" and mode == "assessment":
                return (
                    '{"summary": "The user still wants regular cheese pizza, '
                    'which conflicts with the stored cheese allergy.", '
                    '"strengths": ["the user stated a clear preference"], '
                    '"limitations": ["regular cheese remains safety-conflicted"], '
                    '"overall": "conflicted"}'
                )
            if module == "recommender":
                return (
                    '{"recommendation_type": "safe_alternative", '
                    '"target_food": "pizza", '
                    '"suggestion": "use a cheese-free pizza or a safe non-dairy topping", '
                    '"reasoning": "This preserves the pizza direction without endorsing the allergy-conflicted item.", '
                    '"expected_impact": "keeps the plan safer while respecting the user preference"}'
                )
            if module == "response_generator":
                system = messages[0]["content"]
                if "feedback message" in system:
                    return "I can’t treat regular cheese as safe with your stored cheese allergy."
                return "If you still want pizza, a cheese-free pizza or safe non-dairy topping keeps the same direction without using the conflicted cheese."
            raise AssertionError(f"Unexpected module: {module}, mode={mode}")

    result = ConversationEngine(FakeLLM().generate).generate_chat_replies(
        CoachingTurnRequest(
            current_message="No, regular cheese pizza.",
            profile=UserProfileContext(allergies=["cheese"]),
            state=CoachingState(safety_clarification_counts={"allergy:cheese": 1}),
            enable_guardrail=False,
        )
    )

    assert result.metadata["commitment_gate"]["gate"] == "safety_conflict_to_safe_plan"
    assert result.metadata["commitment_gate"]["prior_clarification_count"] == 1
    assert result.state.safety_clarification_counts["allergy:cheese"] == 2
    assert result.metadata["planning_policy"]["effective_action"] == "assess"
    assert result.state.phase == "recommendation"
    assert result.state.status == "active"
    assert [message.kind for message in result.assistant_messages] == [
        "assessment",
        "recommendation",
    ]


def test_repeated_safety_conflict_can_close_after_explicit_stop_boundary():
    class FakeLLM:
        def generate(self, *, module, messages, mode):
            if module in {"recommender", "meal_assessor", "info_seeker"}:
                raise AssertionError(f"explicit stop safety close should not call {module}")
            if module == "meal_tracker":
                return (
                    "[Tracking State]\n"
                    "- Confirmed food items: cheese pizza\n\n"
                    "[Published Meal Base]\n"
                    "- Food items: cheese pizza\n"
                    "- Ingredients: cheese, pizza crust"
                )
            if module == "context_tracker":
                return "The user repeated cheese pizza despite the allergy conflict."
            if module == "interaction_tracker":
                return (
                    '{"answered_facts": ["the user repeated cheese pizza"], '
                    '"open_questions": [], '
                    '"rejected_options": [], '
                    '"unavailable_options": [], '
                    '"accepted_options": ["cheese pizza"], '
                    '"latest_user_position": "The user repeated cheese pizza and said to stop asking."}'
                )
            if module == "dialogue_planner":
                return (
                    '{"intent_summary": "The user repeats the cheese pizza choice and wants to stop.", '
                    '"user_intent": "rejecting", '
                    '"phase": "negotiation", '
                    '"actionability": "conflicted", '
                    '"action": "close", '
                    '"closure_readiness": "boundary_close", '
                    '"reasoning": "The user wants to stop further discussion.", '
                    '"instruction": "Close.", '
                    '"assessment_followup_action": "", '
                    '"assessment_followup_phase": "", '
                    '"assessment_followup_instruction": "", '
                    '"confidence": 0.8}'
                )
            if module == "response_generator":
                assert "Finalization style: **reflective**" in messages[0]["content"]
                content = messages[1]["content"]
                assert "cheese pizza" in content
                return "I hear you want to stop here, and I still can’t frame regular cheese as safe with your stored allergy."
            raise AssertionError(f"Unexpected module: {module}, mode={mode}")

    result = ConversationEngine(FakeLLM().generate).generate_chat_replies(
        CoachingTurnRequest(
            current_message="No, regular cheese pizza. Stop asking.",
            profile=UserProfileContext(allergies=["cheese"]),
            state=CoachingState(safety_clarification_counts={"allergy:cheese": 1}),
            enable_guardrail=False,
        )
    )

    assert result.metadata["commitment_gate"]["gate"] == "safety_conflict_reflective_close"
    assert result.metadata["commitment_gate"]["prior_clarification_count"] == 1
    assert result.state.safety_clarification_counts["allergy:cheese"] == 2
    assert result.metadata["finalization_style"] == "reflective"
    assert result.state.status == "terminated"


def test_conversation_engine_assess_then_inquire_followup():
    class FakeLLM:
        def generate(self, *, module, messages, mode):
            if module == "meal_tracker":
                return (
                    "[Tracking State]\n"
                    "- Confirmed food items: turkey sandwich\n"
                    "- Tentative food items: none\n"
                    "- Rejected food items: none\n"
                    "- Decision context: none\n\n"
                    "[Published Meal Base]\n"
                    "- Food items: turkey sandwich\n"
                    "- Ingredients: turkey, whole wheat bread, lettuce\n"
                    "- Preparation methods: assembled sandwich\n"
                    "- Portions/amounts: one sandwich\n"
                    "- Beverages: water\n"
                    "- Additional notes: not yet mentioned"
                )
            if module == "context_tracker":
                return "The user described a turkey sandwich meal."
            if module == "interaction_tracker":
                return (
                    '{"answered_facts": ["turkey sandwich on whole wheat with lettuce and water"], '
                    '"open_questions": ["vegetable preferences"], '
                    '"rejected_options": [], '
                    '"unavailable_options": [], '
                    '"accepted_options": [], '
                    '"latest_user_position": "The user gave enough detail for assessment."}'
                )
            if module == "alignment_estimator":
                return '{"answer": "0", "reasoning": "Vegetables could be improved."}'
            if module == "meal_assessor" and mode == "assessment":
                return (
                    '{"summary": "A turkey sandwich with water.", '
                    '"strengths": ["lean protein"], '
                    '"limitations": ["could add more vegetables"], '
                    '"overall": "partially_aligned"}'
                )
            if module == "dialogue_planner":
                return (
                    '{"intent_summary": "The user gave enough meal detail.", '
                    '"user_intent": "informing", '
                    '"phase": "exploration", '
                    '"action": "assess", '
                    '"reasoning": "The meal can now be assessed.", '
                    '"instruction": "", '
                    '"assessment_followup_action": "inquire", '
                    '"assessment_followup_phase": "recommendation", '
                    '"assessment_followup_instruction": "Ask about vegetable preferences.", '
                    '"confidence": 0.76}'
                )
            if module == "info_seeker":
                return (
                    '{"question_type": "preference", '
                    '"target": "vegetables", '
                    '"reasoning": "Preferences help tailor the recommendation.", '
                    '"question_template": "What vegetables do you like?"}'
                )
            if module == "response_generator":
                system = messages[0]["content"]
                if "feedback message" in system:
                    return "Your sandwich has a solid lean protein base."
                return "What vegetables do you like?"
            raise AssertionError(f"Unexpected module: {module}, mode={mode}")

    result = ConversationEngine(FakeLLM().generate).generate_chat_replies(
        CoachingTurnRequest(
            current_message="Turkey sandwich on whole wheat with lettuce and water.",
            profile=UserProfileContext(name="Alice"),
            enable_guardrail=False,
        )
    )

    assert [m.kind for m in result.assistant_messages] == ["assessment", "question"]
    assert result.state.phase == "exploration"
    assert result.state.status == "active"
    assert result.metadata["post_assessment_decision"]["action"] == "inquire"


def test_aligned_assessment_does_not_close_without_readiness():
    class FakeLLM:
        def generate(self, *, module, messages, mode):
            if module == "meal_tracker":
                return (
                    "[Tracking State]\n"
                    "- Confirmed food items: grilled chicken\n\n"
                    "[Published Meal Base]\n"
                    "- Food items: grilled chicken\n"
                    "- Ingredients: chicken breast\n"
                    "- Preparation methods: grilled\n"
                    "- Portions/amounts: palm-sized\n"
                    "- Beverages: none mentioned"
                )
            if module == "context_tracker":
                return "The user described part of a dinner."
            if module == "interaction_tracker":
                return (
                    '{"answered_facts": ["grilled chicken breast"], '
                    '"open_questions": ["side dish details"], '
                    '"rejected_options": [], '
                    '"unavailable_options": [], '
                    '"accepted_options": ["grilled chicken"], '
                    '"latest_user_position": "The user confirmed the protein."}'
                )
            if module == "alignment_estimator":
                return '{"answer": "1", "reasoning": "The protein is lean."}'
            if module == "meal_assessor" and mode == "assessment":
                return (
                    '{"summary": "The protein is grilled chicken breast.", '
                    '"strengths": ["lean protein"], '
                    '"limitations": ["side details are not resolved"], '
                    '"overall": "aligned"}'
                )
            if module == "dialogue_planner":
                return (
                    '{"intent_summary": "The user confirmed the protein.", '
                    '"user_intent": "informing", '
                    '"phase": "exploration", '
                    '"action": "assess", '
                    '"closure_readiness": "not_ready", '
                    '"reasoning": "Only one meal component is clear.", '
                    '"instruction": "", '
                    '"assessment_followup_action": "", '
                    '"confidence": 0.78}'
                )
            if module == "info_seeker":
                return (
                    '{"question_type": "side_detail", '
                    '"target": "side dishes", '
                    '"reasoning": "The full meal is not clear.", '
                    '"question_template": "What sides will you have with it?"}'
                )
            if module == "response_generator":
                system = messages[0]["content"]
                if "feedback message" in system:
                    return "The chicken is a strong lean-protein choice."
                return "What sides will you have with it?"
            raise AssertionError(f"Unexpected module: {module}, mode={mode}")

    result = ConversationEngine(FakeLLM().generate).generate_chat_replies(
        CoachingTurnRequest(
            current_message="Skinless grilled chicken breast, palm-sized.",
            enable_guardrail=False,
        )
    )

    assert [m.kind for m in result.assistant_messages] == ["assessment", "question"]
    assert result.state.status == "active"
    assert result.metadata["post_assessment_decision"]["action"] == "inquire"


def test_settled_recommendation_redirects_to_assessment():
    class FakeLLM:
        def generate(self, *, module, messages, mode):
            if module == "recommender":
                raise AssertionError("settled plan should be assessed before another recommendation")
            if module == "meal_tracker":
                return (
                    "[Tracking State]\n"
                    "- Confirmed food items: chicken salad with apple\n\n"
                    "[Published Meal Base]\n"
                    "- Food items: chicken salad, apple\n"
                    "- Ingredients: chicken, lettuce, apple\n"
                    "- Beverages: water"
                )
            if module == "context_tracker":
                return "The user has settled on chicken salad with apple."
            if module == "interaction_tracker":
                return (
                    '{"answered_facts": ["chicken salad with apple"], '
                    '"open_questions": [], '
                    '"rejected_options": [], '
                    '"unavailable_options": [], '
                    '"accepted_options": ["chicken salad with apple"], '
                    '"latest_user_position": "The user is settled."}'
                )
            if module == "dialogue_planner":
                return (
                    '{"intent_summary": "The user is settled on the current plan.", '
                    '"user_intent": "informing", '
                    '"phase": "recommendation", '
                    '"actionability": "settled", '
                    '"action": "recommend", '
                    '"closure_readiness": "ready_to_close", '
                    '"reasoning": "The current plan is settled.", '
                    '"instruction": "Recommend one more refinement.", '
                    '"assessment_followup_action": "", '
                    '"assessment_followup_phase": "", '
                    '"assessment_followup_instruction": "", '
                    '"confidence": 0.82}'
                )
            if module == "meal_assessor" and mode == "assessment":
                return (
                    '{"summary": "Chicken salad with apple and water.", '
                    '"strengths": ["lean protein", "fruit"], '
                    '"limitations": [], '
                    '"overall": "aligned"}'
                )
            if module == "response_generator":
                system = messages[0]["content"]
                if "feedback message" in system:
                    return "That plan is well aligned with your goal."
                return "Nice choice. You have a clear, workable dinner plan."
            raise AssertionError(f"Unexpected module: {module}, mode={mode}")

    result = ConversationEngine(FakeLLM().generate).generate_chat_replies(
        CoachingTurnRequest(
            current_message="The chicken salad with apple is fine.",
            enable_guardrail=False,
        )
    )

    assert [m.kind for m in result.assistant_messages] == ["assessment", "confirmation"]
    assert result.state.status == "active"
    assert result.metadata["planning_policy"]["override"] == (
        "settled_recommendation_redirected"
    )
    assert result.metadata["post_assessment_decision"]["action"] == "confirm"


def test_recent_actionable_recommendation_redirects_to_assessment():
    class FakeLLM:
        def generate(self, *, module, messages, mode):
            if module == "recommender":
                raise AssertionError("recent actionable plan should not repeat recommendation")
            if module == "meal_tracker":
                return (
                    "[Tracking State]\n"
                    "- Confirmed food items: brie with salad\n\n"
                    "[Published Meal Base]\n"
                    "- Food items: brie, salad\n"
                    "- Ingredients: brie, lettuce, cucumber\n"
                    "- Additional notes: buffet options are limited"
                )
            if module == "context_tracker":
                return "The user is working within limited buffet options."
            if module == "interaction_tracker":
                return (
                    '{"answered_facts": ["brie and salad are available"], '
                    '"open_questions": [], '
                    '"rejected_options": [], '
                    '"unavailable_options": ["other protein options"], '
                    '"accepted_options": ["brie with salad"], '
                    '"latest_user_position": "The user can proceed with brie and salad."}'
                )
            if module == "dialogue_planner":
                return (
                    '{"intent_summary": "The current buffet plan is workable.", '
                    '"user_intent": "informing", '
                    '"phase": "recommendation", '
                    '"actionability": "workable", '
                    '"action": "recommend", '
                    '"closure_readiness": "actionable", '
                    '"reasoning": "A workable plan already exists.", '
                    '"instruction": "Recommend another minor refinement.", '
                    '"assessment_followup_action": "", '
                    '"assessment_followup_phase": "", '
                    '"assessment_followup_instruction": "", '
                    '"confidence": 0.8}'
                )
            if module == "meal_assessor" and mode == "assessment":
                return (
                    '{"summary": "Brie with salad within buffet limits.", '
                    '"strengths": ["uses available foods"], '
                    '"limitations": [], '
                    '"overall": "aligned"}'
                )
            if module == "response_generator":
                system = messages[0]["content"]
                if "feedback message" in system:
                    return "Within the buffet limits, this is a workable choice."
                return "That sounds like a practical plan for the buffet."
            raise AssertionError(f"Unexpected module: {module}, mode={mode}")

    result = ConversationEngine(FakeLLM().generate).generate_chat_replies(
        CoachingTurnRequest(
            current_message="I can do brie with salad.",
            state=CoachingState(
                recommendation_history=(
                    {
                        "turn_idx": 1,
                        "recommendation_type": "composition",
                        "target_food": "brie with salad",
                        "suggestion": "pair brie with salad",
                    },
                )
            ),
            enable_guardrail=False,
        )
    )

    assert [m.kind for m in result.assistant_messages] == ["assessment", "confirmation"]
    assert result.state.status == "active"
    assert result.metadata["planning_policy"]["override"] == (
        "redundant_recommendation_redirected"
    )
    assert result.metadata["post_assessment_decision"]["action"] == "confirm"


def test_redundant_recommendation_redirect_blocks_post_assessment_recommendation():
    class FakeLLM:
        def generate(self, *, module, messages, mode):
            if module == "recommender":
                raise AssertionError("post-assessment gate should block repeated recommendation")
            if module == "meal_tracker":
                return (
                    "[Tracking State]\n"
                    "- Confirmed food items: brie with salad\n\n"
                    "[Published Meal Base]\n"
                    "- Food items: brie, salad\n"
                    "- Additional notes: buffet options are limited"
                )
            if module == "context_tracker":
                return "The user is working within a limited buffet context."
            if module == "interaction_tracker":
                return (
                    '{"answered_facts": ["brie and salad are available"], '
                    '"open_questions": [], '
                    '"rejected_options": [], '
                    '"unavailable_options": ["other protein options"], '
                    '"accepted_options": ["brie with salad"], '
                    '"latest_user_position": "The user can proceed with brie and salad."}'
                )
            if module == "dialogue_planner":
                return (
                    '{"intent_summary": "The current buffet plan is workable.", '
                    '"user_intent": "informing", '
                    '"phase": "recommendation", '
                    '"actionability": "workable", '
                    '"action": "recommend", '
                    '"closure_readiness": "ready_to_close", '
                    '"reasoning": "A workable plan already exists.", '
                    '"instruction": "Recommend another refinement.", '
                    '"assessment_followup_action": "recommend", '
                    '"assessment_followup_phase": "recommendation", '
                    '"assessment_followup_instruction": "Recommend another refinement.", '
                    '"confidence": 0.8}'
                )
            if module == "meal_assessor" and mode == "assessment":
                return (
                    '{"summary": "Brie with salad within buffet limits.", '
                    '"strengths": ["uses available foods"], '
                    '"limitations": ["not a very lean protein"], '
                    '"overall": "partially_aligned"}'
                )
            if module == "response_generator":
                system = messages[0]["content"]
                if "feedback message" in system:
                    return "This is workable within your buffet limit."
                return "Given what is available, this is a practical place to stop."
            raise AssertionError(f"Unexpected module: {module}, mode={mode}")

    result = ConversationEngine(FakeLLM().generate).generate_chat_replies(
        CoachingTurnRequest(
            current_message="Brie with salad is what I can do.",
            state=CoachingState(
                recommendation_history=(
                    {
                        "turn_idx": 1,
                        "recommendation_type": "composition",
                        "target_food": "brie with salad",
                        "suggestion": "pair brie with salad",
                    },
                )
            ),
            enable_guardrail=False,
        )
    )

    assert [m.kind for m in result.assistant_messages] == ["assessment", "confirmation"]
    assert result.metadata["post_assessment_gate"]["gate"] == (
        "post_assessment_redundant_recommendation_closed"
    )
    assert result.metadata["post_assessment_decision"]["action"] == "confirm"


def test_nonaccepting_redundant_recommendation_can_still_recommend_after_assessment():
    class FakeLLM:
        def generate(self, *, module, messages, mode):
            if module == "meal_tracker":
                return (
                    "[Tracking State]\n"
                    "- Confirmed food items: turkey sandwich and chips\n\n"
                    "[Published Meal Base]\n"
                    "- Food items: turkey sandwich, chips"
                )
            if module == "context_tracker":
                return "The user added allergy and preference constraints."
            if module == "interaction_tracker":
                return (
                    '{"answered_facts": ["turkey sandwich and chips", "tomato allergy"], '
                    '"open_questions": ["whether the user wants produce add-ons"], '
                    '"rejected_options": ["tomatoes", "raw onions"], '
                    '"unavailable_options": ["tomatoes"], '
                    '"candidate_options": ["baby carrots", "grapes"], '
                    '"accepted_options": ["turkey sandwich", "chips"], '
                    '"latest_user_position": "The user added constraints, not commitment."}'
                )
            if module == "dialogue_planner":
                return (
                    '{"intent_summary": "The user added constraints without accepting the previous recommendation.", '
                    '"user_intent": "informing", '
                    '"phase": "recommendation", '
                    '"actionability": "workable", '
                    '"action": "recommend", '
                    '"closure_readiness": "actionable", '
                    '"reasoning": "A workable recommendation can be adjusted to constraints.", '
                    '"instruction": "Recommend within constraints.", '
                    '"assessment_followup_action": "recommend", '
                    '"assessment_followup_phase": "recommendation", '
                    '"assessment_followup_instruction": "Recommend within constraints.", '
                    '"confidence": 0.8}'
                )
            if module == "meal_assessor" and mode == "assessment":
                return (
                    '{"summary": "Turkey sandwich and chips with new constraints.", '
                    '"strengths": ["clear constraints"], '
                    '"limitations": ["produce add-on unresolved"], '
                    '"overall": "partially_aligned"}'
                )
            if module == "recommender":
                return (
                    '{"recommendation_type": "addition", '
                    '"target_food": "produce side", '
                    '"suggestion": "add grapes or baby carrots", '
                    '"reasoning": "Fits constraints.", '
                    '"expected_impact": "adds fruit or vegetables"}'
                )
            if module == "response_generator":
                system = messages[0]["content"]
                if "feedback message" in system:
                    return "The constraints are clear."
                return "Try grapes or baby carrots with the sandwich."
            raise AssertionError(f"Unexpected module: {module}, mode={mode}")

    result = ConversationEngine(FakeLLM().generate).generate_chat_replies(
        CoachingTurnRequest(
            current_message="I'm allergic to tomatoes, and I don't like raw onions.",
            state=CoachingState(
                recommendation_history=(
                    {
                        "turn_idx": 0,
                        "recommendation_type": "addition",
                        "target_food": "produce side",
                        "suggestion": "add carrots and grapes",
                    },
                )
            ),
            enable_guardrail=False,
        )
    )

    assert result.metadata["planning_policy"]["override"] == (
        "redundant_recommendation_redirected"
    )
    assert "post_assessment_gate" not in result.metadata
    assert result.metadata["post_assessment_decision"]["action"] == "recommend"
    assert [m.kind for m in result.assistant_messages] == [
        "assessment",
        "recommendation",
    ]


def test_accepted_plan_blocks_post_assessment_replacement_recommendation():
    class FakeLLM:
        def generate(self, *, module, messages, mode):
            if module == "recommender":
                raise AssertionError("accepted plan should not be replaced after assessment")
            if module == "meal_tracker":
                return (
                    "[Tracking State]\n"
                    "- Confirmed food items: turkey sandwich with apple\n\n"
                    "[Published Meal Base]\n"
                    "- Food items: turkey sandwich, apple\n"
                    "- Additional notes: user accepted apple as easy fruit option"
                )
            if module == "context_tracker":
                return "The user accepted apple as the easiest fruit addition."
            if module == "interaction_tracker":
                return (
                    '{"answered_facts": ["turkey sandwich", "apple is easy"], '
                    '"open_questions": [], '
                    '"rejected_options": [], '
                    '"unavailable_options": [], '
                    '"accepted_options": ["apple"], '
                    '"latest_user_position": "The user accepted apple."}'
                )
            if module == "dialogue_planner":
                return (
                    '{"intent_summary": "The user accepted apple as the add-on.", '
                    '"user_intent": "accepting", '
                    '"phase": "recommendation", '
                    '"actionability": "workable", '
                    '"action": "assess", '
                    '"closure_readiness": "actionable", '
                    '"reasoning": "Assess the accepted plan.", '
                    '"instruction": "Assess the accepted plan.", '
                    '"assessment_followup_action": "recommend", '
                    '"assessment_followup_phase": "recommendation", '
                    '"assessment_followup_instruction": "Recommend a different vegetable add-on.", '
                    '"confidence": 0.8}'
                )
            if module == "meal_assessor" and mode == "assessment":
                return (
                    '{"summary": "Turkey sandwich with apple.", '
                    '"strengths": ["adds fruit"], '
                    '"limitations": ["could add more vegetables"], '
                    '"overall": "partially_aligned"}'
                )
            if module == "response_generator":
                system = messages[0]["content"]
                if "feedback message" in system:
                    return "The apple is a useful fruit addition."
                return "That is a clear, doable improvement for lunch."
            raise AssertionError(f"Unexpected module: {module}, mode={mode}")

    result = ConversationEngine(FakeLLM().generate).generate_chat_replies(
        CoachingTurnRequest(
            current_message="The apple sounds easiest.",
            enable_guardrail=False,
        )
    )

    assert [m.kind for m in result.assistant_messages] == ["assessment", "confirmation"]
    assert result.metadata["post_assessment_gate"]["gate"] == (
        "post_assessment_accepted_plan_closed"
    )
    assert result.metadata["post_assessment_decision"]["action"] == "confirm"


def test_safety_boundary_close_uses_reflective_finalization_style():
    class FakeLLM:
        def generate(self, *, module, messages, mode):
            if module == "meal_tracker":
                return (
                    "[Tracking State]\n"
                    "- Confirmed food items: cheese pizza\n\n"
                    "[Published Meal Base]\n"
                    "- Food items: cheese pizza"
                )
            if module == "context_tracker":
                return "The user is choosing a food that conflicts with an allergy."
            if module == "interaction_tracker":
                return (
                    '{"answered_facts": ["cheese pizza"], '
                    '"open_questions": [], '
                    '"rejected_options": ["safe alternatives"], '
                    '"unavailable_options": [], '
                    '"accepted_options": ["cheese pizza"], '
                    '"latest_user_position": "The user wants to keep cheese pizza."}'
                )
            if module == "dialogue_planner":
                return (
                    '{"intent_summary": "The user wants to keep the conflicted food.", '
                    '"user_intent": "disengaging", '
                    '"phase": "negotiation", '
                    '"actionability": "boundary", '
                    '"action": "close", '
                    '"closure_readiness": "boundary_close", '
                    '"reasoning": "The boundary is clear.", '
                    '"instruction": "Close with safety concern acknowledged.", '
                    '"assessment_followup_action": "", '
                    '"assessment_followup_phase": "", '
                    '"assessment_followup_instruction": "", '
                    '"confidence": 0.8}'
                )
            if module == "response_generator":
                content = messages[0]["content"]
                assert "Finalization style: **reflective**" in content
                return "I hear your choice, but I cannot frame cheese as safe with your stored allergy. Please use your medical guidance and choose the safest option available."
            raise AssertionError(f"Unexpected module: {module}, mode={mode}")

    result = ConversationEngine(FakeLLM().generate).generate_chat_replies(
        CoachingTurnRequest(
            current_message="I still want regular cheese pizza.",
            profile=UserProfileContext(allergies=("cheese",)),
            state=CoachingState(
                safety_clarification_counts={"allergy:cheese": 1},
            ),
            enable_guardrail=False,
        )
    )

    assert result.status == "terminated"
    assert result.metadata["finalization_style"] == "reflective"
    assert result.assistant_messages[-1].metadata["finalization_style"] == "reflective"


def test_assessment_response_receives_previous_recommendation_context():
    class FakeLLM:
        def generate(self, *, module, messages, mode):
            if module == "meal_tracker":
                return (
                    "[Tracking State]\n"
                    "- Confirmed food items: chicken, rice, salad\n\n"
                    "[Published Meal Base]\n"
                    "- Food items: chicken, rice, salad"
                )
            if module == "context_tracker":
                return "The user accepted the lighter dressing suggestion."
            if module == "interaction_tracker":
                return (
                    '{"answered_facts": ["chicken, rice, salad"], '
                    '"open_questions": [], '
                    '"rejected_options": [], '
                    '"unavailable_options": [], '
                    '"accepted_options": ["light dressing"], '
                    '"latest_user_position": "The user accepted light dressing."}'
                )
            if module == "dialogue_planner":
                return (
                    '{"intent_summary": "The user accepted the lighter dressing.", '
                    '"user_intent": "accepting", '
                    '"phase": "recommendation", '
                    '"actionability": "settled", '
                    '"action": "assess", '
                    '"closure_readiness": "ready_to_close", '
                    '"reasoning": "Assess and close the settled plan.", '
                    '"instruction": "Assess without adding another refinement.", '
                    '"assessment_followup_action": "close", '
                    '"assessment_followup_phase": "motivational_ending", '
                    '"assessment_followup_instruction": "Close without repeating the dressing suggestion.", '
                    '"confidence": 0.8}'
                )
            if module == "meal_assessor" and mode == "assessment":
                return (
                    '{"summary": "Chicken, rice, salad, and light dressing.", '
                    '"strengths": ["lean protein"], '
                    '"limitations": [], '
                    '"overall": "aligned"}'
                )
            if module == "response_generator":
                content = messages[1]["content"]
                if "[Assessment Result]" in content:
                    assert "[Previous Recommendations Already Given]" in content
                    assert "use light dressing" in content
                    return "This plan is now well settled."
                assert "confirmation message" in messages[0]["content"]
                assert "chicken, rice, salad" in content
                return "I have chicken, rice, salad, and light dressing. Does that look right before we wrap up?"
            raise AssertionError(f"Unexpected module: {module}, mode={mode}")

    result = ConversationEngine(FakeLLM().generate).generate_chat_replies(
        CoachingTurnRequest(
            current_message="Yes, light dressing works for me.",
            state=CoachingState(
                recommendation_history=(
                    {
                        "turn_idx": 2,
                        "recommendation_type": "modify",
                        "target_food": "dressing",
                        "suggestion": "use light dressing",
                    },
                )
            ),
            enable_guardrail=False,
        )
    )

    assert [m.kind for m in result.assistant_messages] == ["confirmation"]
    assert result.metadata["assessment_saturation_gate"]["gate"] == (
        "assessment_saturation_to_confirmation"
    )
    assert result.state.phase == "confirmation"


def test_post_assessment_close_deferred_when_open_question_remains():
    class FakeLLM:
        def generate(self, *, module, messages, mode):
            if module == "meal_tracker":
                return (
                    "[Tracking State]\n"
                    "- Confirmed food items: chicken, rice, salad\n\n"
                    "[Published Meal Base]\n"
                    "- Food items: chicken, rice, salad"
                )
            if module == "context_tracker":
                return "The user has not confirmed final portion or dressing."
            if module == "interaction_tracker":
                return (
                    '{"answered_facts": ["chicken, rice, salad"], '
                    '"open_questions": ["dressing choice"], '
                    '"rejected_options": [], '
                    '"unavailable_options": [], '
                    '"accepted_options": ["chicken", "rice", "salad"], '
                    '"latest_user_position": "The user added meal facts but has not committed."}'
                )
            if module == "dialogue_planner":
                return (
                    '{"intent_summary": "The user provided another meal fact.", '
                    '"user_intent": "informing", '
                    '"phase": "recommendation", '
                    '"actionability": "workable", '
                    '"action": "assess", '
                    '"closure_readiness": "ready_to_close", '
                    '"reasoning": "Assess and close.", '
                    '"instruction": "Assess.", '
                    '"assessment_followup_action": "close", '
                    '"assessment_followup_phase": "motivational_ending", '
                    '"assessment_followup_instruction": "Close.", '
                    '"confidence": 0.8}'
                )
            if module == "meal_assessor" and mode == "assessment":
                return (
                    '{"summary": "Chicken, rice, and salad.", '
                    '"strengths": ["lean protein"], '
                    '"limitations": ["dressing unresolved"], '
                    '"overall": "aligned"}'
                )
            if module == "info_seeker":
                return (
                    '{"question_type": "detail", '
                    '"target": "dressing", '
                    '"reasoning": "It affects the final guidance.", '
                    '"question_template": "What dressing will you use?"}'
                )
            if module == "response_generator":
                if "feedback message" in messages[0]["content"]:
                    return "The plan is mostly workable."
                return "What dressing will you use?"
            raise AssertionError(f"Unexpected module: {module}, mode={mode}")

    result = ConversationEngine(FakeLLM().generate).generate_chat_replies(
        CoachingTurnRequest(
            current_message="The chicken is grilled.",
            enable_guardrail=False,
        )
    )

    assert [m.kind for m in result.assistant_messages] == ["assessment", "question"]
    assert result.metadata["post_assessment_gate"]["gate"] == (
        "post_assessment_open_question_close_deferred"
    )
    assert result.metadata["post_assessment_decision"]["action"] == "inquire"
    assert result.status == "active"


def test_rejected_boundary_close_uses_educational_finalization_style():
    class FakeLLM:
        def generate(self, *, module, messages, mode):
            if module == "meal_tracker":
                return (
                    "[Tracking State]\n"
                    "- Confirmed food items: tofu, broccoli, mushrooms\n\n"
                    "[Published Meal Base]\n"
                    "- Food items: tofu, broccoli, mushrooms"
                )
            if module == "context_tracker":
                return "The user wants a no-starch dinner."
            if module == "interaction_tracker":
                return (
                    '{"answered_facts": ["tofu, broccoli, mushrooms"], '
                    '"open_questions": [], '
                    '"rejected_options": ["rice", "bread", "pasta", "potatoes"], '
                    '"unavailable_options": [], '
                    '"accepted_options": ["tofu with broccoli and mushrooms"], '
                    '"latest_user_position": "The user wants the no-starch plan."}'
                )
            if module == "dialogue_planner":
                return (
                    '{"intent_summary": "The user accepted the no-starch plan.", '
                    '"user_intent": "accepting", '
                    '"phase": "negotiation", '
                    '"actionability": "settled", '
                    '"action": "assess", '
                    '"closure_readiness": "ready_to_close", '
                    '"reasoning": "Assess and close while respecting boundary.", '
                    '"instruction": "Assess.", '
                    '"assessment_followup_action": "close", '
                    '"assessment_followup_phase": "motivational_ending", '
                    '"assessment_followup_instruction": "Close while noting the carb-goal limitation.", '
                    '"confidence": 0.8}'
                )
            if module == "meal_assessor" and mode == "assessment":
                return (
                    '{"summary": "Tofu with broccoli and mushrooms.", '
                    '"strengths": ["uses preferred foods"], '
                    '"limitations": ["does not include a starchy carb portion"], '
                    '"overall": "partially_aligned"}'
                )
            if module == "response_generator":
                if "feedback message" in messages[0]["content"]:
                    return "This works within the user's boundary."
                assert "confirmation message" in messages[0]["content"]
                return "I have tofu with broccoli and mushrooms as your plan. Does that look right before we wrap up?"
            raise AssertionError(f"Unexpected module: {module}, mode={mode}")

    result = ConversationEngine(FakeLLM().generate).generate_chat_replies(
        CoachingTurnRequest(
            current_message="Tofu with broccoli and mushrooms is what I want.",
            enable_guardrail=False,
        )
    )

    assert result.status == "active"
    assert result.metadata["post_assessment_decision"]["action"] == "confirm"
    assert result.state.phase == "confirmation"


def test_intent_policy_hands_off_after_repeated_rejection():
    class FakeLLM:
        def generate(self, *, module, messages, mode):
            if module == "meal_tracker":
                return (
                    "[Tracking State]\n"
                    "- Confirmed food items: pizza\n"
                    "- Tentative food items: none\n"
                    "- Rejected food items: none\n"
                    "- Decision context: user rejected alternatives\n\n"
                    "[Published Meal Base]\n"
                    "- Food items: pizza\n"
                    "- Ingredients: cheese, pepperoni\n"
                    "- Preparation methods: baked\n"
                    "- Portions/amounts: not yet mentioned\n"
                    "- Beverages: none mentioned\n"
                    "- Additional notes: not yet mentioned"
                )
            if module == "context_tracker":
                return "The user does not want more recommendations."
            if module == "interaction_tracker":
                return (
                    '{"answered_facts": ["the user rejected changing the meal"], '
                    '"open_questions": [], '
                    '"rejected_options": ["more recommendations"], '
                    '"unavailable_options": [], '
                    '"accepted_options": [], '
                    '"latest_user_position": "The user does not want another change."}'
                )
            if module == "alignment_estimator":
                return '{"answer": "0", "reasoning": "The meal is not aligned."}'
            if module == "dialogue_planner":
                return (
                    '{"intent_summary": "The user rejects another suggestion.", '
                    '"user_intent": "rejecting", '
                    '"phase": "negotiation", '
                    '"action": "recommend", '
                    '"reasoning": "They are resisting recommendations.", '
                    '"instruction": "Respect the user choice.", '
                    '"assessment_followup_action": "", '
                    '"assessment_followup_phase": "", '
                    '"assessment_followup_instruction": "", '
                    '"confidence": 0.91}'
                )
            if module == "response_generator":
                assert "control-handoff message" in messages[0]["content"]
                return "Would you like another idea, keep the pizza as-is, or stop here?"
            raise AssertionError(f"Unexpected module: {module}")

    result = ConversationEngine(FakeLLM().generate).generate_chat_replies(
            CoachingTurnRequest(
                current_message="No, I don't want to change it.",
                state=CoachingState(
                    phase="recommendation",
                    recommendation_rejection_count=2,
                ),
                enable_guardrail=False,
            )
    )

    assert result.state.last_user_intent == "rejecting"
    assert result.state.recommendation_rejection_count == 3
    assert result.state.phase == "negotiation"
    assert result.state.status == "active"
    assert result.terminated_by is None
    assert result.metadata["intent_policy"]["override"] == (
        "resistance_threshold_handoff"
    )
    assert result.assistant_messages[-1].kind == "handoff"


def test_interaction_tracker_incremental_input_uses_latest_turn_only():
    class FakeLLM:
        def generate(self, *, module, messages, mode):
            if module == "interaction_tracker":
                content = messages[1]["content"]
                assert "Previous interaction_state:" in content
                assert "New conversation turns:" in content
                assert "User: Previous meal detail." not in content
                assert "User: Current meal detail." in content
                return (
                    '{"answered_facts": ["current meal detail"], '
                    '"open_questions": [], '
                    '"rejected_options": [], '
                    '"unavailable_options": [], '
                    '"accepted_options": ["current meal detail"], '
                    '"latest_user_position": "The user added current detail."}'
                )
            if module == "meal_tracker":
                content = messages[1]["content"]
                assert "Previous tracking state:" in content
                assert "previous tracked meal" in content
                assert "User: Previous meal detail." not in content
                assert "User: Current meal detail." in content
                return (
                    "[Tracking State]\n"
                    "- Confirmed food items: current meal detail\n\n"
                    "[Published Meal Base]\n"
                    "- Food items: current meal detail"
                )
            if module == "context_tracker":
                return "The user added current meal detail."
            if module == "alignment_estimator":
                return '{"answer": "1", "reasoning": "The meal detail is aligned."}'
            if module == "dialogue_planner":
                return (
                    '{"intent_summary": "The user added current detail.", '
                    '"user_intent": "informing", '
                    '"phase": "exploration", '
                    '"action": "respond", '
                    '"closure_readiness": "not_ready", '
                    '"reasoning": "Acknowledge the detail.", '
                    '"instruction": "Acknowledge briefly.", '
                    '"assessment_followup_action": "", '
                    '"confidence": 0.8}'
                )
            if module == "response_generator":
                return "Got it."
            raise AssertionError(f"Unexpected module: {module}, mode={mode}")

    result = ConversationEngine(FakeLLM().generate).generate_chat_replies(
        CoachingTurnRequest(
            current_message="Current meal detail.",
            history=[
                ChatMessage("assistant", "What did you have before?"),
                ChatMessage("user", "Previous meal detail."),
                ChatMessage("assistant", "What else?"),
            ],
            state=CoachingState(
                tracker_state=(
                    "[Tracking State]\n"
                    "- Confirmed food items: previous tracked meal"
                ),
                interaction_state="Answered facts:\n- previous meal detail",
            ),
            enable_guardrail=False,
        )
    )

    assert "current meal detail" in result.state.interaction_state


def test_new_recommendation_is_grounded_by_assessment_bubble():
    class FakeLLM:
        def generate(self, *, module, messages, mode):
            if module == "meal_tracker":
                return (
                    "[Tracking State]\n"
                    "- Confirmed food items: chicken sandwich and chips\n\n"
                    "[Published Meal Base]\n"
                    "- Food items: chicken sandwich, chips"
                )
            if module == "context_tracker":
                return "The user described a chicken sandwich and chips."
            if module == "interaction_tracker":
                return (
                    '{"answered_facts": ["chicken sandwich and chips"], '
                    '"open_questions": [], '
                    '"rejected_options": [], '
                    '"unavailable_options": [], '
                    '"candidate_options": [], '
                    '"accepted_options": [], '
                    '"latest_user_position": "The user has a workable meal."}'
                )
            if module == "dialogue_planner":
                return (
                    '{"intent_summary": "The user has a workable meal for recommendation.", '
                    '"user_intent": "informing", '
                    '"phase": "recommendation", '
                    '"actionability": "workable", '
                    '"action": "recommend", '
                    '"closure_readiness": "actionable", '
                    '"reasoning": "A recommendation can help.", '
                    '"instruction": "Suggest a small produce add-on.", '
                    '"assessment_followup_action": "", '
                    '"assessment_followup_phase": "", '
                    '"assessment_followup_instruction": "", '
                    '"confidence": 0.85}'
                )
            if module == "meal_assessor" and mode == "assessment":
                return (
                    '{"summary": "Chicken sandwich and chips.", '
                    '"strengths": ["clear meal base"], '
                    '"limitations": ["no fruit or vegetable noted"], '
                    '"overall": "partially_aligned"}'
                )
            if module == "recommender":
                return (
                    '{"recommendation_type": "add", '
                    '"target_food": "produce side", '
                    '"suggestion": "add a small apple or side salad", '
                    '"reasoning": "It addresses the assessment limitation.", '
                    '"expected_impact": "medium"}'
                )
            if module == "response_generator":
                system = messages[0]["content"]
                user = messages[1]["content"]
                if "feedback message" in system:
                    return "Your meal has a clear base, but it is missing produce."
                if "recommendation message" in system:
                    assert "[Current Assessment]" in user
                    assert "no fruit or vegetable noted" in user
                    return "So, how about adding a small apple or side salad?"
            raise AssertionError(f"Unexpected module: {module}, mode={mode}")

    result = ConversationEngine(FakeLLM().generate).generate_chat_replies(
        CoachingTurnRequest(
            current_message="I have a chicken sandwich and chips.",
            nutrition_goal="half_fruits_vegetables",
            enable_guardrail=False,
        )
    )

    assert [m.kind for m in result.assistant_messages] == [
        "assessment",
        "recommendation",
    ]
    assert result.metadata["planning_policy"]["effective_action"] == "assess"
    assert result.metadata["planning_policy"]["override"] == (
        "recommendation_grounded_by_assessment"
    )
    assert result.metadata["post_assessment_decision"]["action"] == "recommend"
    assert result.state.phase == "recommendation"


def test_profile_constraints_are_visible_to_question_generator():
    class FakeLLM:
        def generate(self, *, module, messages, mode):
            if module == "meal_tracker":
                return (
                    "[Tracking State]\n"
                    "- Confirmed food items: cereal\n\n"
                    "[Published Meal Base]\n"
                    "- Food items: cereal"
                )
            if module == "context_tracker":
                return "The user has cereal. The known profile includes an egg allergy."
            if module == "interaction_tracker":
                return (
                    '{"answered_facts": ["cereal"], '
                    '"open_questions": ["milk type"], '
                    '"rejected_options": [], '
                    '"unavailable_options": [], '
                    '"candidate_options": [], '
                    '"accepted_options": [], '
                    '"latest_user_position": "The user described cereal."}'
                )
            if module == "dialogue_planner":
                return (
                    '{"intent_summary": "The user described cereal.", '
                    '"user_intent": "informing", '
                    '"phase": "exploration", '
                    '"actionability": "insufficient", '
                    '"action": "inquire", '
                    '"closure_readiness": "not_ready", '
                    '"reasoning": "Need the milk type.", '
                    '"instruction": "Ask about the milk or liquid used.", '
                    '"assessment_followup_action": "", '
                    '"assessment_followup_phase": "", '
                    '"assessment_followup_instruction": "", '
                    '"confidence": 0.8}'
                )
            if module == "info_seeker":
                system = messages[0]["content"]
                assert "Allergies: egg" in system
                assert "DO NOT ask about information already listed here" in system
                assert "Known profile constraints:" in system
                return (
                    '{"question_type": "ingredient", '
                    '"target": "milk or liquid", '
                    '"reasoning": "The liquid affects the meal.", '
                    '"question_template": "What milk or liquid are you using with the cereal?"}'
                )
            if module == "response_generator":
                return "What milk or liquid are you using with the cereal?"
            raise AssertionError(f"Unexpected module: {module}, mode={mode}")

    result = ConversationEngine(FakeLLM().generate).generate_chat_replies(
        CoachingTurnRequest(
            current_message="I am having cereal.",
            profile=UserProfileContext(allergies=["egg"]),
            enable_guardrail=False,
        )
    )

    assert result.assistant_messages[-1].kind == "question"
    assert "Allergy constraint: egg" in result.state.interaction_state
    assert "Allergies: egg" in result.state.user_preferences


def test_profile_allergy_alias_triggers_safety_gate_for_omelet():
    class FakeLLM:
        def generate(self, *, module, messages, mode):
            if module in {"meal_assessor", "recommender", "info_seeker"}:
                raise AssertionError(f"safety gate should not call {module}")
            if module == "meal_tracker":
                return (
                    "[Tracking State]\n"
                    "- Confirmed food items: cheese omelet\n\n"
                    "[Published Meal Base]\n"
                    "- Food items: cheese omelet\n"
                    "- Ingredients: cheese, omelet"
                )
            if module == "context_tracker":
                return "The user mentioned a cheese omelet and has a stored egg allergy."
            if module == "interaction_tracker":
                return (
                    '{"answered_facts": ["cheese omelet"], '
                    '"open_questions": [], '
                    '"rejected_options": [], '
                    '"unavailable_options": [], '
                    '"candidate_options": [], '
                    '"accepted_options": ["cheese omelet"], '
                    '"latest_user_position": "The user is considering a cheese omelet."}'
                )
            if module == "dialogue_planner":
                return (
                    '{"intent_summary": "The user is considering an omelet.", '
                    '"user_intent": "accepting", '
                    '"phase": "recommendation", '
                    '"actionability": "settled", '
                    '"action": "recommend", '
                    '"closure_readiness": "ready_to_close", '
                    '"reasoning": "The user chose the omelet.", '
                    '"instruction": "Recommend the omelet.", '
                    '"assessment_followup_action": "", '
                    '"assessment_followup_phase": "", '
                    '"assessment_followup_instruction": "", '
                    '"confidence": 0.8}'
                )
            if module == "response_generator":
                content = messages[1]["content"]
                assert "stored allergy constraint (egg)" in content
                return (
                    "I have egg listed as an allergy, so I can’t treat the omelet "
                    "as safe without clarification. Is there an egg-free option?"
                )
            raise AssertionError(f"Unexpected module: {module}, mode={mode}")

    result = ConversationEngine(FakeLLM().generate).generate_chat_replies(
        CoachingTurnRequest(
            current_message="I think the cheese omelet should be fine.",
            profile=UserProfileContext(allergies=["egg"]),
            enable_guardrail=False,
        )
    )

    assert result.metadata["commitment_gate"]["gate"] == "safety_conflict_clarification"
    assert result.metadata["commitment_gate"]["constraint"] == "egg"
    assert result.metadata["planning_policy"]["effective_phase"] == "exploration"
    assert result.state.phase == "exploration"
    assert result.assistant_messages[-1].kind == "answer"


def test_user_requested_safety_conflict_uses_cautious_continuation_recommendation():
    class FakeLLM:
        def generate(self, *, module, messages, mode):
            if module == "meal_tracker":
                return (
                    "[Tracking State]\n"
                    "- Confirmed food items: plain yogurt, blueberry pancake, potatoes, cheese omelet\n"
                    "- Decision context: user insists on omelet despite egg allergy\n\n"
                    "[Published Meal Base]\n"
                    "- Food items: plain yogurt, blueberry pancake, potatoes, cheese omelet\n"
                    "- Ingredients: yogurt, blueberries, pancake, potatoes, eggs, cheese\n"
                    "- Portions/amounts: yogurt 1 cup, small pancake, modest potatoes"
                )
            if module == "context_tracker":
                return "The user wants the omelet because their mom cooked it, despite a stored egg allergy."
            if module == "interaction_tracker":
                return (
                    '{"answered_facts": ["user wants omelet because mom cooked it"], '
                    '"open_questions": [], '
                    '"rejected_options": [], '
                    '"unavailable_options": [], '
                    '"safety_conflicted_options": ["eggs", "cheese omelet"], '
                    '"user_requested_conflicted_options": ["omelet is okay", "a little bit eggs"], '
                    '"candidate_options": ["plain yogurt", "blueberry pancake", "potatoes"], '
                    '"accepted_options": ["plain yogurt", "small pancake", "modest potatoes"], '
                    '"meal_slots": ["omelet is safety-conflicted but user still requests it"], '
                    '"active_issue": "User explicitly wants the egg omelet despite allergy; do not repeat removal as the main advice.", '
                    '"latest_user_position": "The user insists the omelet is okay and wants to enjoy it."}'
                )
            if module == "dialogue_planner":
                return (
                    '{"intent_summary": "User insists on a conflicted omelet.", '
                    '"user_intent": "rejecting", '
                    '"phase": "negotiation", '
                    '"actionability": "conflicted", '
                    '"action": "recommend", '
                    '"closure_readiness": "not_ready", '
                    '"reasoning": "A safety conflict remains.", '
                    '"instruction": "Continue safely.", '
                    '"assessment_followup_action": "", '
                    '"assessment_followup_phase": "", '
                    '"assessment_followup_instruction": "", '
                    '"confidence": 0.8}'
                )
            if module == "meal_assessor" and mode == "assessment":
                return (
                    '{"summary": "The user wants an omelet despite an egg allergy; the rest of dinner can still be adjusted.", '
                    '"strengths": ["Plain yogurt supports the goal"], '
                    '"limitations": ["The omelet remains safety-conflicted"], '
                    '"overall": "partially_aligned"}'
                )
            if module == "recommender":
                system = messages[0]["content"]
                assert "cautious_continuation" in system
                assert "repeat a removal recommendation" in system
                assert "user_requested_conflicted_options" in system
                assert "omelet is okay" in system
                assert "a little bit eggs" in system
                return (
                    '{"recommendation_type": "cautious_continuation", '
                    '"target_food": "overall meal plan", '
                    '"suggestion": "If the user still includes a small amount of omelet, focus the rest of dinner on yogurt, small pancake, modest potatoes, and no coffee.", '
                    '"reasoning": "The omelet remains safety-conflicted, so the recommendation cannot endorse it as safe; the actionable focus is the non-conflicted meal.", '
                    '"expected_impact": "medium", '
                    '"options": [{'
                    '"option_id": "opt1", '
                    '"target_food": "non-conflicted dinner components", '
                    '"suggestion": "Keep yogurt as the main protein support and keep pancake and potatoes modest.", '
                    '"reasoning": "This supports lean protein without repeating removal of the conflicted omelet.", '
                    '"expected_impact": "medium"'
                    '}]}'
                )
            if module == "response_generator":
                system = messages[0]["content"]
                if "feedback message" in system:
                    return (
                        "It makes sense that you want to enjoy what your mom made. "
                        "The omelet still cannot be treated as safe because of your egg allergy."
                    )
                assert "cautious_continuation" in messages[1]["content"]
                return (
                    "I can’t call the omelet safe with your egg allergy. "
                    "If you still include a little, keep the rest of your plan centered on yogurt, "
                    "a small pancake, and modest potatoes."
                )
            raise AssertionError(f"Unexpected module: {module}, mode={mode}")

    result = ConversationEngine(FakeLLM().generate).generate_chat_replies(
        CoachingTurnRequest(
            current_message="No, I said the omelet is okay. My mom cooked it and I want to enjoy it.",
            profile=UserProfileContext(allergies=["egg"]),
            state=CoachingState(safety_clarification_counts={"allergy:egg": 1}),
            enable_guardrail=False,
        )
    )

    assert result.metadata["commitment_gate"]["gate"] == "safety_conflict_to_safe_plan"
    assert result.metadata["recommendation_result"]["recommendation_type"] == "cautious_continuation"
    assert result.assistant_messages[-1].kind == "recommendation"
    assert "Remove the omelet" not in result.assistant_messages[-1].content
    assert "I can’t call the omelet safe" in result.assistant_messages[-1].content
