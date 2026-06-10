from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
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
            if module == "alignment_estimator":
                return '{"answer": "0", "reasoning": "More details are needed."}'
            if module == "phase_predictor":
                return (
                    '{"predicted_phase": "exploration", '
                    '"confidence": 0.82, '
                    '"reasoning": "Meal details are still being gathered."}'
                )
            if module == "orchestrator":
                return (
                    '{"intent_summary": "The user shared their meal.", '
                    '"user_intent": "informing", '
                    '"accepted_phase": "exploration", '
                    '"action": "inquire", '
                    '"reasoning": "Need preparation details.", '
                    '"instruction": "Ask about preparation."}'
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
    assert result.state.last_user_intent == "informing"
    assert result.state.stall_count == 0
    assert result.state.recommendation_rejection_count == 0
    assert result.metadata["intent_policy"]["user_intent"] == "informing"
    assert result.primary_message is result.assistant_messages[-1]


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
            if module == "alignment_estimator":
                return '{"answer": "0", "reasoning": "Vegetables could be improved."}'
            if module == "phase_predictor":
                return (
                    '{"predicted_phase": "exploration", '
                    '"confidence": 0.76, '
                    '"reasoning": "The meal can now be assessed."}'
                )
            if module == "orchestrator" and mode == "assessment":
                return (
                    '{"summary": "A turkey sandwich with water.", '
                    '"strengths": ["lean protein"], '
                    '"limitations": ["could add more vegetables"], '
                    '"overall": "partially_aligned"}'
                )
            if module == "orchestrator":
                system = messages[0]["content"]
                if "completed a meal evaluation" in system:
                    return (
                        '{"action": "inquire", '
                        '"accepted_phase": "recommendation", '
                        '"reasoning": "Need preference context.", '
                        '"instruction": "Ask about vegetable preferences."}'
                    )
                return (
                    '{"intent_summary": "The user gave enough meal detail.", '
                    '"user_intent": "informing", '
                    '"accepted_phase": "exploration", '
                    '"action": "assess", '
                    '"reasoning": "The meal can now be assessed.", '
                    '"instruction": ""}'
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
    assert result.state.phase == "recommendation"
    assert result.state.status == "active"
    assert result.metadata["post_assessment_decision"]["action"] == "inquire"


def test_intent_policy_closes_after_repeated_rejection():
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
            if module == "alignment_estimator":
                return '{"answer": "0", "reasoning": "The meal is not aligned."}'
            if module == "phase_predictor":
                return (
                    '{"predicted_phase": "negotiation", '
                    '"confidence": 0.91, '
                    '"reasoning": "The user is rejecting suggestions."}'
                )
            if module == "orchestrator":
                return (
                    '{"intent_summary": "The user rejects another suggestion.", '
                    '"user_intent": "rejecting", '
                    '"accepted_phase": "negotiation", '
                    '"action": "recommend", '
                    '"reasoning": "They are resisting recommendations.", '
                    '"instruction": "Respect the user choice."}'
                )
            if module == "response_generator":
                return "Thanks for talking it through. You know what works for you."
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
    assert result.state.phase == "motivational_ending"
    assert result.state.status == "terminated"
    assert result.terminated_by == "close"
    assert result.metadata["intent_policy"]["override"] == (
        "resistance_threshold_close"
    )
