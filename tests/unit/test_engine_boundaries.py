from code_interactive.agents.contracts import (
    CoachingState,
    CoachingTurnRequest,
    UserProfileContext,
)
from code_interactive.agents.engine import ConversationEngine


def test_stop_boundary_detects_confirmed_wrap_language():
    text = "Yes, that's accurate. I'm happy to wrap it there."

    assert ConversationEngine._message_sets_stop_boundary(text)


def test_preservation_boundary_detects_accepted_compromise_language():
    text = "That works for me. I'll stick with that dairy-free smoothie as planned."

    assert ConversationEngine._message_preserves_current_plan_boundary(text)


def test_late_added_food_is_material_update_even_after_acceptance():
    text = "That works, but I also want a bowl of creamy soup with it."

    assert ConversationEngine._message_adds_material_meal_update(text)


def test_confirmation_satisfied_after_confirmed_user_reply():
    prior_state = CoachingState(phase="confirmation")
    interaction_state = """
Latest user position:
- Yes, that's accurate. I'm happy to wrap it there.
Active issue:
- Confirm current commitment before closing.
"""

    assert ConversationEngine._confirmation_satisfied(
        prior_state=prior_state,
        interaction_state=interaction_state,
        user_intent="accepting",
        actionability="settled",
    )


def test_milk_allergy_allows_explicit_non_dairy_substitute_context():
    request = CoachingTurnRequest(
        current_message="",
        profile=UserProfileContext(allergies=["milk"]),
    )

    conflict = ConversationEngine._detect_profile_constraint_conflict(
        request=request,
        current_message=(
            "I can make the smoothie dairy-free with soy milk, banana, oats, "
            "and peanut butter."
        ),
        meal_base="",
    )

    assert conflict is None


def test_milk_allergy_still_flags_actual_dairy_foods():
    request = CoachingTurnRequest(
        current_message="",
        profile=UserProfileContext(allergies=["milk"]),
    )

    conflict = ConversationEngine._detect_profile_constraint_conflict(
        request=request,
        current_message="I want Greek yogurt with milk for breakfast.",
        meal_base="",
    )

    assert conflict is not None
    assert conflict["constraint"] == "milk"
