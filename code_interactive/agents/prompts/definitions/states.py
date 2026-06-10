"""Shared dialogue state definitions and state-use instructions."""

STATE_DEFINITIONS = """\
=== STATE DEFINITIONS ===

- Dialogue state scores are numeric evidence about the current conversation. \
They include alignment score and uncertainty score when provided.
- Dialogue state rationales are verbal evidence explaining why those scores \
were produced. They include alignment reasoning and uncertainty reasoning when \
provided.\
"""


def build_state_merge(
    *,
    include_scores: bool,
    include_rationales: bool,
) -> str:
    """Return bridge guidance for whichever dialogue state blocks are enabled."""
    if include_scores and include_rationales:
        return (
            "Use the dialogue state scores and rationales together when choosing "
            "the next action. Scores provide coarse thresholds; rationales explain "
            "why those scores were produced."
        )
    if include_scores:
        return (
            "Use the dialogue state scores when choosing the next action. "
            "Apply the score thresholds as guidance."
        )
    if include_rationales:
        return (
            "Use the dialogue state rationales when choosing the next action. "
            "They explain what is known, missing, strong, or weak in the meal."
        )
    return ""


STATE_SCORE_INSTRUCTIONS = """\
=== STATE SCORE INSTRUCTIONS ===

- Uncertainty score HIGH (>= 0.85) -> Enough information has been gathered. \
In EXPLORATION, choose ASSESS. Do not keep asking questions.
- Uncertainty score LOW (< 0.5) -> Critical details are still missing. \
Continue INQUIRE unless the user is unable or unwilling to provide more.
- Alignment score LOW + user accepts recommendation -> The meal is still incomplete \
or far from the goal. Do NOT choose CLOSE. Instead, use INQUIRE to explore \
whether the user has additional food items for this meal.
- Alignment score HIGH (>= 0.8) + user accepts -> Safe to CLOSE.\
"""

STATE_RATIONALE_INSTRUCTIONS = """\
=== STATE RATIONALE INSTRUCTIONS ===

- Read the Alignment reasoning and Uncertainty reasoning to understand WHY \
scores are what they are.
- Use the rationales to decide what is missing, what is already strong, \
and what the most useful next action should be.
- Do not rely on hardcoded score thresholds alone when the rationales show \
important context about the user's meal or constraints.\
"""
