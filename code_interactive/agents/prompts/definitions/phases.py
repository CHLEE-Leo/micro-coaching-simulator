"""Shared dialogue phase definitions and phase-use instructions."""

PHASE_DEFINITIONS = """\
=== PHASE DEFINITIONS ===

- EXPLORATION - The coach is establishing the user's meal situation, \
including foods under consideration, preparation, portions, constraints, \
profile conflicts, available options, and relevant context. Use exploration \
for early safety or feasibility clarification before any system recommendation \
has become the object of discussion.

- ASSESSMENT - The coach is analyzing and evaluating the user's current meal \
state based on the information collected so far. This phase should precede any \
new user-facing recommendation.

- RECOMMENDATION - The coach is preparing or providing a concrete meal \
improvement recommendation grounded in the immediately preceding assessment.

- NEGOTIATION - The user is responding to, questioning, accepting, rejecting, \
or revising a previously proposed recommendation, option bundle, or assessed \
plan. Do not use negotiation merely because a safety or feasibility conflict \
exists in the first meal description.

- CONFIRMATION - The coach is verifying the current settled meal plan before \
finalization. This phase gives the user one clear opportunity to confirm, add, \
or change meal-relevant information.

- FINALIZATION - The dialogue is ready for a final wrap-up after the meal has \
been assessed, relevant recommendation choices have been resolved, and the user \
has confirmed the plan or set a clear boundary.\
"""

PHASE_INSTRUCTIONS = """\
=== PHASE INSTRUCTIONS ===

Phase is the dialogue-stage label, not a synonym for action. The same action \
can appear in more than one phase, but with a different function.

Use this phase-action guide:
- EXPLORATION typically uses INQUIRE or RESPOND to establish meal/context facts; \
it can move to ASSESS when enough evidence exists.
- ASSESSMENT uses ASSESS to make the current meal-state evaluation visible.
- RECOMMENDATION uses RECOMMEND only after an immediately preceding assessment.
- NEGOTIATION uses RESPOND, INQUIRE, ASSESS, or CONFIRM to handle user feedback \
about an existing recommendation or option bundle.
- CONFIRMATION uses CONFIRM or RESPOND to verify the settled plan before \
finalization, or ASSESS if new meal evidence reopens the plan.
- FINALIZATION uses CLOSE or TERMINATE.

Do not force a fixed sequence. Use the current meal state, context state, \
dialogue state, user intent, recent conversation, and recommendation history to \
decide which phase best describes the dialogue stage.\
"""
