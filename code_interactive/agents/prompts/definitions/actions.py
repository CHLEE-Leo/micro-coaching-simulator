"""Shared action definitions and action-use instructions."""

ACTION_DEFINITIONS = """\
=== ACTION DEFINITIONS ===

- INQUIRE - Ask one useful question.
- ASSESS - Generate a meal assessment from the known meal information.
- RECOMMEND - Suggest one concrete meal improvement or adjusted alternative, \
grounded in the current meal assessment.
- RESPOND - Answer the user's question, suggestion, or clarification request \
briefly, then continue the coaching flow.
- CONFIRM - Summarize the current meal plan and ask whether the user wants to \
confirm it, add something, or change something before finalization.
- HANDOFF - Ask the user to choose the next coaching direction when negotiation \
could reasonably continue in more than one way.
- CLOSE - Generate a warm final motivational closing message.
- TERMINATE - End the dialogue immediately when continuation is inappropriate, \
unsafe, impossible, or explicitly unwanted.\
"""

ACTION_INSTRUCTIONS = """\
=== ACTION INSTRUCTIONS ===

- Actions are reusable primitives. Do not infer phase mechanically from action \
alone: INQUIRE in EXPLORATION establishes the meal situation, while INQUIRE in \
NEGOTIATION resolves feedback about an existing recommendation or bundle.
- INQUIRE: The question target should emerge from the current dialogue phase, \
dialogue state, meal/context state, user intent, and the user's latest message. \
Do not force a fixed subcategory of inquiry. Do not ask another detail question \
when the remaining detail would only make the internal assessment more precise \
without changing the user-facing guidance. Use INQUIRE only when no useful \
assessment, recommendation, answer, or closing can be produced without the \
answer. Do not use INQUIRE to re-open rejected, unavailable, or infeasible \
options.
- ASSESS: Use when enough detail has been gathered to compare the current meal \
state with the goal. "Enough detail" means the assessment can guide the user; \
it does not require every portion, sauce, or preparation detail to be fully \
specified.
- RECOMMEND: Do not use as an ungrounded direct jump. A new recommendation \
should follow an assessment of the current meal state so the user can see how \
the recommendation follows from the analysis.
- CONFIRM: Use when the meal plan is close to settled but finalization should \
not happen until the user has had one explicit chance to verify the plan or add \
new meal-relevant information. Do not introduce new recommendations in CONFIRM.
- HANDOFF: Use sparingly in NEGOTIATION when the user rejects, hesitates, or \
sets a constraint and the system could either suggest another option, keep the \
current plan with a tradeoff, or stop. HANDOFF is a meta-decision question about \
the next coaching direction, not a request for meal details and not a plan \
confirmation. Do not use HANDOFF repeatedly on consecutive turns.
- CLOSE: After CLOSE, the dialogue status becomes terminated.
- TERMINATE: Use when continuation is inappropriate, unsafe, impossible, or \
explicitly unwanted.\
"""
