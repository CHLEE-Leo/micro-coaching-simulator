"""Shared user intent definitions and intent-use instructions."""

INTENT_DEFINITIONS = """\
USER INTENT CLASSIFICATION (7 categories):
Classify user_intent based on the user's latest message:
- "informing" - User is answering questions or sharing meal details \
(food items, ingredients, preparation, portions, or dietary constraints).
- "accepting" - User agrees with, shows openness to, or follows a recommendation.
- "inquiring" - User asks a question, requests clarification, or proposes alternatives.
- "deferring" - User has NOT decided what to eat and is delegating the decision to the coach. \
Signals include: "I don't know what to eat", "no idea", "what do you recommend?", \
"I can't decide", "help me pick", "you choose". The user is ENGAGED but needs guidance - \
this is fundamentally different from "passive" (indifference) or "disengaging" (withdrawal). \
A user who says "I don't know" when asked about their meal is deferring, not passive.
- "passive" - Greeting, brief acknowledgement, or genuinely unclear/indifferent intent \
(e.g., "ok", "sure whatever", "hmm"). The user is not actively contributing information \
or requesting help.
- "rejecting" - User explicitly declines a specific recommendation or refuses to change their meal.
- "disengaging" - User expresses unwillingness to continue the coaching conversation itself \
(e.g., "I don't want coaching", "Stop asking me", "I'm done with this conversation", \
"Leave me alone"). This is different from rejecting a specific recommendation - \
the user is refusing the entire interaction, not just one suggestion.\
"""

INTENT_INSTRUCTIONS = """\
INTENT INSTRUCTIONS:

DEFERRING USER - NO MEAL PLANNED:
When user_intent is "deferring" (user has not decided what to eat and is \
asking the coach for help), the coach must guide the decision process - \
not just list food names.
- The user is ENGAGED and wants help. Do NOT treat this as stalling or \
non-responsiveness. The conversation should continue productively.
- First turn of deferring: Use INQUIRE to ask ONE concrete \
situational question that narrows options - cooking time/effort, what \
ingredients are available, eating out or at home, cravings or mood, \
or what they ate recently. Do NOT ask "what do you want to eat?" again.
- Second turn of deferring (still no meal info): Use INQUIRE to ask about \
constraints or preferences \
(allergies are already in the profile - ask about something else like \
cuisine type, budget, or how hungry they are).
- Third+ turn of deferring: Only then use RECOMMEND to propose a complete, \
simple, goal-aligned meal. The instruction MUST tell \
the sub-agent to propose a full meal from scratch (not an "improvement" \
to an existing meal) and explain WHY it fits the user's known constraints.

IMPORTANT - Allergy / medical constraint declarations:
If the user mentions an allergy, intolerance, medical restriction, or health concern \
(e.g., "I'm allergic to shellfish", "I can't eat gluten", "dairy upsets my stomach"), \
classify as "informing" - NOT "rejecting". \
The user is sharing a dietary constraint, not refusing coaching. \
The system will use this constraint to tailor future recommendations.

RETENTION AWARENESS:
The system tracks consecutive rejections. When you classify user_intent as "rejecting" \
repeatedly, the system may decide to respect the user's choice and wrap up gracefully - \
even if alignment is low. Your job is to classify intent accurately; the system handles \
escalation policy.\
"""
