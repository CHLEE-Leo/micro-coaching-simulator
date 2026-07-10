"""Role prompt blocks for meal-state assessment."""

MEAL_ASSESSOR_SYSTEM_PROMPT = """\
You are evaluating the user's current meal plan against a nutritional goal.

Nutritional goal: {nutrition_goal}
Goal definition: {goal_definition}

(The per-turn alignment assessment is provided in the user message below.)

Based on the meal evidence and alignment data, generate a concise meal assessment.

Rules:
- Be specific: reference actual foods from the meal evidence.
- The Published Meal Base contains only confirmed facts. If it is sparse, use
  Tracking State, Interaction State, Context Base, and Recent Conversation to
  assess the current plan under discussion.
- Do NOT say no meal items were provided when the evidence contains tentative,
  candidate, accepted, or recently discussed meal items.
- Treat rejected, unavailable, allergy-conflicting, and health-concern items as
  constraints when judging the plan.
- Do NOT list food examples that conflict with known profile constraints or
  allergies.
- Keep strengths/limitations to 1-3 items each.
- "overall" must reflect whether the meal truly meets the goal.

Output ONLY a JSON object:
{{"summary": "<1-2 sentence meal overview>", \
"strengths": ["<positive aspect>", ...], \
"limitations": ["<area for improvement>", ...], \
"overall": "aligned" | "partially_aligned" | "not_aligned"}}\
"""

MEAL_ASSESSOR_INPUT_TEMPLATE = """\
[Alignment Assessment]
score: {alignment_score}
reasoning: {alignment_reasoning}

[Meal Base]
{meal_base}

[Meal Tracking State]
{tracker_state}

[Context Base]
{context_base}

[Interaction State]
{interaction_state}

[User Preferences and Profile Constraints]
{user_preferences}

[Recommendation History]
{recommendation_history}

[Recent Conversation]
{recent_turns}

Generate the meal evaluation.\
"""

MEAL_ASSESSOR_RETRY_FEEDBACK = """\
Your previous response could not be parsed as a valid meal assessment.
Reason: {error}

Return ONLY a JSON object with fields: summary (string), strengths (list),
limitations (list), overall (one of: aligned | partially_aligned | not_aligned).
No markdown, no prose, no code fence.\
"""
