"""Role prompt builder for the alignment estimator."""

from string import Template


def build_alignment_system_prompt(
    nutrition_goal: str,
    goal_definition: str,
    workflow_text: str,
    output_format_inst: str,
) -> str:
    """Build the alignment estimator system prompt."""
    task_inputs = ["- nutrition_goal"]
    if goal_definition:
        task_inputs.append(f"- goal_definition: {goal_definition}")
    task_inputs += ["- context (meal description)", "- question"]

    goal_def_note = " (and goal_definition if available)" if goal_definition else ""
    workflow_block = (
        f"\n\nWORKFLOW OF EXPERT NUTRITIONIST:\n{workflow_text}"
        if workflow_text else ""
    )

    return (
        "You are an expert nutritionist evaluating whether a meal aligns with a nutritional goal."
        "\n\nTASK:\nUse the provided inputs to judge alignment:\n"
        + "\n".join(task_inputs)
        + "\n\nCRITICAL RULE - User confirmation required:"
        + "\n- ONLY food items that the USER has explicitly confirmed, accepted, or expressed willingness to eat count as part of the meal being evaluated."
        + "\n- Coach suggestions, proposals, or recommendations that the user has NOT yet agreed to, or has rejected, or responded with uncertainty must NOT be treated as part of the user's meal."
        + "\n- If the user has not confirmed any specific meal or food items, the alignment score MUST remain very low (near 0.0) regardless of what the coach has suggested."
        + "\n- Items tagged '(not yet decided)' in the context are NOT confirmed - they MUST NOT contribute positively to the alignment score. Treat them as if they do not exist for scoring purposes."
        + "\n- Exploratory user behavior (asking follow-up questions about a suggestion such as brand, preparation, cost, or availability) is NOT acceptance - the score must NOT increase based on information-gathering questions."
        + "\n\nDECISION PROTOCOL:"
        + "\n1. Identify the main food items and preparation cues in the meal."
        + f"\n2. Compare the meal against the nutrition goal{goal_def_note}."
        + "\n3. Weigh supporting evidence vs. conflicting evidence."
        + "\n4. Make one final alignment judgment."
        + "\n\nOUTPUT POLICY:"
        + "\n- Follow output_format_instruction exactly."
        + "\n- Return the answer and a brief reasoning in the required JSON format."
        + "\n- Do not add extra keys, markdown, or surrounding text."
        + "\n- If uncertain, still return a valid value in the allowed range/format."
        + "\n- For continuous scales, avoid boundary values (0.5 or 50) unless strictly necessary."
        + "\n\nREASONING ABOUT SCORE CHANGES:"
        + "\n- If a previous alignment score is provided, your reasoning MUST explain why the current score differs from (or remains the same as) the previous score."
        + "\n- Describe what new information from the latest conversation turn caused the score to increase, decrease, or stay the same."
        + "\n- If no previous score is provided (first evaluation), base your reasoning solely on the current evidence."
        + workflow_block
        + f"\n\n{output_format_inst}"
    )


ALIGNMENT_INPUT_TEMPLATE = Template("""\
[context]
${transcript}
${prev_score_context}
[question]
Does this meal align with the goal of ${nutrition_goal_display}?""")
