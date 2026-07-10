"""API-level structured-output schemas for agent modules.

Prompts describe semantics; these schemas enforce syntax at the LLM API
boundary for modules that feed downstream parsers and state machines.
"""

from __future__ import annotations


def _string_array(description: str) -> dict:
    return {
        "type": "array",
        "description": description,
        "items": {"type": "string"},
    }


INTERACTION_STATE_SCHEMA = {
    "name": "interaction_state",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "answered_facts": _string_array("Facts the user has already answered."),
            "open_questions": _string_array("Still-useful unresolved questions."),
            "rejected_options": _string_array("Foods, suggestions, or directions rejected by the user."),
            "unavailable_options": _string_array("Options the user said are unavailable or infeasible."),
            "safety_conflicted_options": _string_array("Options that conflict with safety-relevant profile facts."),
            "user_requested_conflicted_options": _string_array("Conflicted options the user still explicitly requests."),
            "candidate_options": _string_array("Possible options not yet committed to."),
            "accepted_options": _string_array("Suggestions or meal components accepted by the user."),
            "meal_slots": _string_array("Slot-scoped meal facts, e.g., main dish accepted; side unresolved."),
            "active_issue": {
                "type": "string",
                "description": "The current unresolved user-facing task.",
            },
            "latest_user_position": {
                "type": "string",
                "description": "One concise sentence about the latest user stance.",
            },
        },
        "required": [
            "answered_facts",
            "open_questions",
            "rejected_options",
            "unavailable_options",
            "safety_conflicted_options",
            "user_requested_conflicted_options",
            "candidate_options",
            "accepted_options",
            "meal_slots",
            "active_issue",
            "latest_user_position",
        ],
    },
}


DIALOGUE_PLAN_SCHEMA = {
    "name": "dialogue_plan",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "intent_summary": {"type": "string"},
            "user_intent": {
                "type": "string",
                "enum": [
                    "informing",
                    "accepting",
                    "inquiring",
                    "deferring",
                    "passive",
                    "rejecting",
                    "disengaging",
                ],
            },
            "phase": {
                "type": "string",
                "enum": [
                    "exploration",
                    "assessment",
                    "recommendation",
                    "negotiation",
                    "confirmation",
                    "finalization",
                ],
            },
            "actionability": {
                "type": "string",
                "enum": ["insufficient", "workable", "settled", "boundary", "conflicted"],
            },
            "action": {
                "type": "string",
                "enum": [
                    "inquire",
                    "assess",
                    "recommend",
                    "respond",
                    "confirm",
                    "handoff",
                    "close",
                    "terminate",
                ],
            },
            "closure_readiness": {
                "type": "string",
                "enum": ["not_ready", "actionable", "ready_to_close", "boundary_close"],
            },
            "reasoning": {"type": "string"},
            "instruction": {"type": "string"},
            "assessment_followup_action": {
                "type": "string",
                "enum": ["", "inquire", "recommend", "confirm", "handoff", "close", "terminate"],
            },
            "assessment_followup_phase": {
                "type": "string",
                "enum": [
                    "",
                    "exploration",
                    "recommendation",
                    "negotiation",
                    "confirmation",
                    "finalization",
                ],
            },
            "assessment_followup_instruction": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": [
            "intent_summary",
            "user_intent",
            "phase",
            "actionability",
            "action",
            "closure_readiness",
            "reasoning",
            "instruction",
            "assessment_followup_action",
            "assessment_followup_phase",
            "assessment_followup_instruction",
            "confidence",
        ],
    },
}


ASSESSMENT_SCHEMA = {
    "name": "meal_assessment",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "summary": {"type": "string"},
            "strengths": _string_array("Goal-aligned or user-aligned strengths."),
            "limitations": _string_array("Current limitations or risks."),
            "overall": {
                "type": "string",
                "enum": ["aligned", "partially_aligned", "not_aligned"],
            },
        },
        "required": ["summary", "strengths", "limitations", "overall"],
    },
}


RECOMMENDATION_SCHEMA = {
    "name": "meal_recommendation",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "recommendation_type": {
                "type": "string",
                "enum": [
                    "add",
                    "remove",
                    "modify",
                    "substitute",
                    "swap",
                    "portion",
                    "preparation",
                    "confirm",
                    "cautious_continuation",
                ],
            },
            "target_food": {"type": "string"},
            "suggestion": {"type": "string"},
            "reasoning": {"type": "string"},
            "expected_impact": {
                "type": "string",
                "enum": ["low", "medium", "high"],
            },
            "options": {
                "type": "array",
                "minItems": 1,
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "option_id": {"type": "string"},
                        "target_food": {"type": "string"},
                        "suggestion": {"type": "string"},
                        "reasoning": {"type": "string"},
                        "expected_impact": {
                            "type": "string",
                            "enum": ["low", "medium", "high"],
                        },
                    },
                    "required": [
                        "option_id",
                        "target_food",
                        "suggestion",
                        "reasoning",
                        "expected_impact",
                    ],
                },
            },
        },
        "required": [
            "recommendation_type",
            "target_food",
            "suggestion",
            "reasoning",
            "expected_impact",
            "options",
        ],
    },
}
