"""User-facing response generation for the portable coaching engine."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Dict, List, Optional

from ..prompts.roles.response_generator import (
    RESPONSE_ANSWER_INPUT_TEMPLATE,
    RESPONSE_ANSWER_SYSTEM_PROMPT,
    RESPONSE_ASSESSMENT_INPUT_TEMPLATE,
    RESPONSE_ASSESSMENT_SYSTEM_PROMPT,
    RESPONSE_MOTIVATIONAL_INPUT_TEMPLATE,
    RESPONSE_MOTIVATIONAL_SYSTEM_PROMPT,
    RESPONSE_QUESTION_INPUT_TEMPLATE,
    RESPONSE_QUESTION_SYSTEM_PROMPT,
    RESPONSE_RECOMMENDATION_INPUT_TEMPLATE,
    RESPONSE_RECOMMENDATION_SYSTEM_PROMPT,
)

if TYPE_CHECKING:
    from ..agent_config import AgentConfig
    from ..memory.conversation_memory import SharedConversationHistory


class ResponseGenerator:
    """Convert structured internal outputs into user-facing chat bubbles."""

    def __init__(self, nutrition_goal: str, config: "AgentConfig"):
        self.nutrition_goal = nutrition_goal
        self.config = config
        goal_text = nutrition_goal.replace("_", " ")
        self._question_system = RESPONSE_QUESTION_SYSTEM_PROMPT.format(
            nutrition_goal=goal_text,
        )
        self._recommendation_system = RESPONSE_RECOMMENDATION_SYSTEM_PROMPT.format(
            nutrition_goal=goal_text,
        )
        self._assessment_system = RESPONSE_ASSESSMENT_SYSTEM_PROMPT.format(
            nutrition_goal=goal_text,
        )
        self._answer_system = RESPONSE_ANSWER_SYSTEM_PROMPT.format(
            nutrition_goal=goal_text,
        )

    def get_assessment_messages(
        self,
        assessment: Dict,
        needs_recommendation: bool,
        history: "SharedConversationHistory",
    ) -> List[Dict[str, str]]:
        payload = {**assessment, "needs_recommendation": needs_recommendation}
        user = RESPONSE_ASSESSMENT_INPUT_TEMPLATE.format(
            assessment_json=json.dumps(payload, ensure_ascii=False, indent=2),
            recent_turns=history.to_recent_turns_text(),
        )
        return [
            {"role": "system", "content": self._assessment_system},
            {"role": "user", "content": user},
        ]

    def get_question_messages(
        self,
        question_template: Dict,
        history: "SharedConversationHistory",
    ) -> List[Dict[str, str]]:
        user = RESPONSE_QUESTION_INPUT_TEMPLATE.format(
            question_json=json.dumps(question_template, ensure_ascii=False, indent=2),
            recent_turns=history.to_recent_turns_text(),
        )
        return [
            {"role": "system", "content": self._question_system},
            {"role": "user", "content": user},
        ]

    def get_recommendation_messages(
        self,
        rec_template: Dict,
        history: "SharedConversationHistory",
        recommendation_history: Optional[List[Dict]] = None,
    ) -> List[Dict[str, str]]:
        prev_recs_text = ""
        if recommendation_history:
            rec_lines = [
                f"- Turn {r.get('turn_idx', '?')}: "
                f"{r.get('suggestion', '?')} (target: {r.get('target_food', '?')})"
                for r in recommendation_history
            ]
            prev_recs_text = (
                "\n[Previous Recommendations Already Given]\n"
                + "\n".join(rec_lines)
                + "\nDo NOT contradict or reverse any previously accepted suggestion. "
                "Build upon the user's choices.\n"
            )

        user = RESPONSE_RECOMMENDATION_INPUT_TEMPLATE.format(
            recommendation_json=json.dumps(rec_template, ensure_ascii=False, indent=2),
            recent_turns=history.to_recent_turns_text(),
            previous_recommendations_context=prev_recs_text,
        )
        return [
            {"role": "system", "content": self._recommendation_system},
            {"role": "user", "content": user},
        ]

    def get_motivational_messages(
        self,
        assessment: Dict,
        history: "SharedConversationHistory",
        exit_context: str = "",
    ) -> List[Dict[str, str]]:
        system = RESPONSE_MOTIVATIONAL_SYSTEM_PROMPT.format(
            nutrition_goal=self.nutrition_goal.replace("_", " "),
            exit_context=exit_context,
        )
        user = RESPONSE_MOTIVATIONAL_INPUT_TEMPLATE.format(
            assessment_json=json.dumps(assessment, ensure_ascii=False, indent=2),
            recent_turns=history.to_recent_turns_text(),
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def get_answer_messages(
        self,
        instruction: str,
        history: "SharedConversationHistory",
    ) -> List[Dict[str, str]]:
        user = RESPONSE_ANSWER_INPUT_TEMPLATE.format(
            instruction=instruction,
            meal_base=history.meal_base or "(not yet available)",
            recent_turns=history.to_recent_turns_text(),
        )
        return [
            {"role": "system", "content": self._answer_system},
            {"role": "user", "content": user},
        ]

    def clean_response_text(self, raw_output: str) -> str:
        return (raw_output or "").strip().strip('"').strip("'").strip()

    def fallback_question_text(self, question_template: Dict) -> str:
        return str(
            question_template.get("question_template")
            or "Could you tell me a bit more?"
        )

    def fallback_recommendation_text(self, rec_template: Dict) -> str:
        suggestion = str(rec_template.get("suggestion") or "").strip()
        return suggestion or "I have one small suggestion, but I need a bit more detail first."

    def fallback_assessment_text(self, assessment: Dict) -> str:
        summary = str(assessment.get("summary") or "").strip()
        return summary or "Thanks, I have enough to review your meal."

    def fallback_closing_text(self, instruction: str = "") -> str:
        return instruction.strip() or "Thanks for sharing about your meal."

    def fallback_motivational_ending_text(self, assessment: Dict) -> str:
        summary = str(assessment.get("summary") or "").strip()
        return summary or "Thanks for sharing about your meal."
