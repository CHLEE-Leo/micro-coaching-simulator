"""User-facing response generation for the portable coaching engine."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Dict, List, Optional

from ..prompts.roles.response_generator import (
    RESPONSE_ANSWER_INPUT_TEMPLATE,
    RESPONSE_ANSWER_SYSTEM_PROMPT,
    RESPONSE_ASSESSMENT_INPUT_TEMPLATE,
    RESPONSE_ASSESSMENT_SYSTEM_PROMPT,
    RESPONSE_CONFIRMATION_INPUT_TEMPLATE,
    RESPONSE_CONFIRMATION_SYSTEM_PROMPT,
    RESPONSE_HANDOFF_INPUT_TEMPLATE,
    RESPONSE_HANDOFF_SYSTEM_PROMPT,
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
        self._confirmation_system = RESPONSE_CONFIRMATION_SYSTEM_PROMPT.format(
            nutrition_goal=goal_text,
        )
        self._handoff_system = RESPONSE_HANDOFF_SYSTEM_PROMPT.format(
            nutrition_goal=goal_text,
        )

    def get_assessment_messages(
        self,
        assessment: Dict,
        needs_recommendation: bool,
        history: "SharedConversationHistory",
        recommendation_history: Optional[List[Dict]] = None,
    ) -> List[Dict[str, str]]:
        payload = {
            **self._public_assessment_payload(assessment),
            "needs_recommendation": needs_recommendation,
        }
        user = RESPONSE_ASSESSMENT_INPUT_TEMPLATE.format(
            assessment_json=json.dumps(payload, ensure_ascii=False, indent=2),
            recent_turns=history.to_recent_turns_text(),
            previous_recommendations_context=self._format_previous_recommendations(
                recommendation_history
            ),
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
        current_assessment: Optional[Dict] = None,
    ) -> List[Dict[str, str]]:
        user = RESPONSE_RECOMMENDATION_INPUT_TEMPLATE.format(
            recommendation_json=json.dumps(rec_template, ensure_ascii=False, indent=2),
            current_assessment_context=self._format_current_assessment(
                current_assessment
            ),
            recent_turns=history.to_recent_turns_text(),
            previous_recommendations_context=self._format_previous_recommendations(
                recommendation_history
            ),
        )
        return [
            {"role": "system", "content": self._recommendation_system},
            {"role": "user", "content": user},
        ]

    @staticmethod
    def _public_assessment_payload(assessment: Dict) -> Dict:
        """Strip internal telemetry before prompting user-facing text."""
        if assessment.get("_degraded"):
            return {
                "summary": "",
                "strengths": [],
                "limitations": [],
                "overall": "partially_aligned",
            }
        return {
            "summary": str(assessment.get("summary") or ""),
            "strengths": assessment.get("strengths") or [],
            "limitations": assessment.get("limitations") or [],
            "overall": str(assessment.get("overall") or "partially_aligned"),
            **(
                {"override_note": assessment["override_note"]}
                if assessment.get("override_note")
                else {}
            ),
        }

    @classmethod
    def _format_current_assessment(cls, assessment: Optional[Dict]) -> str:
        if not assessment:
            return ""
        if assessment.get("_degraded"):
            return (
                "[Current Assessment]\n"
                "Use only confirmed meal, context, and interaction-state "
                "evidence. Keep the recommendation conservative and do not "
                "cite unavailable assessment details.\n"
            )
        return (
            "[Current Assessment]\n"
            + json.dumps(cls._public_assessment_payload(assessment), ensure_ascii=False, indent=2)
            + "\nUse this as the immediate rationale for the recommendation.\n"
        )

    def get_motivational_messages(
        self,
        assessment: Dict,
        history: "SharedConversationHistory",
        exit_context: str = "",
        finalization_style: str = "motivational",
        recommendation_history: Optional[List[Dict]] = None,
    ) -> List[Dict[str, str]]:
        system = RESPONSE_MOTIVATIONAL_SYSTEM_PROMPT.format(
            nutrition_goal=self.nutrition_goal.replace("_", " "),
            finalization_style=finalization_style,
            exit_context=exit_context,
        )
        user = RESPONSE_MOTIVATIONAL_INPUT_TEMPLATE.format(
            assessment_json=json.dumps(
                self._public_assessment_payload(assessment),
                ensure_ascii=False,
                indent=2,
            ),
            recent_turns=history.to_recent_turns_text(),
            previous_recommendations_context=self._format_previous_recommendations(
                recommendation_history
            ),
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def get_confirmation_messages(
        self,
        history: "SharedConversationHistory",
    ) -> List[Dict[str, str]]:
        user = RESPONSE_CONFIRMATION_INPUT_TEMPLATE.format(
            meal_base=history.meal_base or "(not yet available)",
            interaction_state=history.interaction_state or "(not yet available)",
            context_base=history.context_base or "(not yet available)",
            recent_turns=history.to_recent_turns_text(),
        )
        return [
            {"role": "system", "content": self._confirmation_system},
            {"role": "user", "content": user},
        ]

    def get_handoff_messages(
        self,
        instruction: str,
        history: "SharedConversationHistory",
    ) -> List[Dict[str, str]]:
        user = RESPONSE_HANDOFF_INPUT_TEMPLATE.format(
            instruction=instruction,
            meal_base=history.meal_base or "(not yet available)",
            interaction_state=history.interaction_state or "(not yet available)",
            context_base=history.context_base or "(not yet available)",
            recent_turns=history.to_recent_turns_text(),
        )
        return [
            {"role": "system", "content": self._handoff_system},
            {"role": "user", "content": user},
        ]

    @staticmethod
    def _format_previous_recommendations(
        recommendation_history: Optional[List[Dict]],
    ) -> str:
        """Format compact advice memory for response realization."""
        if not recommendation_history:
            return ""
        rec_lines = []
        for r in recommendation_history:
            options = r.get("options")
            if isinstance(options, list) and options:
                option_text = "; ".join(
                    str(option.get("suggestion", ""))
                    for option in options
                    if isinstance(option, dict) and option.get("suggestion")
                )
                rec_lines.append(
                    f"- Turn {r.get('turn_idx', '?')}: "
                    f"parallel adjustments -> {option_text}"
                )
                continue
            rec_lines.append(
                f"- Turn {r.get('turn_idx', '?')}: "
                f"{r.get('suggestion', '?')} (target: {r.get('target_food', '?')})"
            )
        return (
            "\n[Previous Recommendations Already Given]\n"
            + "\n".join(rec_lines)
            + "\nDo not contradict, reverse, repeat, or lightly rephrase these. "
            "Treat bundle items as compatible adjustments. If the user accepted "
            "one, treat it as part of the current plan while unresolved bundle "
            "items remain available for follow-up.\n"
        )

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
        text = (raw_output or "").strip().strip('"').strip("'").strip()
        return self._repair_perspective_and_register(text)

    @staticmethod
    def _repair_perspective_and_register(text: str) -> str:
        """Repair common user-meal ownership and register slips.

        LLM realization can occasionally summarize the user's meal as if the
        coach owned it ("For dinner, I have ..."). This small deterministic
        guard preserves the generated content while correcting speaker role.
        """
        if not text:
            return text
        replacements = (
            (r"\bFor dinner,\s+I have\b", "For dinner, your plan has"),
            (r"\bFor dinner,\s+I'm having\b", "For dinner, you're planning"),
            (r"\bmy dinner plan\b", "your dinner plan"),
            (r"\bmy meal plan\b", "your meal plan"),
            (r"\bI get wanting to\b", "It makes sense that you want to"),
            (r"\bI['’]d keep\b", "Keep"),
            (r"\bI['’]d use\b", "Use"),
            (r"\bI['’]d go with\b", "Use"),
        )
        repaired = text
        for pattern, repl in replacements:
            repaired = re.sub(pattern, repl, repaired, flags=re.IGNORECASE)
        repaired = ResponseGenerator._remove_decision_menu_tail(repaired)
        return repaired

    @staticmethod
    def _remove_decision_menu_tail(text: str) -> str:
        """Remove high-burden choice-menu endings from recommendation bubbles."""
        replacements = (
            (
                r"\s*(?:Which|What)\s+(?:of\s+these\s+)?(?:adjustments?|ones?|options?)"
                r".{0,120}\?$",
                "",
            ),
            (
                r"\s*(?:What\s+would\s+you\s+like\s+to\s+do\s+next|"
                r"How\s+would\s+you\s+like\s+to\s+proceed|"
                r"How\s+would\s+you\s+like\s+to\s+handle\s+it)\?"
                r"(?:\s*\n\s*-\s*.{0,160}){1,5}\s*$",
                "",
            ),
            (
                r"\s*You\s+can\s+(?:use|choose|pick)\s+any\s+combination"
                r".{0,120}\.?$",
                "",
            ),
            (r"\s*(?:Tell me|Let me know)\s+which\s+.{0,120}\.?$", ""),
            (
                r"\s*Does\s+that\s+(?:all\s+)?(?:still\s+)?(?:look|sound)\s+right"
                r".{0,140}(?:add|change).{0,80}\?$",
                " If that is accurate, I can wrap it there.",
            ),
            (
                r"\s*Does\s+that\s+(?:feel|seem|look|sound)\s+like\s+your\s+final"
                r".{0,140}(?:change|add|anything).{0,80}\?$",
                " If that is accurate, I can wrap it there.",
            ),
            (
                r"\s*Is\s+there\s+anything\s+you\s+want\s+to\s+(?:add|change)"
                r".{0,80}\?$",
                " If that is accurate, I can wrap it there.",
            ),
            (
                r"\s*Is\s+that\s+everything\s+for\s+this\s+meal"
                r".{0,140}(?:change|add|detail).{0,80}\?$",
                " If that is accurate, I can wrap it there.",
            ),
            (
                r"\s*[—–-]?\s*(?:Want\s+to\s+go\s+with\s+that|Does\s+that\s+work|"
                r"Sound\s+good)\?$",
                "",
            ),
        )
        repaired = text
        for pattern, replacement in replacements:
            repaired = re.sub(
                pattern,
                replacement,
                repaired,
                flags=re.IGNORECASE | re.DOTALL,
            )
        return repaired.strip()

    def fallback_question_text(self, question_template: Dict) -> str:
        return str(
            question_template.get("question_template")
            or "Could you tell me a bit more?"
        )

    def fallback_recommendation_text(self, rec_template: Dict) -> str:
        suggestion = str(rec_template.get("suggestion") or "").strip()
        return suggestion or "I have one small suggestion, but I need a bit more detail first."

    def fallback_assessment_text(self, assessment: Dict) -> str:
        if assessment.get("_degraded"):
            return (
                "I have enough context to continue, so I’ll keep the next step "
                "conservative and grounded in what you’ve already shared."
            )
        summary = str(assessment.get("summary") or "").strip()
        return summary or "Thanks, I have enough to review your meal."

    def fallback_closing_text(self, instruction: str = "") -> str:
        return instruction.strip() or "Thanks for sharing about your meal."

    def fallback_motivational_ending_text(self, assessment: Dict) -> str:
        if assessment.get("_degraded"):
            return "Thanks for sharing your meal plan. I’ll leave it there."
        summary = str(assessment.get("summary") or "").strip()
        return summary or "Thanks for sharing about your meal."

    def fallback_confirmation_text(self, history: "SharedConversationHistory") -> str:
        meal = (history.meal_base or "your current meal plan").strip()
        return (
            f"Here is your current plan as I understand it: {meal}. "
            "If that is accurate, I can wrap it there."
        )

    def fallback_handoff_text(self) -> str:
        return (
            "Let’s keep your current plan for now and avoid adding more decisions.\n"
            "- I’ll preserve the workable parts you already chose.\n"
            "- I’ll note the main tradeoff without pushing another change."
        )
