"""Phase prediction module for the portable coaching engine."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Mapping, Sequence

from ..json_output import JSONOutputError, load_json_object
from ..prompts.definitions.phases import PHASE_DEFINITIONS, PHASE_INSTRUCTIONS
from ..prompts.roles.phase_predictor import (
    PHASE_PREDICTOR_INPUT_TEMPLATE,
    PHASE_PREDICTOR_OUTPUT_SCHEMA,
    PHASE_PREDICTOR_ROLE_PROMPT,
)
from .orchestrator import PHASES

if TYPE_CHECKING:
    from ..agent_config import AgentConfig
    from ..memory.conversation_memory import SharedConversationHistory


_VALID_PHASES = frozenset(PHASES)


class PhasePredictor:
    """Predict a phase candidate before the orchestrator accepts or overrides it."""

    def __init__(self, nutrition_goal: str, config: "AgentConfig") -> None:
        self.nutrition_goal = nutrition_goal
        self.config = config
        self._system_prompt = "\n\n".join(
            block.strip()
            for block in [
                PHASE_PREDICTOR_ROLE_PROMPT.format(nutrition_goal=nutrition_goal),
                PHASE_DEFINITIONS,
                PHASE_INSTRUCTIONS,
                self._build_state_evidence_instruction(config),
                PHASE_PREDICTOR_OUTPUT_SCHEMA,
            ]
            if block and block.strip()
        )

    def get_messages(
        self,
        *,
        history: "SharedConversationHistory",
        turn_idx: int,
        current_phase: str,
        recommendation_history: Sequence[Mapping[str, Any]] = (),
        last_alignment_score: float | None = None,
        last_alignment_reasoning: str | None = None,
        last_certainty_score: float | None = None,
        last_certainty_reasoning: str | None = None,
        user_preferences: str = "",
    ) -> list[dict[str, str]]:
        alignment_state = self._format_state(
            score=last_alignment_score,
            reasoning=last_alignment_reasoning,
        )
        uncertainty_state = self._format_state(
            score=last_certainty_score,
            reasoning=last_certainty_reasoning,
        )
        user = PHASE_PREDICTOR_INPUT_TEMPLATE.format(
            turn_idx=turn_idx,
            max_turns=self.config.max_turns,
            current_phase=(current_phase or "exploration"),
            meal_base=history.meal_base or "(empty)",
            context_base=history.context_base or "(empty)",
            user_preferences=user_preferences or "(none)",
            recommendation_history=self._format_recommendation_history(
                recommendation_history
            ),
            alignment_state=alignment_state,
            uncertainty_state=uncertainty_state,
            recent_turns=history.to_recent_turns_text(),
        )
        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user},
        ]

    def parse_output(self, raw_output: str, fallback_phase: str) -> Dict[str, Any]:
        text = (raw_output or "").strip()
        phase = fallback_phase if fallback_phase in _VALID_PHASES else "exploration"
        if not text:
            return {
                "predicted_phase": phase,
                "confidence": 0.0,
                "reasoning": "(fallback: empty phase predictor response)",
            }

        try:
            data = load_json_object(text)
        except (JSONOutputError, ValueError, TypeError, AttributeError):
            return {
                "predicted_phase": phase,
                "confidence": 0.0,
                "reasoning": "(fallback: invalid phase predictor JSON)",
            }

        parsed_phase = str(
            data.get("predicted_phase", data.get("phase", ""))
        ).strip().lower()
        if parsed_phase not in _VALID_PHASES:
            parsed_phase = phase
        try:
            confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        return {
            "predicted_phase": parsed_phase,
            "confidence": confidence,
            "reasoning": str(data.get("reasoning", "")),
        }

    @staticmethod
    def _build_state_evidence_instruction(config: "AgentConfig") -> str:
        use_scores = bool(config.phase_predictor_use_state_scores)
        use_rationales = bool(config.phase_predictor_use_state_rationales)

        if use_scores and use_rationales:
            evidence = "alignment/uncertainty scores and rationales"
        elif use_scores:
            evidence = "alignment/uncertainty scores"
        elif use_rationales:
            evidence = "alignment/uncertainty rationales"
        else:
            return (
                "Use the meal state, context state, user intent, and recent "
                "conversation together. No dialogue state scores or rationales "
                "are provided."
            )

        return (
            f"Use the meal state, context state, {evidence}, user intent, "
            "and recent conversation together."
        )

    @staticmethod
    def _format_state(
        *,
        score: float | None,
        reasoning: str | None,
    ) -> str:
        if score is None and not reasoning:
            return "(not yet available)"
        return f"score={score if score is not None else 'unknown'}; reasoning={reasoning or '(none)'}"

    @staticmethod
    def _format_recommendation_history(
        recommendation_history: Sequence[Mapping[str, Any]],
    ) -> str:
        if not recommendation_history:
            return "(none)"
        lines = []
        for idx, rec in enumerate(recommendation_history[-3:], start=1):
            lines.append(f"{idx}. {dict(rec)}")
        return "\n".join(lines)
