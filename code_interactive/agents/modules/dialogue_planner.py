"""Dialogue planner for one compact turn-level plan."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, Dict, Mapping, Sequence

from ..json_output import JSONOutputError, load_json_object
from ..prompts.definitions.actions import ACTION_DEFINITIONS, ACTION_INSTRUCTIONS
from ..prompts.definitions.intents import INTENT_DEFINITIONS, INTENT_INSTRUCTIONS
from ..prompts.definitions.phases import PHASE_DEFINITIONS, PHASE_INSTRUCTIONS
from ..prompts.definitions.states import (
    STATE_DEFINITIONS,
    STATE_RATIONALE_INSTRUCTIONS,
    STATE_SCORE_INSTRUCTIONS,
    build_state_merge,
)
from ..prompts.roles.dialogue_planner import (
    DIALOGUE_PLANNER_INPUT_TEMPLATE,
    DIALOGUE_PLANNER_OUTPUT_SCHEMA,
    DIALOGUE_PLANNER_ROLE_PROMPT,
    DIALOGUE_PLANNER_RULES,
)

if TYPE_CHECKING:
    from ..agent_config import AgentConfig
    from ..memory.conversation_memory import SharedConversationHistory


PHASES = (
    "exploration",
    "assessment",
    "recommendation",
    "negotiation",
    "confirmation",
    "finalization",
)
VALID_ACTIONS = frozenset({
    "inquire",
    "assess",
    "recommend",
    "respond",
    "confirm",
    "handoff",
    "close",
    "terminate",
})
VALID_INTENTS = frozenset({
    "informing",
    "accepting",
    "inquiring",
    "deferring",
    "passive",
    "rejecting",
    "disengaging",
})
POST_ASSESS_ACTIONS = frozenset({
    "inquire",
    "recommend",
    "confirm",
    "handoff",
    "close",
    "terminate",
})
CLOSURE_READINESS = frozenset({
    "not_ready",
    "actionable",
    "ready_to_close",
    "boundary_close",
})
ACTIONABILITY = frozenset({
    "insufficient",
    "workable",
    "settled",
    "boundary",
    "conflicted",
})


class DialoguePlanner:
    """Produce one phase/action/follow-up plan for the current turn."""

    def __init__(self, nutrition_goal: str, config: "AgentConfig") -> None:
        self.nutrition_goal = nutrition_goal
        self.config = config

        from .meal_recommender import _load_goal_definitions

        goal_spec = _load_goal_definitions().get(nutrition_goal, {})
        goal_definition = goal_spec.get("definition", "")
        goal_text = nutrition_goal.replace("_", " ")
        self._system_prompt = self._build_system_prompt(
            nutrition_goal=goal_text,
            goal_definition=goal_definition,
            config=config,
        )

    def get_messages(
        self,
        *,
        history: "SharedConversationHistory",
        turn_idx: int,
        current_phase: str,
        recommendation_history: Sequence[Mapping[str, Any]] = (),
        consecutive_qa_count: int = 0,
        last_alignment_score: float | None = None,
        last_alignment_reasoning: str | None = None,
        last_certainty_score: float | None = None,
        last_certainty_reasoning: str | None = None,
        user_preferences: str = "",
        interaction_state: str = "",
    ) -> list[dict[str, str]]:
        user = DIALOGUE_PLANNER_INPUT_TEMPLATE.format(
            turn_idx=turn_idx,
            max_turns=self.config.max_turns,
            current_phase=current_phase or "exploration",
            meal_base=history.meal_base or "(not yet available)",
            context_base=history.context_base or "(not yet available)",
            interaction_state=interaction_state or history.interaction_state or "(not yet available)",
            user_preferences=user_preferences or "(not yet provided)",
            recommendation_history=self._format_recommendation_history(
                recommendation_history
            ),
            dialogue_state_section=self._format_dialogue_state_section(
                alignment_score=last_alignment_score,
                alignment_reasoning=last_alignment_reasoning,
                certainty_score=last_certainty_score,
                certainty_reasoning=last_certainty_reasoning,
            ),
            recent_turns=history.to_recent_turns_text(),
            qa_status=self._format_qa_status(consecutive_qa_count),
        )
        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user},
        ]

    def parse_output(self, raw_output: str, fallback_phase: str) -> Dict[str, Any]:
        """Parse a planner response without retrying the LLM."""
        phase = self._normalize_phase(fallback_phase)
        try:
            data = load_json_object(raw_output)
        except (JSONOutputError, ValueError, TypeError, AttributeError) as exc:
            data = self._recover_partial_fields(raw_output)
            if not data:
                return self._fallback_plan(phase, f"invalid planner JSON: {exc}")
            data["parse_warning"] = f"partial planner JSON recovery: {exc}"

        parsed_phase = self._normalize_phase(
            str(data.get("phase", data.get("accepted_phase", phase))).strip().lower()
        )
        if parsed_phase not in PHASES:
            parsed_phase = phase

        action = str(data.get("action", "")).strip().lower()
        if action not in VALID_ACTIONS:
            return self._fallback_plan(parsed_phase, f"invalid action: {action!r}")

        user_intent = str(data.get("user_intent", "passive")).strip().lower()
        if user_intent not in VALID_INTENTS:
            user_intent = "passive"

        actionability = str(
            data.get("actionability", "")
            or self._infer_actionability(
                action=action,
                user_intent=user_intent,
                closure_readiness=str(data.get("closure_readiness", "")).strip().lower(),
            )
        ).strip().lower()
        if actionability not in ACTIONABILITY:
            actionability = "insufficient"

        closure_readiness = str(
            data.get("closure_readiness", "")
            or self._infer_closure_readiness(
                action=action,
                user_intent=user_intent,
                follow_action=str(data.get("assessment_followup_action", ""))
                .strip()
                .lower(),
            )
        ).strip().lower()
        if closure_readiness not in CLOSURE_READINESS:
            closure_readiness = "not_ready"

        follow_action = str(data.get("assessment_followup_action", "")).strip().lower()
        follow_phase = str(data.get("assessment_followup_phase", "")).strip().lower()
        if action == "assess":
            if follow_action not in POST_ASSESS_ACTIONS:
                follow_action = ""
            if follow_action:
                expected_follow_phase = (
                    "finalization"
                    if follow_action in ("close", "terminate")
                    else "negotiation"
                    if follow_action == "handoff"
                    else "confirmation"
                    if follow_action == "confirm"
                    else "exploration"
                    if follow_action == "inquire"
                    else "recommendation"
                )
                if follow_phase != expected_follow_phase:
                    follow_phase = expected_follow_phase
        else:
            follow_action = ""
            follow_phase = ""

        try:
            confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        return {
            "intent_summary": str(data.get("intent_summary", "")),
            "user_intent": user_intent,
            "accepted_phase": parsed_phase,
            "phase": parsed_phase,
            "actionability": actionability,
            "action": action,
            "closure_readiness": closure_readiness,
            "reasoning": str(data.get("reasoning", "")),
            "instruction": str(data.get("instruction", "")),
            "assessment_followup_action": follow_action,
            "assessment_followup_phase": follow_phase,
            "assessment_followup_instruction": str(
                data.get("assessment_followup_instruction", "")
            ),
            "confidence": confidence,
            "parse_warning": str(data.get("parse_warning", "")),
        }

    @classmethod
    def _build_system_prompt(
        cls,
        *,
        nutrition_goal: str,
        goal_definition: str,
        config: "AgentConfig",
    ) -> str:
        blocks = [
            DIALOGUE_PLANNER_ROLE_PROMPT.format(
                nutrition_goal=nutrition_goal,
                goal_definition=goal_definition,
            ),
            PHASE_DEFINITIONS,
            PHASE_INSTRUCTIONS,
            ACTION_DEFINITIONS,
            ACTION_INSTRUCTIONS,
            cls._build_dialogue_state_prompt(config),
        ]
        if config.dialogue_planner_use_intents:
            blocks.append(INTENT_DEFINITIONS)
            blocks.append(INTENT_INSTRUCTIONS)
        blocks.extend([
            DIALOGUE_PLANNER_RULES,
            DIALOGUE_PLANNER_OUTPUT_SCHEMA,
        ])
        return "\n\n".join(block.strip() for block in blocks if block and block.strip())

    @staticmethod
    def _build_dialogue_state_prompt(config: "AgentConfig") -> str:
        include_scores = bool(config.dialogue_planner_use_state_scores)
        include_rationales = bool(config.dialogue_planner_use_state_rationales)
        if not include_scores and not include_rationales:
            return ""
        blocks = [
            build_state_merge(
                include_scores=include_scores,
                include_rationales=include_rationales,
            ),
            STATE_DEFINITIONS,
        ]
        if include_scores:
            blocks.append(STATE_SCORE_INSTRUCTIONS)
        if include_rationales:
            blocks.append(STATE_RATIONALE_INSTRUCTIONS)
        return "\n\n".join(block.strip() for block in blocks if block and block.strip())

    @staticmethod
    def _normalize_phase(phase: str) -> str:
        """Normalize legacy phase names to the current phase taxonomy."""
        normalized = str(phase or "").strip().lower()
        if normalized == "motivational_ending":
            return "finalization"
        return normalized if normalized in PHASES else "exploration"

    @staticmethod
    def _format_qa_status(consecutive_qa_count: int) -> str:
        if consecutive_qa_count >= 2:
            return f"LIMIT REACHED ({consecutive_qa_count}/2). Do NOT choose RESPOND."
        if consecutive_qa_count == 1:
            return "1/2 used. RESPOND is available once more if needed."
        return "0/2 used. RESPOND is available."

    def _format_dialogue_state_section(
        self,
        *,
        alignment_score: float | None,
        alignment_reasoning: str | None,
        certainty_score: float | None,
        certainty_reasoning: str | None,
    ) -> str:
        blocks = []
        include_scores = bool(self.config.dialogue_planner_use_state_scores)
        include_rationales = bool(self.config.dialogue_planner_use_state_rationales)
        if include_scores or include_rationales:
            alignment_state = self._format_dialogue_state(
                score=alignment_score,
                reasoning=alignment_reasoning,
                label="alignment",
                include_score=include_scores,
                include_reasoning=include_rationales,
            )
            if alignment_state:
                blocks.append("[Alignment State]\n" + alignment_state)
            certainty_state = self._format_dialogue_state(
                score=certainty_score,
                reasoning=certainty_reasoning,
                label="uncertainty",
                include_score=include_scores,
                include_reasoning=include_rationales,
            )
            if certainty_state:
                blocks.append("[Uncertainty State]\n" + certainty_state)
        return "\n\n".join(blocks)

    @staticmethod
    def _format_dialogue_state(
        *,
        score: float | None,
        reasoning: str | None,
        label: str,
        include_score: bool,
        include_reasoning: bool,
    ) -> str:
        lines = []
        if include_score and score is not None:
            lines.append(f"{label}_score = {score}")
        if include_reasoning and reasoning:
            lines.append(f"reasoning: {reasoning}")
        return "\n".join(lines)

    @staticmethod
    def _format_recommendation_history(
        recommendation_history: Sequence[Mapping[str, Any]],
    ) -> str:
        if not recommendation_history:
            return "None"
        lines = []
        for item in recommendation_history[-3:]:
            rec = dict(item)
            options = rec.get("options")
            if isinstance(options, list) and options:
                option_text = "; ".join(
                    str(option.get("suggestion", ""))
                    for option in options
                    if isinstance(option, dict) and option.get("suggestion")
                )
                if option_text:
                    lines.append(
                        f"Turn {rec.get('turn_idx', '?')}: bundle -> {option_text}"
                    )
                    continue
            lines.append(
                f"Turn {rec.get('turn_idx', '?')}: "
                f"{rec.get('recommendation_type', '?')} - "
                f"{rec.get('target_food', '?')} -> {rec.get('suggestion', '?')}"
            )
        return "\n".join(lines)

    @staticmethod
    def _infer_closure_readiness(
        *,
        action: str,
        user_intent: str,
        follow_action: str,
    ) -> str:
        """Infer a conservative readiness value for older planner outputs."""
        if user_intent == "disengaging" or (
            action in ("close", "terminate") and user_intent == "rejecting"
        ):
            return "boundary_close"
        if action == "confirm" or follow_action == "confirm":
            return "ready_to_close"
        if action in ("close", "terminate") or follow_action in ("close", "terminate"):
            return "ready_to_close"
        return "not_ready"

    @staticmethod
    def _infer_actionability(
        *,
        action: str,
        user_intent: str,
        closure_readiness: str,
    ) -> str:
        """Infer actionability for older planner outputs."""
        if closure_readiness == "boundary_close" or user_intent == "disengaging":
            return "boundary"
        if closure_readiness == "ready_to_close" or action in ("confirm", "close", "terminate"):
            return "settled"
        if closure_readiness == "actionable":
            return "workable"
        if action in ("assess", "recommend", "respond"):
            return "workable"
        return "insufficient"

    @staticmethod
    def _recover_partial_fields(raw_output: str) -> Dict[str, Any]:
        """Recover short top-level fields from a truncated planner JSON object.

        This preserves fields the model already emitted before truncation so a
        malformed rationale cannot erase intent/action information.
        """
        text = raw_output or ""
        recovered: Dict[str, Any] = {}
        string_keys = (
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
        )
        for key in string_keys:
            match = re.search(
                rf'"{re.escape(key)}"\s*:\s*"((?:\\.|[^"\\])*)"',
                text,
                re.DOTALL,
            )
            if match:
                recovered[key] = DialoguePlanner._decode_recovered_string(
                    match.group(1)
                )
        confidence_match = re.search(r'"confidence"\s*:\s*([0-9.]+)', text)
        if confidence_match:
            try:
                recovered["confidence"] = float(confidence_match.group(1))
            except ValueError:
                pass
        return recovered

    @staticmethod
    def _decode_recovered_string(value: str) -> str:
        """Decode a JSON string body without corrupting non-ASCII text."""
        try:
            return json.loads(f'"{value}"')
        except json.JSONDecodeError:
            return value

    @staticmethod
    def _fallback_plan(phase: str, reason: str) -> Dict[str, Any]:
        return {
            "intent_summary": "",
            "user_intent": "passive",
            "accepted_phase": phase,
            "phase": phase,
            "actionability": "insufficient",
            "action": "inquire",
            "closure_readiness": "not_ready",
            "reasoning": f"(fallback: {reason})",
            "instruction": "Ask one concrete question about missing meal details.",
            "assessment_followup_action": "",
            "assessment_followup_phase": "",
            "assessment_followup_instruction": "",
            "confidence": 0.0,
            "parse_warning": reason,
        }
