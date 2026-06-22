"""Portable micro-coaching agent package module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

from ..json_output import JSONOutputError, load_json_object
from ..prompts.definitions.actions import ACTION_DEFINITIONS, ACTION_INSTRUCTIONS
from ..prompts.definitions.phases import PHASE_DEFINITIONS, PHASE_INSTRUCTIONS
from ..prompts.definitions.intents import INTENT_DEFINITIONS, INTENT_INSTRUCTIONS
from ..prompts.definitions.states import (
    STATE_DEFINITIONS,
    STATE_RATIONALE_INSTRUCTIONS,
    STATE_SCORE_INSTRUCTIONS,
    build_state_merge,
)
from ..prompts.roles.orchestrator import (
    ORCHESTRATOR_CONVERSATION_RULES,
    ORCHESTRATOR_OUTPUT_SCHEMA,
    ORCHESTRATOR_PHASE_ADVISORY,
    ORCHESTRATOR_ROLE_PROMPT,
    ORCHESTRATOR_INPUT_TEMPLATE,
)

if TYPE_CHECKING:
    from ..agent_config import AgentConfig
    from ..memory.conversation_memory import SharedConversationHistory


# ------------------------------------------------------------------------------
# Phase definitions
# ------------------------------------------------------------------------------

PHASES = (
    "exploration",
    "recommendation",
    "negotiation",
    "motivational_ending",
)


def _clean_prompt_blocks(blocks: List[str]) -> str:
    return "\n\n".join(block.strip() for block in blocks if block and block.strip())


def _build_dialogue_state_prompt(config: "AgentConfig") -> str:
    include_scores = bool(config.orchestrator_use_state_scores)
    include_rationales = bool(config.orchestrator_use_state_rationales)
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
    if not any(blocks):
        return ""
    return "=== DECISION GUIDELINES ===\n\n" + _clean_prompt_blocks(blocks)


def build_router_system_prompt(
    *,
    nutrition_goal: str,
    goal_definition: str,
    config: "AgentConfig",
) -> str:
    blocks = [
        ORCHESTRATOR_ROLE_PROMPT.format(
            nutrition_goal=nutrition_goal,
            goal_definition=goal_definition,
        ),
        PHASE_DEFINITIONS,
        PHASE_INSTRUCTIONS,
    ]
    blocks.append(ORCHESTRATOR_PHASE_ADVISORY)
    blocks.append(ACTION_DEFINITIONS)
    blocks.append(ACTION_INSTRUCTIONS)
    blocks.append(_build_dialogue_state_prompt(config))
    if config.orchestrator_use_intents:
        blocks.append(INTENT_DEFINITIONS)
        blocks.append(INTENT_INSTRUCTIONS)
    blocks.extend([
        ORCHESTRATOR_CONVERSATION_RULES,
        ORCHESTRATOR_OUTPUT_SCHEMA,
    ])
    return _clean_prompt_blocks(blocks)

# ------------------------------------------------------------------------------
# Prompt
# assess action Router .
# Allowed values
# ------------------------------------------------------------------------------

_POST_ASSESS_ROUTER_SYSTEM = """\
You are the central orchestrator of a nutritional micro-coaching conversation.

The system has just completed a meal evaluation. You must now decide the \
FOLLOW-UP action for this same turn. The evaluation feedback has already been \
shown to the user as the first message in this turn.

Nutritional goal: {nutrition_goal}
Goal definition: {goal_definition}

(The per-turn context - Evaluation result, Alignment State, and the user's \
current intent - is provided in the user message below.)

=== FOLLOW-UP ACTIONS ===

Choose ONE follow-up action. The evaluation message is already sent - \
your choice generates a SECOND message in the same turn (consecutive bubbles).

- INQUIRE - Ask about preferences, constraints, or feasibility to prepare a recommendation. \
Choose when: meal needs improvement AND user preferences are unknown.
- RECOMMEND - Suggest a specific meal improvement immediately. \
Choose when: meal needs improvement AND user preferences are already known \
(from profile or conversation). Skips preference gathering.
- CLOSE - Wrap up with encouragement. \
Choose when: meal is well-aligned with the goal, OR user has indicated they don't want \
recommendations (rejecting/disengaging intent).
- TERMINATE - End the conversation immediately with a brief close. \
Choose when: conversation cannot continue or max turns are near.

RULES:
- Do NOT choose ASSESS (already done) or ask for meal exploration details (assessment is complete).
- If the user has been rejecting/disengaging, choose CLOSE.
- If alignment is high (>= 0.8), prefer CLOSE.
- If user preferences are already known (from profile or conversation context), \
skip INQUIRE and go directly to RECOMMEND.

Return ONLY a JSON object:
{{"action": "inquire" | "recommend" | "close" | "terminate", \
"accepted_phase": "recommendation" | "motivational_ending", \
"reasoning": "<1-2 sentences>", \
"instruction": "<SHORT guidance for sub-agent>"}}\
"""

_POST_ASSESS_ROUTER_USER = """\
[Turn {turn_idx} / {max_turns}]

[Evaluation Result]
overall: {assessment_overall}
strengths: {assessment_strengths}
limitations: {assessment_limitations}

[Alignment State]
{alignment_state}

[User Intent (from initial Router call)]
{user_intent}

[Meal Base]
{meal_base}

[User Preferences]
{user_preferences}

[Recent Conversation]
{recent_turns}

Decide the follow-up action after evaluation.\
"""

# ------------------------------------------------------------------------------
# System prompt
# ------------------------------------------------------------------------------

_ASSESSMENT_SYSTEM = """\
You are evaluating a user's meal against a nutritional goal.

Nutritional goal: {nutrition_goal}
Goal definition: {goal_definition}

(The per-turn alignment assessment is provided in the user message below.)

Based on the Meal Fact Sheet and alignment data, generate a concise meal assessment.

Rules:
- Be specific: reference actual foods from the Fact Sheet.
- Keep strengths/limitations to 1-3 items each.
- "overall" must reflect whether the meal truly meets the goal.

Output ONLY a JSON object:
{{"summary": "<1-2 sentence meal overview>", \
"strengths": ["<positive aspect>", ...], \
"limitations": ["<area for improvement>", ...], \
"overall": "aligned" | "partially_aligned" | "not_aligned"}}\
"""

_ASSESSMENT_USER = """\
[Alignment Assessment]
score: {alignment_score}
reasoning: {alignment_reasoning}

[Meal Base]
{meal_base}

Generate the meal evaluation.\
"""


# ------------------------------------------------------------------------------
# action
# ------------------------------------------------------------------------------

_VALID_ACTIONS = frozenset({
    "inquire", "assess", "recommend", "respond", "close", "terminate",
})

_VALID_PHASES = frozenset(PHASES)

# user_intent 7
# Dialogue Act Theory + Mixed-Initiative Interaction
# Meal
# Coach
# inquiring
# Coach
# passive / /
# rejecting
# disengaging
_VALID_INTENTS = frozenset({
    "informing", "accepting", "inquiring", "deferring",
    "passive", "rejecting", "disengaging",
})

_FALLBACK_DECISION = {
    "intent_summary": "",
    "user_intent": "passive",
    "action": "inquire",
    "reasoning": "(fallback: orchestrator output could not be parsed)",
    "instruction": "",
}

_FALLBACK_ASSESSMENT = {
    "summary": "",
    "strengths": [],
    "limitations": [],
    "overall": "partially_aligned",
}


# ------------------------------------------------------------------------------
# Retry handling
# ------------------------------------------------------------------------------
#
# Orchestrator JSON LLM /
# Error handling
# b retry . 3-
#
# 1 Strict JSON + OrchestratorParseError
# . retry
# 2 Retry strict LLM
# . fallback decision .
# 3 Fallback counter fallback _FALLBACK_COUNTS
# _log_fallback . get_fallback_stats
# Read helpers
# ------------------------------------------------------------------------------


class OrchestratorParseError(ValueError):
    """OrchestratorParseError component for the portable micro-coaching agent package."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


# Fallback .
# .
_FALLBACK_COUNTS: Dict[str, int] = {}


def _log_fallback(source: str, reason: str, raw: str) -> None:
    """_log_fallback helper for the portable micro-coaching agent package."""
    _FALLBACK_COUNTS[source] = _FALLBACK_COUNTS.get(source, 0) + 1
    # grep
    print(
        f"[Orchestrator-Fallback:{source}] "
        f"count={_FALLBACK_COUNTS[source]} reason={reason} "
        f"raw={(raw or '').strip()[:120]!r}"
    )


def get_fallback_stats() -> Dict[str, int]:
    """get_fallback_stats helper for the portable micro-coaching agent package."""
    return dict(_FALLBACK_COUNTS)


def reset_fallback_stats() -> None:
    """reset_fallback_stats helper for the portable micro-coaching agent package."""
    _FALLBACK_COUNTS.clear()


# Prompt
#
# strict messages user
# LLM .
# LLM JSON .

_ROUTER_RETRY_FEEDBACK = (
    "Your previous response could not be parsed as a valid routing decision.\n"
    "Reason: {error}\n\n"
    "Return ONLY a single JSON object with the exact fields specified earlier "
    "(intent_summary, user_intent, accepted_phase, action, reasoning, instruction). "
    "No markdown, no prose, no code fence."
)

_POST_ASSESS_RETRY_FEEDBACK = (
    "Your previous response could not be parsed as a valid post-assessment "
    "follow-up decision.\nReason: {error}\n\n"
    "Return ONLY a JSON object with fields: action, accepted_phase, reasoning, "
    "instruction. action must be one of inquire | recommend | close | terminate."
)

_ASSESSMENT_RETRY_FEEDBACK = (
    "Your previous response could not be parsed as a valid meal assessment.\n"
    "Reason: {error}\n\n"
    "Return ONLY a JSON object with fields: summary (string), strengths (list), "
    "limitations (list), overall (one of: aligned | partially_aligned | not_aligned)."
)


# ------------------------------------------------------------------------------
# Orchestrator
# ------------------------------------------------------------------------------

class Orchestrator:
    """Orchestrator component for the portable micro-coaching agent package."""

    def __init__(self, nutrition_goal: str, config: "AgentConfig"):
        self.nutrition_goal = nutrition_goal
        self.config = config

        from .meal_recommender import _load_goal_definitions
        goal_spec = _load_goal_definitions().get(nutrition_goal, {})
        self._goal_definition = goal_spec.get("definition", "")

        # System prompt Phase 2-2
        # OpenAI Responses API prefix prompt cache
        # System prompt
        # user . __init__
        # format -> .
        _goal_text = nutrition_goal.replace("_", " ")
        self._router_system = build_router_system_prompt(
            nutrition_goal=_goal_text,
            goal_definition=self._goal_definition,
            config=self.config,
        )
        self._post_assess_system = _POST_ASSESS_ROUTER_SYSTEM.format(
            nutrition_goal=_goal_text,
            goal_definition=self._goal_definition,
        )
        self._assessment_system = _ASSESSMENT_SYSTEM.format(
            nutrition_goal=_goal_text,
            goal_definition=self._goal_definition,
        )
        self._decision_history: List[Dict] = []
        self._last_decision: Optional[Dict] = None
        self._last_assessment: Optional[Dict] = None

    # ======================================================================
    # Router
    # ======================================================================

    def get_routing_messages(
        self,
        history: "SharedConversationHistory",
        turn_idx: int,
        phase: str = "exploration",
        recommendation_history: Optional[List[Dict]] = None,
        consecutive_qa_count: int = 0,
        last_alignment_score: Optional[float] = None,
        last_alignment_reasoning: Optional[str] = None,
        last_certainty_score: Optional[float] = None,
        last_certainty_reasoning: Optional[str] = None,
        user_preferences: str = "",
        phase_prediction_reasoning: str = "",
        phase_prediction_confidence: Optional[float] = None,
    ) -> List[Dict[str, str]]:
        """get_routing_messages helper for the portable micro-coaching agent package."""
        # QA abuse
        if consecutive_qa_count >= 2:
            qa_status = f"LIMIT REACHED ({consecutive_qa_count}/2). Do NOT choose RESPOND this turn."
        elif consecutive_qa_count == 1:
            qa_status = f"1/2 used. You may use RESPOND once more if needed."
        else:
            qa_status = "0/2 used. RESPOND is available."

        rec_history_text = "None"
        if recommendation_history:
            rec_lines = [
                f"  Turn {r.get('turn_idx', '?')}: "
                f"{r.get('recommendation_type', '?')} - "
                f"{r.get('target_food', '?')} -> {r.get('suggestion', '?')}"
                for r in recommendation_history
            ]
            rec_history_text = "\n".join(rec_lines)

        # System prompt
        # . current_phase / qa_status user
        # prompt cache .
        system = self._router_system

        if last_alignment_score is not None:
            alignment_state = f"score = {last_alignment_score:.2f} (0 = not aligned, 1 = fully aligned)"
            if last_alignment_reasoning:
                alignment_state += f"\nreasoning: {last_alignment_reasoning}"
        else:
            alignment_state = "(not yet measured)"

        if last_certainty_score is not None:
            uncertainty_state = f"score = {last_certainty_score:.2f} (0 = no info, 0.85+ = enough info to judge, 1 = complete)"
            if last_certainty_reasoning:
                uncertainty_state += f"\nreasoning: {last_certainty_reasoning}"
        else:
            uncertainty_state = "(not yet measured)"

        user = ORCHESTRATOR_INPUT_TEMPLATE.format(
            turn_idx=turn_idx,
            max_turns=self.config.max_turns,
            predicted_phase=phase,
            phase_confidence=(
                f"{phase_prediction_confidence:.2f}"
                if phase_prediction_confidence is not None
                else "(not provided)"
            ),
            phase_reasoning=phase_prediction_reasoning or "(not provided)",
            meal_base=history.meal_base or "(not yet available)",
            context_base=history.context_base or "(not yet available)",
            user_preferences=user_preferences or "(not yet provided)",
            recommendation_history=rec_history_text,
            alignment_state=alignment_state,
            uncertainty_state=uncertainty_state,
            recent_turns=history.to_recent_turns_text(),
            qa_status=qa_status,
        )
        return [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]

    # ----------------------------------------------------------------------
    # Post-evaluation Router
    # assess action .
    # ----------------------------------------------------------------------

    # Allowed values
    _POST_ASSESS_VALID = frozenset({
        "inquire", "recommend", "close", "terminate",
    })

    def get_post_assessment_routing_messages(
        self,
        history: "SharedConversationHistory",
        turn_idx: int,
        assessment_result: Dict,
        alignment_score: Optional[float] = None,
        alignment_reasoning: Optional[str] = None,
        user_intent: str = "passive",
        user_preferences: str = "",
    ) -> List[Dict[str, str]]:
        """get_post_assessment_routing_messages helper for the portable micro-coaching agent package."""
        if alignment_score is not None:
            alignment_state = f"score = {alignment_score:.2f}"
            if alignment_reasoning:
                alignment_state += f"\nreasoning: {alignment_reasoning}"
        else:
            alignment_state = "(not measured)"

        # System prompt
        # Evaluation result/alignment/user_intent user .
        system = self._post_assess_system
        user = _POST_ASSESS_ROUTER_USER.format(
            turn_idx=turn_idx,
            max_turns=self.config.max_turns,
            assessment_overall=assessment_result.get("overall", "partially_aligned"),
            assessment_strengths=", ".join(assessment_result.get("strengths", [])) or "(none)",
            assessment_limitations=", ".join(assessment_result.get("limitations", [])) or "(none)",
            alignment_state=alignment_state,
            user_intent=user_intent,
            meal_base=history.meal_base or "(not yet available)",
            user_preferences=user_preferences or "(not yet provided)",
            recent_turns=history.to_recent_turns_text(),
        )
        return [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]

    def _parse_post_assessment_strict(self, raw_output: str) -> Dict:
        """_parse_post_assessment_strict helper for the portable micro-coaching agent package."""
        text = (raw_output or "").strip()
        if not text:
            raise OrchestratorParseError("empty LLM response")

        try:
            data = load_json_object(text)
        except (JSONOutputError, ValueError, TypeError, AttributeError) as e:
            raise OrchestratorParseError(f"JSON decode failed: {e}") from e

        action = str(data.get("action", "")).strip().lower()
        if not action:
            raise OrchestratorParseError("missing 'action' field")
        if action not in self._POST_ASSESS_VALID:
            raise OrchestratorParseError(
                f"invalid action '{action}' "
                f"(allowed: {sorted(self._POST_ASSESS_VALID)})"
            )
        raw_phase = str(data.get("accepted_phase", "")).strip().lower()
        if raw_phase in ("null", "none", ""):
            raw_phase = ""
        if raw_phase and raw_phase not in _VALID_PHASES:
            raise OrchestratorParseError(
                f"invalid accepted_phase '{raw_phase}' "
                f"(allowed: {sorted(_VALID_PHASES)})"
            )
        if not raw_phase:
            raw_phase = (
                "motivational_ending"
                if action in ("close", "terminate")
                else "recommendation"
            )

        return {
            "action": action,
            "accepted_phase": raw_phase,
            "reasoning": str(data.get("reasoning", "")),
            "instruction": str(data.get("instruction", "")),
        }

    def parse_post_assessment_routing(self, raw_output: str) -> Dict:
        """parse_post_assessment_routing helper for the portable micro-coaching agent package."""
        try:
            return self._parse_post_assessment_strict(raw_output)
        except OrchestratorParseError as e:
            _log_fallback("post_assessment", e.reason, raw_output)
            return {
                "action": "inquire",
                "accepted_phase": "recommendation",
                "reasoning": f"(fallback: {e.reason})",
                "instruction": "",
            }

    @staticmethod
    def _parse_routing_strict(raw_output: str, phase: str) -> Dict:
        """_parse_routing_strict helper for the portable micro-coaching agent package."""
        text = (raw_output or "").strip()
        if not text:
            raise OrchestratorParseError("empty LLM response")

        try:
            data = load_json_object(text)
        except (JSONOutputError, ValueError, TypeError, AttributeError) as e:
            raise OrchestratorParseError(f"JSON decode failed: {e}") from e

        # Validation
        action = str(data.get("action", "")).strip().lower()
        if not action:
            raise OrchestratorParseError("missing 'action' field")
        if action not in _VALID_ACTIONS:
            raise OrchestratorParseError(
                f"invalid action '{action}' "
                f"(allowed: {sorted(_VALID_ACTIONS)})"
            )
        accepted_phase = str(data.get("accepted_phase", phase)).strip().lower()
        if accepted_phase in ("null", "none", ""):
            accepted_phase = phase
        if accepted_phase not in _VALID_PHASES:
            raise OrchestratorParseError(
                f"invalid accepted_phase '{accepted_phase}' "
                f"(allowed: {sorted(_VALID_PHASES)})"
            )

        # Normalization
        # Error handling
        # strict . LLM .
        raw_intent = str(data.get("user_intent", "passive")).strip().lower()
        if raw_intent not in _VALID_INTENTS:
            raw_intent = "passive"

        return {
            "intent_summary": str(data.get("intent_summary", "")),
            "user_intent": raw_intent,
            "accepted_phase": accepted_phase,
            "action": action,
            "reasoning": str(data.get("reasoning", "")),
            "instruction": str(data.get("instruction", "")),
        }

    def parse_routing(
        self,
        raw_output: str,
        turn_idx: int = 0,
        phase: str = "exploration",
    ) -> Dict:
        """parse_routing helper for the portable micro-coaching agent package."""
        try:
            decision = self._parse_routing_strict(raw_output, phase)
        except OrchestratorParseError as e:
            _log_fallback("routing", e.reason, raw_output)
            decision = dict(_FALLBACK_DECISION)
            decision["action"] = "inquire"
            decision["accepted_phase"] = phase
            decision["reasoning"] = (
                f"(parse error: {e.reason}) raw: {(raw_output or '')[:200]}"
            )

        self._last_decision = decision
        self._decision_history.append({"turn_idx": turn_idx, **decision})
        return decision

    def route(
        self,
        history: "SharedConversationHistory",
        turn_idx: int,
        phase: str = "exploration",
        recommendation_history: Optional[List[Dict]] = None,
        generate_fn=None,
        llm=None,
    ) -> Dict:
        """route helper for the portable micro-coaching agent package."""
        if generate_fn is None:
            from ..openai_client import generate_response
            generate_fn = generate_response

        # Safety guard
        if turn_idx >= self.config.max_turns - 1:
            forced = {
                "action": "terminate",
                "accepted_phase": "motivational_ending",
                "reasoning": f"Max turns ({self.config.max_turns}) reached.",
                "instruction": "Thank you for sharing about your meal!",
            }
            self._last_decision = forced
            self._decision_history.append({"turn_idx": turn_idx, **forced})
            return forced

        msgs = self.get_routing_messages(
            history=history,
            turn_idx=turn_idx,
            phase=phase,
            recommendation_history=recommendation_history,
        )
        raw = generate_fn(
            llm, msgs,
            max_new_tokens=getattr(self.config, 'orchestrator_max_new_tokens', 200),
            sampling="greedy",
        )
        return self.parse_routing(raw, turn_idx=turn_idx, phase=phase)

    # ======================================================================
    # Assessment
    # ======================================================================

    def get_assessment_messages(
        self,
        history: "SharedConversationHistory",
        alignment_score: Optional[float] = None,
        alignment_reasoning: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """get_assessment_messages helper for the portable micro-coaching agent package."""
        user = _ASSESSMENT_USER.format(
            alignment_score=(
                f"{alignment_score:.2f}" if alignment_score is not None else "N/A"
            ),
            alignment_reasoning=alignment_reasoning or "N/A",
            meal_base=history.meal_base or "(no meal base available)",
        )
        return [
            {"role": "system", "content": self._assessment_system},
            {"role": "user",   "content": user},
        ]

    # Allowed values
    _ASSESSMENT_OVERALL_VALID = frozenset({"aligned", "partially_aligned", "not_aligned"})

    @classmethod
    def _parse_assessment_strict(cls, raw_output: str) -> Dict:
        """_parse_assessment_strict helper for the portable micro-coaching agent package."""
        text = (raw_output or "").strip()
        if not text:
            raise OrchestratorParseError("empty LLM response")

        try:
            data = load_json_object(text)
        except (JSONOutputError, ValueError, TypeError, AttributeError) as e:
            raise OrchestratorParseError(f"JSON decode failed: {e}") from e

        overall = str(data.get("overall", "")).strip().lower()
        if overall not in cls._ASSESSMENT_OVERALL_VALID:
            raise OrchestratorParseError(
                f"invalid overall '{overall}' "
                f"(allowed: {sorted(cls._ASSESSMENT_OVERALL_VALID)})"
            )

        # strengths / limitations
        def _as_list(v) -> List:
            if v is None:
                return []
            if isinstance(v, list):
                return v
            return [v]

        return {
            "summary": str(data.get("summary", "")),
            "strengths": _as_list(data.get("strengths")),
            "limitations": _as_list(data.get("limitations")),
            "overall": overall,
        }

    def parse_assessment(self, raw_output: str) -> Dict:
        """parse_assessment helper for the portable micro-coaching agent package."""
        try:
            assessment = self._parse_assessment_strict(raw_output)
        except OrchestratorParseError as e:
            _log_fallback("assessment", e.reason, raw_output)
            assessment = dict(_FALLBACK_ASSESSMENT)
            assessment["summary"] = (
                f"(parse error: {e.reason}) raw: {(raw_output or '')[:200]}"
            )

        self._last_assessment = assessment
        return assessment

    def assess(
        self,
        history: "SharedConversationHistory",
        alignment_score: Optional[float] = None,
        alignment_reasoning: Optional[str] = None,
        generate_fn=None,
        llm=None,
    ) -> Dict:
        """assess helper for the portable micro-coaching agent package."""
        if generate_fn is None:
            from ..openai_client import generate_response
            generate_fn = generate_response

        msgs = self.get_assessment_messages(
            history=history,
            alignment_score=alignment_score,
            alignment_reasoning=alignment_reasoning,
        )
        raw = generate_fn(
            llm, msgs,
            max_new_tokens=getattr(self.config, 'assessment_max_new_tokens', 500),
            sampling="greedy",
            stop_at_newline=False,
        )
        return self.parse_assessment_with_retry(
            base_msgs=msgs,
            raw_output=raw,
            reinvoke_fn=lambda retry_msgs: generate_fn(
                llm,
                retry_msgs,
                max_new_tokens=getattr(self.config, 'assessment_max_new_tokens', 500),
                sampling="greedy",
                stop_at_newline=False,
            ),
        )

    # ======================================================================
    # Parse-retry strict LLM .
    # session_manager / engine LLM
    # raw raw + retry re-invoke
    # . Router 4-way pool
    # Retry handling
    # ======================================================================

    @staticmethod
    def _reinvoke_with_feedback(
        base_msgs: List[Dict[str, str]],
        assistant_prev: str,
        feedback_template: str,
        error_reason: str,
        reinvoke_fn,
    ) -> str:
        """_reinvoke_with_feedback helper for the portable micro-coaching agent package."""
        retry_msgs = list(base_msgs) + [
            {"role": "assistant", "content": assistant_prev or ""},
            {"role": "user", "content": feedback_template.format(error=error_reason)},
        ]
        return reinvoke_fn(retry_msgs)

    def parse_routing_with_retry(
        self,
        base_msgs: List[Dict[str, str]],
        raw_output: str,
        turn_idx: int,
        phase: str,
        reinvoke_fn,
    ) -> Dict:
        """parse_routing_with_retry helper for the portable micro-coaching agent package."""
        # 1 strict parse
        try:
            decision = self._parse_routing_strict(raw_output, phase)
            self._last_decision = decision
            self._decision_history.append({"turn_idx": turn_idx, **decision})
            return decision
        except OrchestratorParseError as first_err:
            first_error_reason = first_err.reason
            print(
                f"[Orchestrator-Retry:routing] 1st parse failed "
                f"(turn={turn_idx}, phase={phase}) - {first_error_reason}; retrying..."
            )

        # 2
        try:
            retry_raw = self._reinvoke_with_feedback(
                base_msgs=base_msgs,
                assistant_prev=raw_output,
                feedback_template=_ROUTER_RETRY_FEEDBACK,
                error_reason=first_error_reason,
                reinvoke_fn=reinvoke_fn,
            )
        except Exception as e:
            # reinvoke_fn
            _log_fallback("routing_retry_exception", str(e)[:120], raw_output)
            return self.parse_routing(raw_output, turn_idx=turn_idx, phase=phase)

        try:
            decision = self._parse_routing_strict(retry_raw, phase)
            # Recovery handling
            _FALLBACK_COUNTS["routing_retry_recovered"] = (
                _FALLBACK_COUNTS.get("routing_retry_recovered", 0) + 1
            )
            self._last_decision = decision
            self._decision_history.append({"turn_idx": turn_idx, **decision})
            return decision
        except OrchestratorParseError as second_err:
            # 2 safe parse_routing fallback
            print(
                f"[Orchestrator-Retry:routing] 2nd parse also failed "
                f"(turn={turn_idx}) - {second_err.reason}; using fallback."
            )
            return self.parse_routing(retry_raw, turn_idx=turn_idx, phase=phase)

    def parse_post_assessment_with_retry(
        self,
        base_msgs: List[Dict[str, str]],
        raw_output: str,
        reinvoke_fn,
    ) -> Dict:
        """parse_post_assessment_with_retry helper for the portable micro-coaching agent package."""
        try:
            return self._parse_post_assessment_strict(raw_output)
        except OrchestratorParseError as first_err:
            first_error_reason = first_err.reason
            print(
                f"[Orchestrator-Retry:post_assessment] 1st parse failed - "
                f"{first_error_reason}; retrying..."
            )

        try:
            retry_raw = self._reinvoke_with_feedback(
                base_msgs=base_msgs,
                assistant_prev=raw_output,
                feedback_template=_POST_ASSESS_RETRY_FEEDBACK,
                error_reason=first_error_reason,
                reinvoke_fn=reinvoke_fn,
            )
        except Exception as e:
            _log_fallback("post_assessment_retry_exception", str(e)[:120], raw_output)
            return self.parse_post_assessment_routing(raw_output)

        try:
            decision = self._parse_post_assessment_strict(retry_raw)
            _FALLBACK_COUNTS["post_assessment_retry_recovered"] = (
                _FALLBACK_COUNTS.get("post_assessment_retry_recovered", 0) + 1
            )
            return decision
        except OrchestratorParseError as second_err:
            print(
                f"[Orchestrator-Retry:post_assessment] 2nd parse also failed - "
                f"{second_err.reason}; using fallback."
            )
            return self.parse_post_assessment_routing(retry_raw)

    def parse_assessment_with_retry(
        self,
        base_msgs: List[Dict[str, str]],
        raw_output: str,
        reinvoke_fn,
    ) -> Dict:
        """parse_assessment_with_retry helper for the portable micro-coaching agent package."""
        try:
            assessment = self._parse_assessment_strict(raw_output)
            self._last_assessment = assessment
            return assessment
        except OrchestratorParseError as first_err:
            first_error_reason = first_err.reason
            print(
                f"[Orchestrator-Retry:assessment] 1st parse failed - "
                f"{first_error_reason}; retrying..."
            )

        try:
            retry_raw = self._reinvoke_with_feedback(
                base_msgs=base_msgs,
                assistant_prev=raw_output,
                feedback_template=_ASSESSMENT_RETRY_FEEDBACK,
                error_reason=first_error_reason,
                reinvoke_fn=reinvoke_fn,
            )
        except Exception as e:
            _log_fallback("assessment_retry_exception", str(e)[:120], raw_output)
            return self.parse_assessment(raw_output)

        try:
            assessment = self._parse_assessment_strict(retry_raw)
            _FALLBACK_COUNTS["assessment_retry_recovered"] = (
                _FALLBACK_COUNTS.get("assessment_retry_recovered", 0) + 1
            )
            self._last_assessment = assessment
            return assessment
        except OrchestratorParseError as second_err:
            print(
                f"[Orchestrator-Retry:assessment] 2nd parse also failed - "
                f"{second_err.reason}; using fallback."
            )
            return self.parse_assessment(retry_raw)

    # Properties

    @property
    def last_decision(self) -> Optional[Dict]:
        """last_decision helper for the portable micro-coaching agent package."""
        return self._last_decision

    @property
    def decision_history(self) -> List[Dict]:
        """decision_history helper for the portable micro-coaching agent package."""
        return list(self._decision_history)

    @property
    def last_assessment(self) -> Optional[Dict]:
        """last_assessment helper for the portable micro-coaching agent package."""
        return self._last_assessment
