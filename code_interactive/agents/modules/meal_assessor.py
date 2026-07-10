"""Meal assessment component for the portable coaching engine."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Sequence

from ..json_output import JSONOutputError, load_json_object
from ..prompts.roles.meal_assessor import (
    MEAL_ASSESSOR_INPUT_TEMPLATE,
    MEAL_ASSESSOR_RETRY_FEEDBACK,
    MEAL_ASSESSOR_SYSTEM_PROMPT,
)

if TYPE_CHECKING:
    from ..agent_config import AgentConfig
    from ..memory.conversation_memory import SharedConversationHistory


FALLBACK_ASSESSMENT = {
    "summary": "",
    "strengths": [],
    "limitations": [],
    "overall": "partially_aligned",
}

FALLBACK_COUNTS: Dict[str, int] = {}


class MealAssessmentParseError(ValueError):
    """Raised when a meal-assessment LLM response violates the output schema."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def log_fallback(source: str, reason: str, raw: str) -> None:
    """Record and print a compact fallback trace for assessment parsing."""
    FALLBACK_COUNTS[source] = FALLBACK_COUNTS.get(source, 0) + 1
    print(
        f"[MealAssessor-Fallback:{source}] "
        f"count={FALLBACK_COUNTS[source]} reason={reason} "
        f"raw={(raw or '').strip()[:120]!r}"
    )


def get_fallback_stats() -> Dict[str, int]:
    """Return fallback counters for tests and runtime diagnostics."""
    return dict(FALLBACK_COUNTS)


def reset_fallback_stats() -> None:
    """Clear fallback counters."""
    FALLBACK_COUNTS.clear()


def _raw_excerpt(raw: str, limit: int = 200) -> str:
    """Compact raw model output for telemetry without bloating session logs."""
    return (raw or "").strip().replace("\n", " ")[:limit]


class MealAssessor:
    """Generate and parse structured meal assessments."""

    _ASSESSMENT_OVERALL_VALID = frozenset({
        "aligned",
        "partially_aligned",
        "not_aligned",
    })

    def __init__(self, nutrition_goal: str, config: "AgentConfig") -> None:
        self.nutrition_goal = nutrition_goal
        self.config = config

        from .meal_recommender import _load_goal_definitions

        goal_spec = _load_goal_definitions().get(nutrition_goal, {})
        self._goal_definition = goal_spec.get("definition", "")
        self._system_prompt = MEAL_ASSESSOR_SYSTEM_PROMPT.format(
            nutrition_goal=nutrition_goal.replace("_", " "),
            goal_definition=self._goal_definition,
        )
        self._last_assessment: Dict[str, Any] | None = None

    def get_messages(
        self,
        *,
        history: "SharedConversationHistory",
        alignment_score: float | None = None,
        alignment_reasoning: str | None = None,
        user_preferences: str = "",
        recommendation_history: Sequence[Mapping[str, Any]] = (),
    ) -> List[Dict[str, str]]:
        """Build LLM messages for assessing the current meal state."""
        user = MEAL_ASSESSOR_INPUT_TEMPLATE.format(
            alignment_score=(
                f"{alignment_score:.2f}" if alignment_score is not None else "N/A"
            ),
            alignment_reasoning=alignment_reasoning or "N/A",
            meal_base=history.meal_base or "(no meal base available)",
            tracker_state=history.tracker_state or "(no tracking state available)",
            context_base=history.context_base or "(no context base available)",
            interaction_state=history.interaction_state or "(no interaction state available)",
            user_preferences=user_preferences or "(no profile constraints provided)",
            recommendation_history=self._format_recommendation_history(
                recommendation_history
            ),
            recent_turns=history.to_recent_turns_text(),
        )
        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user},
        ]

    @classmethod
    def parse_strict(cls, raw_output: str) -> Dict[str, Any]:
        """Parse and validate a strict assessment JSON object."""
        text = (raw_output or "").strip()
        if not text:
            raise MealAssessmentParseError("empty LLM response")

        try:
            data = load_json_object(text)
        except (JSONOutputError, ValueError, TypeError, AttributeError) as exc:
            raise MealAssessmentParseError(f"JSON decode failed: {exc}") from exc

        overall = str(data.get("overall", "")).strip().lower()
        if overall not in cls._ASSESSMENT_OVERALL_VALID:
            raise MealAssessmentParseError(
                f"invalid overall '{overall}' "
                f"(allowed: {sorted(cls._ASSESSMENT_OVERALL_VALID)})"
            )

        return {
            "summary": str(data.get("summary", "")),
            "strengths": cls._as_list(data.get("strengths")),
            "limitations": cls._as_list(data.get("limitations")),
            "overall": overall,
        }

    def parse(self, raw_output: str) -> Dict[str, Any]:
        """Parse an assessment and fall back to an explicit degraded result."""
        try:
            assessment = self.parse_strict(raw_output)
        except MealAssessmentParseError as exc:
            log_fallback("assessment", exc.reason, raw_output)
            assessment = self._degraded_assessment(
                reason=exc.reason,
                raw_output=raw_output,
                retry_attempted=False,
            )
        else:
            assessment = self._with_parse_telemetry(assessment)

        self._last_assessment = assessment
        return assessment

    def parse_with_retry(
        self,
        *,
        base_msgs: List[Dict[str, str]],
        raw_output: str,
        reinvoke_fn,
    ) -> Dict[str, Any]:
        """Retry once on malformed assessment output, then use fallback parsing."""
        try:
            assessment = self.parse_strict(raw_output)
            assessment = self._with_parse_telemetry(assessment)
            self._last_assessment = assessment
            return assessment
        except MealAssessmentParseError as first_err:
            first_error_reason = first_err.reason
            print(
                "[MealAssessor-Retry:assessment] 1st parse failed - "
                f"{first_error_reason}; retrying..."
            )

        try:
            retry_raw = self._reinvoke_with_feedback(
                base_msgs=base_msgs,
                assistant_prev=raw_output,
                error_reason=first_error_reason,
                reinvoke_fn=reinvoke_fn,
            )
        except Exception as exc:  # pragma: no cover - defensive runtime path
            log_fallback("assessment_retry_exception", str(exc)[:120], raw_output)
            assessment = self._degraded_assessment(
                reason=str(exc)[:120],
                raw_output=raw_output,
                retry_attempted=True,
            )
            self._last_assessment = assessment
            return assessment

        try:
            assessment = self.parse_strict(retry_raw)
            FALLBACK_COUNTS["assessment_retry_recovered"] = (
                FALLBACK_COUNTS.get("assessment_retry_recovered", 0) + 1
            )
            assessment = self._with_parse_telemetry(
                assessment,
                retry_attempted=True,
                retry_recovered=True,
            )
            self._last_assessment = assessment
            return assessment
        except MealAssessmentParseError as second_err:
            print(
                "[MealAssessor-Retry:assessment] 2nd parse also failed - "
                f"{second_err.reason}; using fallback."
            )
            log_fallback("assessment_retry_failed", second_err.reason, retry_raw)
            assessment = self._degraded_assessment(
                reason=second_err.reason,
                raw_output=retry_raw,
                retry_attempted=True,
            )
            self._last_assessment = assessment
            return assessment

    @property
    def last_assessment(self) -> Dict[str, Any] | None:
        """Return the most recent parsed assessment."""
        return self._last_assessment

    @staticmethod
    def _with_parse_telemetry(
        assessment: Dict[str, Any],
        *,
        retry_attempted: bool = False,
        retry_recovered: bool = False,
    ) -> Dict[str, Any]:
        """Attach internal parse telemetry without changing semantic fields."""
        return {
            **assessment,
            "_degraded": False,
            "_parse_error": "",
            "_raw_output_excerpt": "",
            "_retry_attempted": retry_attempted,
            "_retry_recovered": retry_recovered,
        }

    @staticmethod
    def _degraded_assessment(
        *,
        reason: str,
        raw_output: str,
        retry_attempted: bool,
    ) -> Dict[str, Any]:
        """Create a telemetry-marked fallback that cannot masquerade as normal."""
        return {
            **FALLBACK_ASSESSMENT,
            "_degraded": True,
            "_parse_error": reason,
            "_raw_output_excerpt": _raw_excerpt(raw_output),
            "_retry_attempted": retry_attempted,
            "_retry_recovered": False,
        }

    @staticmethod
    def _format_recommendation_history(
        recommendation_history: Sequence[Mapping[str, Any]],
    ) -> str:
        if not recommendation_history:
            return "(none)"
        rec_lines: list[str] = []
        for item in recommendation_history:
            if not isinstance(item, Mapping):
                continue
            options = item.get("options")
            if isinstance(options, list) and options:
                option_text = "; ".join(
                    str(option.get("suggestion", ""))
                    for option in options
                    if isinstance(option, Mapping) and option.get("suggestion")
                )
                if option_text:
                    rec_lines.append(
                        f"- Turn {item.get('turn_idx', '?')}: "
                        f"parallel adjustments -> {option_text}"
                    )
                    continue
            suggestion = str(item.get("suggestion", "")).strip()
            target = str(item.get("target_food", "")).strip()
            turn_idx = item.get("turn_idx", "?")
            if suggestion or target:
                rec_lines.append(f"- Turn {turn_idx}: {suggestion} (target: {target})")
        return "\n".join(rec_lines) or "(none)"

    @staticmethod
    def _reinvoke_with_feedback(
        *,
        base_msgs: List[Dict[str, str]],
        assistant_prev: str,
        error_reason: str,
        reinvoke_fn,
    ) -> str:
        retry_msgs = list(base_msgs) + [
            {"role": "assistant", "content": assistant_prev or ""},
            {
                "role": "user",
                "content": MEAL_ASSESSOR_RETRY_FEEDBACK.format(
                    error=error_reason
                ),
            },
        ]
        return reinvoke_fn(retry_msgs)

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]
