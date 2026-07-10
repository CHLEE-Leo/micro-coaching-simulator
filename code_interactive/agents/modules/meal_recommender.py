"""Portable micro-coaching agent package module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from ..json_output import JSONOutputError, load_json_object
from ..prompts.roles.meal_recommender import (
    RECOMMENDER_INPUT_TEMPLATE,
    RECOMMENDER_SYSTEM_PROMPT,
)

if TYPE_CHECKING:
    from ..agent_config import AgentConfig


# ------------------------------------------------------------------------------
# Data directory path
# ------------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "additional"

_GOAL_DEF_CACHE: Optional[Dict] = None


def _load_goal_definitions() -> Dict:
    global _GOAL_DEF_CACHE
    if _GOAL_DEF_CACHE is None:
        path = _DATA_DIR / "goal_def_v2.json"
        with open(path, "r", encoding="utf-8") as f:
            _GOAL_DEF_CACHE = json.load(f)
    return _GOAL_DEF_CACHE


# ------------------------------------------------------------------------------
# MealRecommender
# ------------------------------------------------------------------------------

class MealRecommender:
    """MealRecommender component for the portable micro-coaching agent package."""

    def __init__(self, nutrition_goal: str, config: "AgentConfig"):
        self.nutrition_goal = nutrition_goal
        self.config = config

        goal_spec = _load_goal_definitions().get(nutrition_goal, {})
        self._goal_definition = goal_spec.get("definition", "")

        # Recommendation history
        self._recommendation_history: List[Dict] = []
        self._last_recommendation: Optional[Dict] = None

    # Public interface

    def get_messages(
        self,
        meal_base: str,
        alignment_score: float,
        alignment_reasoning: str,
        instruction: str = "",
        user_preferences: str = "",
        interaction_state: str = "",
        recommendation_history: Optional[List[Dict]] = None,
        user_feedback: str = "",
    ) -> List[Dict[str, str]]:
        """get_messages helper for the portable micro-coaching agent package."""

        instruction_block = ""
        if instruction:
            instruction_block = (
                f"\n[Dialogue planner guidance]\n{instruction}\n"
            )

        preferences_block = ""
        if user_preferences:
            preferences_block = (
                f"\n[User preferences & constraints]\n{user_preferences}\n"
                "Respect these when making suggestions. "
                "Do not recommend anything the user cannot eat.\n"
            )

        interaction_state_block = ""
        if interaction_state:
            interaction_state_block = (
                f"\n[Interaction state]\n{interaction_state}\n"
                "Treat answered facts, meal slots, active issue, candidate "
                "options, accepted options, rejected options, unavailable "
                "options, and safety-conflicted options as active planning "
                "evidence. Recommend only for the active issue when one is "
                "present. Preserve slot scope: an item accepted in one slot may "
                "still be rejected as an option for another slot. Prefer concrete "
                "candidate options over invented new options. Do not repeat "
                "or contradict accepted, rejected, or unavailable options. Do "
                "not endorse safety-conflicted options as safe. If a "
                "safety-conflicted option is also user-requested, use "
                "cautious_continuation rather than repeating a removal "
                "recommendation; otherwise, remove or substitute the conflicted "
                "option when appropriate.\n"
            )

        recommendation_history_block = ""
        if recommendation_history:
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
                        "  Turn "
                        f"{r.get('turn_idx', '?')}: parallel adjustments -> "
                        f"{option_text}"
                    )
                    continue
                rec_lines.append(
                    f"  Turn {r.get('turn_idx', '?')}: "
                    f"{r.get('recommendation_type', '?')} - "
                    f"{r.get('target_food', '?')} -> {r.get('suggestion', '?')}"
                )
            recommendation_history_block = (
                "\n[Previous Recommendations]\n"
                + "\n".join(rec_lines)
                + "\nDo NOT repeat these recommendations. "
                "Treat listed bundle items as parallel adjustments, not as "
                "mutually exclusive alternatives. "
                "If the user accepted a previous suggestion, "
                "honor that choice and build upon it instead of contradicting it.\n"
            )

        user_feedback_block = ""
        if user_feedback:
            user_feedback_block = (
                f"\n[User Feedback on Previous Recommendations]\n{user_feedback}\n"
                "Take this feedback into account. "
                "If the user accepted a suggestion, do NOT reverse or escalate it. "
                "If the user rejected it, offer a different alternative.\n"
            )

        system = RECOMMENDER_SYSTEM_PROMPT.format(
            nutrition_goal=self.nutrition_goal.replace("_", " "),
            goal_definition=self._goal_definition,
            alignment_score=f"{alignment_score:.2f}",
            alignment_reasoning=alignment_reasoning or "N/A",
            instruction_block=instruction_block,
            preferences_block=preferences_block,
            interaction_state_block=interaction_state_block,
            recommendation_history_block=recommendation_history_block,
            user_feedback_block=user_feedback_block,
        )
        user = RECOMMENDER_INPUT_TEMPLATE.format(
            meal_base=meal_base or "(no meal base available)",
        )
        return [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]

    def parse_output(self, raw_output: str, turn_idx: int = 0) -> Dict:
        """Parse and store recommendation output with a non-empty fallback."""
        try:
            recommendation = self._parse_recommendation(raw_output)
        except (JSONOutputError, ValueError, TypeError, AttributeError):
            recommendation = self._fallback_recommendation(raw_output)
        self._last_recommendation = recommendation
        self._recommendation_history.append({
            "turn_idx": turn_idx,
            **recommendation,
        })
        return recommendation

    def parse_with_retry(
        self,
        *,
        base_msgs: List[Dict[str, str]],
        raw_output: str,
        reinvoke_fn,
        turn_idx: int = 0,
    ) -> Dict:
        """Parse recommendation JSON, retry once on invalid structured output.

        Recommendation history drives later bundle-scoped negotiation, so a
        malformed output must not silently become an empty recommendation.
        """
        first_err = ""
        try:
            recommendation = self._parse_recommendation(raw_output)
        except (JSONOutputError, ValueError, TypeError, AttributeError) as exc:
            first_err = str(exc)
            retry_msgs = list(base_msgs) + [
                {
                    "role": "user",
                    "content": (
                        "Your previous output was not usable structured JSON.\n"
                        f"Parser error: {first_err}\n\n"
                        "Return exactly one complete JSON object with non-empty "
                        "recommendation fields and an options array. Do not include "
                        "markdown, comments, or text outside JSON."
                    ),
                }
            ]
            retry_raw = reinvoke_fn(retry_msgs)
            try:
                recommendation = self._parse_recommendation(retry_raw)
                recommendation["retry_reason"] = first_err
            except (JSONOutputError, ValueError, TypeError, AttributeError) as retry_exc:
                recommendation = self._fallback_recommendation(
                    retry_raw or raw_output,
                    reason=f"{first_err}; retry: {retry_exc}",
                )

        self._last_recommendation = recommendation
        self._recommendation_history.append({
            "turn_idx": turn_idx,
            **recommendation,
        })
        return recommendation

    @classmethod
    def _parse_recommendation(cls, raw_output: str) -> Dict:
        data = load_json_object(raw_output)
        recommendation = {
            "recommendation_type": cls._normalize_recommendation_type(
                data.get("recommendation_type", "modify")
            ),
            "target_food": str(data.get("target_food", "")).strip(),
            "suggestion": str(data.get("suggestion", "")).strip(),
            "reasoning": str(data.get("reasoning", "")).strip(),
            "expected_impact": cls._normalize_impact(data.get("expected_impact")),
        }
        options = cls._parse_options(data)
        if options:
            recommendation["options"] = options
        cls._validate_recommendation(recommendation)
        return recommendation

    @staticmethod
    def _validate_recommendation(recommendation: Dict) -> None:
        has_suggestion = bool(str(recommendation.get("suggestion", "")).strip())
        has_options = bool(recommendation.get("options"))
        if not has_suggestion and not has_options:
            raise ValueError("recommendation has neither suggestion nor options")
        if has_options:
            for option in recommendation["options"]:
                if not str(option.get("suggestion", "")).strip():
                    raise ValueError("recommendation option missing suggestion")

    @staticmethod
    def _normalize_recommendation_type(value) -> str:
        allowed = {
            "add",
            "remove",
            "modify",
            "substitute",
            "swap",
            "portion",
            "preparation",
            "confirm",
            "cautious_continuation",
        }
        text = str(value or "modify").strip().lower()
        return text if text in allowed else "modify"

    @staticmethod
    def _normalize_impact(value) -> str:
        text = str(value or "low").strip().lower()
        return text if text in {"low", "medium", "high"} else "low"

    @staticmethod
    def _fallback_recommendation(raw_output: str, reason: str = "") -> Dict:
        text = " ".join(str(raw_output or "").split())
        text = text[:260].strip()
        suggestion = (
            text
            if text and not text.startswith("{")
            else (
                "Continue with the safest feasible option already discussed, "
                "while avoiding rejected, unavailable, or safety-conflicted items."
            )
        )
        return {
            "recommendation_type": "modify",
            "target_food": "current unresolved meal component",
            "suggestion": suggestion,
            "reasoning": (
                "Structured recommendation output was invalid; this fallback "
                "keeps the recommendation non-empty and constrained to the "
                "current safe action space."
                + (f" Parser detail: {reason}" if reason else "")
            ),
            "expected_impact": "low",
            "options": [
                {
                    "option_id": "fallback_safe_option",
                    "target_food": "current unresolved meal component",
                    "suggestion": suggestion,
                    "reasoning": "Fallback option produced after invalid structured output.",
                    "expected_impact": "low",
                }
            ],
            "structured_output_degraded": True,
        }

    @staticmethod
    def _parse_options(data: Dict) -> List[Dict[str, str]]:
        raw_options = data.get("options")
        if not isinstance(raw_options, list):
            return []
        options: List[Dict[str, str]] = []
        for idx, raw in enumerate(raw_options[:3], start=1):
            if not isinstance(raw, dict):
                continue
            suggestion = str(raw.get("suggestion", "")).strip()
            target = str(raw.get("target_food", "")).strip()
            if not suggestion and not target:
                continue
            options.append({
                "option_id": str(raw.get("option_id") or f"opt{idx}"),
                "target_food": target,
                "suggestion": suggestion,
                "reasoning": str(raw.get("reasoning", "")),
                "expected_impact": MealRecommender._normalize_impact(
                    raw.get("expected_impact")
                ),
            })
        return options

    def recommend(
        self,
        meal_base: str,
        alignment_score: float,
        alignment_reasoning: str,
        instruction: str = "",
        user_preferences: str = "",
        interaction_state: str = "",
        recommendation_history: Optional[List[Dict]] = None,
        user_feedback: str = "",
        generate_fn=None,
        llm=None,
        turn_idx: int = 0,
    ) -> Dict:
        """recommend helper for the portable micro-coaching agent package."""
        if generate_fn is None:
            from ..openai_client import generate_response
            generate_fn = generate_response

        msgs = self.get_messages(
            meal_base=meal_base,
            alignment_score=alignment_score,
            alignment_reasoning=alignment_reasoning,
            instruction=instruction,
            user_preferences=user_preferences,
            interaction_state=interaction_state,
            recommendation_history=recommendation_history,
            user_feedback=user_feedback,
        )
        raw = generate_fn(
            llm,
            msgs,
            max_new_tokens=getattr(self.config, 'recommendation_max_new_tokens', 300),
            sampling="greedy",
        )
        return self.parse_output(raw, turn_idx=turn_idx)

    # Properties

    @property
    def last_recommendation(self) -> Optional[Dict]:
        """last_recommendation helper for the portable micro-coaching agent package."""
        return self._last_recommendation

    @property
    def recommendation_history(self) -> List[Dict]:
        """recommendation_history helper for the portable micro-coaching agent package."""
        return list(self._recommendation_history)
