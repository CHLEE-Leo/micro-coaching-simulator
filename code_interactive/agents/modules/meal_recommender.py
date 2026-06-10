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
        recommendation_history: Optional[List[Dict]] = None,
        user_feedback: str = "",
    ) -> List[Dict[str, str]]:
        """get_messages helper for the portable micro-coaching agent package."""

        instruction_block = ""
        if instruction:
            instruction_block = (
                f"\n[Orchestrator guidance]\n{instruction}\n"
            )

        preferences_block = ""
        if user_preferences:
            preferences_block = (
                f"\n[User preferences & constraints]\n{user_preferences}\n"
                "Respect these when making suggestions. "
                "Do not recommend anything the user cannot eat.\n"
            )

        recommendation_history_block = ""
        if recommendation_history:
            rec_lines = [
                f"  Turn {r.get('turn_idx', '?')}: "
                f"{r.get('recommendation_type', '?')} - "
                f"{r.get('target_food', '?')} -> {r.get('suggestion', '?')}"
                for r in recommendation_history
            ]
            recommendation_history_block = (
                "\n[Previous Recommendations]\n"
                + "\n".join(rec_lines)
                + "\nDo NOT repeat these recommendations. "
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
        """parse_output helper for the portable micro-coaching agent package."""
        recommendation = {
            "recommendation_type": "modify",
            "target_food": "",
            "suggestion": "",
            "reasoning": "(parse error)",
            "expected_impact": "low",
        }

        try:
            data = load_json_object(raw_output)

            recommendation = {
                "recommendation_type": str(data.get("recommendation_type", "modify")),
                "target_food": str(data.get("target_food", "")),
                "suggestion": str(data.get("suggestion", "")),
                "reasoning": str(data.get("reasoning", "")),
                "expected_impact": str(data.get("expected_impact", "low")),
            }
        except (JSONOutputError, ValueError, TypeError):
            recommendation["reasoning"] = f"(parse error) raw: {raw_output[:200]}"

        self._last_recommendation = recommendation
        self._recommendation_history.append({
            "turn_idx": turn_idx,
            **recommendation,
        })
        return recommendation

    def recommend(
        self,
        meal_base: str,
        alignment_score: float,
        alignment_reasoning: str,
        instruction: str = "",
        user_preferences: str = "",
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
