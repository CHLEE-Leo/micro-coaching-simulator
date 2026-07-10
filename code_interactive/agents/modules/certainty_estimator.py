"""Portable micro-coaching agent package module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from ..json_output import JSONOutputError, load_json_object
from ..prompts.roles.certainty_estimator import (
    CERTAINTY_INPUT_TEMPLATE,
    CERTAINTY_SYSTEM_PROMPT,
)

if TYPE_CHECKING:
    from ..agent_config import AgentConfig
    from ..memory.conversation_memory import SharedConversationHistory

# ------------------------------------------------------------------------------
# Default certainty threshold
# ------------------------------------------------------------------------------
CERTAINTY_THRESHOLD = 0.85

class CertaintyEstimator:
    """CertaintyEstimator component for the portable micro-coaching agent package."""

    def __init__(self, nutrition_goal: str, config: "AgentConfig"):
        self.nutrition_goal = nutrition_goal
        self.config = config
        self.last_reasoning: Optional[str] = None
        self.last_score: Optional[float] = None

    # ----------------------------------------------------------------------
    # Message assembly
    # ----------------------------------------------------------------------
    def get_messages(
        self,
        history: "SharedConversationHistory",
    ) -> List[Dict[str, str]]:
        """get_messages helper for the portable micro-coaching agent package."""
        transcript = history.to_certainty_context()

        # Internal note
        if self.last_score is not None:
            prev_score_context = (
                f"\n[previous certainty score]\n"
                f"The certainty score from the previous turn was {self.last_score:.2f}.\n"
                f"In your reasoning, you MUST explain why the score changed, decreased, increased, "
                f"or stayed the same compared to this previous score.\n"
            )
        else:
            prev_score_context = ""  # Internal note

        system = CERTAINTY_SYSTEM_PROMPT.format(nutrition_goal=self.nutrition_goal)
        user = CERTAINTY_INPUT_TEMPLATE.safe_substitute(
            transcript=transcript, prev_score_context=prev_score_context,
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    # ----------------------------------------------------------------------
    # Internal note
    # ----------------------------------------------------------------------
    def parse_output(self, raw_output: str) -> Tuple[str, float]:
        """parse_output helper for the portable micro-coaching agent package."""
        reasoning = ""
        score = 0.0

        # Error handling
        if raw_output.strip().startswith("[API_ERROR"):
            fallback = self.last_score if self.last_score is not None else 0.0
            reasoning = f"(API error - keeping previous score {fallback:.2f}) {raw_output[:150]}"
            print(f"[CertaintyEstimator] API error detected, preserving score={fallback:.2f}")
            self.last_reasoning = reasoning
            return reasoning, fallback

        try:
            data = load_json_object(raw_output)
            reasoning = str(data.get("reasoning", ""))
            raw_score = float(data.get("certainty_score", data.get("score", 0.0)))
            score = max(0.0, min(1.0, raw_score))  # clamp to [0, 1]
        except (JSONOutputError, ValueError, TypeError):
            fallback = self.last_score if self.last_score is not None else 0.0
            reasoning = f"(parse error - keeping previous score {fallback:.2f}) raw: {raw_output[:150]}"
            print(f"[CertaintyEstimator] Parse error, preserving score={fallback:.2f}")
            self.last_reasoning = reasoning
            return reasoning, fallback

        self.last_reasoning = reasoning
        self.last_score = score
        return reasoning, score

    # ----------------------------------------------------------------------
    # Convenience method
    # ----------------------------------------------------------------------
    def estimate(
        self,
        history: "SharedConversationHistory",
        generate_fn=None,
        llm=None,
        config=None,
    ) -> Tuple[str, float]:
        """estimate helper for the portable micro-coaching agent package."""
        if generate_fn is None:
            from ..openai_client import generate_response
            generate_fn = generate_response

        msgs = self.get_messages(history)
        raw = generate_fn(
            llm,
            msgs,
            max_new_tokens=config.certainty_max_new_tokens if config else 200,
            sampling="greedy",
        )
        return self.parse_output(raw)
