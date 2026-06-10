"""Portable micro-coaching agent package module."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Dict, List

from ..prompts.roles.meal_tracker import (
    TRACKER_FULL_SYSTEM_PROMPT,
    TRACKER_INCREMENTAL_SYSTEM_PROMPT,
)
from ..openai_client import generate_response

if TYPE_CHECKING:
    from ..agent_config import AgentConfig


_TRACKING_STATE_HEADER = "[Tracking State]"
_PUBLISHED_MEAL_BASE_HEADER = "[Published Meal Base]"
_MEAL_BASE_FALLBACK = (
  "- Food items: not yet mentioned\n"
  "- Ingredients: not yet mentioned\n"
  "- Preparation methods: not yet mentioned\n"
  "- Portions/amounts: not yet mentioned\n"
  "- Beverages: none mentioned\n"
  "- Additional notes: not yet mentioned"
)


# ------------------------------------------------------------------------------
# MealTrackerModel
# ------------------------------------------------------------------------------

class MealTrackerModel:
    """MealTrackerModel component for the portable micro-coaching agent package."""

    def __init__(self, model, config: "AgentConfig"):
        self.model  = model
        self.config = config

    # Public interface

    def get_messages(
        self,
        conversation_text: str,
      prev_tracker_state: str = "",
    ) -> List[Dict[str, str]]:
        """get_messages helper for the portable micro-coaching agent package."""
        if prev_tracker_state and conversation_text:
          # tracker_state + -> tracker_state + meal_base
            return [
                {"role": "system", "content": TRACKER_INCREMENTAL_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                f"Previous tracking state:\n{prev_tracker_state}\n\n"
                        f"New conversation turns since the previous fact sheet:\n\n"
                        f"{conversation_text}\n\n"
                "Now write the updated tracking state and published meal base:"
                    ),
                },
            ]
        else:
          # -> tracker_state + published meal_base
            return [
                {"role": "system", "content": TRACKER_FULL_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Conversation to extract meal information from:\n\n"
                        f"{conversation_text}\n\n"
                "Now write the tracking state and published meal base:"
                    ),
                },
            ]

    def extract(
        self,
        conversation_text: str,
        prev_tracker_state: str = "",
    ) -> str:
        """extract helper for the portable micro-coaching agent package."""
        messages = self.get_messages(conversation_text, prev_tracker_state)
        raw_output = generate_response(
            self.model,
            messages,
            sampling="greedy",
            max_new_tokens=self.config.summarize_max_new_tokens,
            stop_at_newline=False,
        )
        return self.parse_tracking_output(raw_output)["meal_base"]

    def parse_tracking_output(self, raw_output: str) -> Dict[str, str]:
        """Split a tracker response into internal tracking state and published meal_base."""
        text = raw_output.strip()
        if not text:
            return {
                "tracker_state": f"{_TRACKING_STATE_HEADER}\n- Confirmed food items: none\n- Tentative food items: none\n- Rejected food items: none\n- Decision context: none",
                "meal_base": _MEAL_BASE_FALLBACK,
            }

        tracker_state = self._extract_section(text, _TRACKING_STATE_HEADER, _PUBLISHED_MEAL_BASE_HEADER)
        meal_base = self._extract_section(text, _PUBLISHED_MEAL_BASE_HEADER)
        if not meal_base:
            meal_base = text
        return {
            "tracker_state": (tracker_state or text).strip(),
            "meal_base": meal_base.strip() or _MEAL_BASE_FALLBACK,
        }

    @staticmethod
    def _extract_section(text: str, header: str, next_header: str | None = None) -> str:
        pattern = re.escape(header) + r"\s*(.*?)"
        if next_header:
            pattern += r"(?=\n" + re.escape(next_header) + r"|\Z)"
        else:
            pattern += r"\Z"
        match = re.search(pattern, text, flags=re.DOTALL)
        return match.group(1).strip() if match else ""
