"""Portable micro-coaching agent package module."""

from __future__ import annotations

from typing import Dict, List

from ..prompts.roles.context_tracker import (
    CONTEXT_FULL_SYSTEM_PROMPT,
    CONTEXT_INCREMENTAL_SYSTEM_PROMPT,
)


# ------------------------------------------------------------------------------
# ContextTracker
# ------------------------------------------------------------------------------

class ContextTracker:
    """ContextTracker component for the portable micro-coaching agent package."""

    def __init__(self):
        # User profile
        self._profile: Dict[str, any] = {
            "activity_level": "",
            "diet_preferences": [],
            "allergies": [],
            "health_concerns": [],
            "availability": [],
            "past_meals": [],
        }

    # ======================================================================
    # LLM context_base
    # ======================================================================

    def get_messages(
        self,
        conversation_text: str,
        prev_context_base: str = "",
    ) -> List[Dict[str, str]]:
        """get_messages helper for the portable micro-coaching agent package."""
        # LLM personal_context
        _profile_block = ""
        _pref = self.get_preferences_text()
        if _pref:
            _profile_block = f"\n[Known User Profile]\n{_pref}\n"

        if prev_context_base and conversation_text:
            # context_base + ->
            return [
                {"role": "system", "content": CONTEXT_INCREMENTAL_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Previous context_base:\n{prev_context_base}\n\n"
                        f"New conversation turns:\n\n"
                        f"{conversation_text}\n"
                        f"{_profile_block}\n"
                        "Now write the updated context_base with [Task Context], "
                        "[Personal Context], and [Environmental Context]:"
                    ),
                },
            ]
        else:
            # -> context_base
            return [
                {"role": "system", "content": CONTEXT_FULL_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Conversation to summarize:\n\n"
                        f"{conversation_text}\n"
                        f"{_profile_block}\n"
                        "Now write the context_base with [Task Context], "
                        "[Personal Context], and [Environmental Context]:"
                    ),
                },
            ]

    @staticmethod
    def parse_output(raw_output: str) -> str:
        """parse_output helper for the portable micro-coaching agent package."""
        return (raw_output or "").strip()

    # ======================================================================
    # Python LLM
    # ======================================================================

    @property
    def profile(self) -> Dict:
        """profile helper for the portable micro-coaching agent package."""
        return dict(self._profile)

    def get_preferences_text(self) -> str:
        """get_preferences_text helper for the portable micro-coaching agent package."""
        lines = []
        if self._profile["activity_level"]:
            lines.append("Activity Level: " + self._profile["activity_level"])
        if self._profile["diet_preferences"]:
            lines.append("Diet Preferences: " + ", ".join(self._profile["diet_preferences"]))
        if self._profile["allergies"]:
            lines.append("Allergies: " + ", ".join(self._profile["allergies"]))
        if self._profile["health_concerns"]:
            lines.append("Health Concerns: " + ", ".join(self._profile["health_concerns"]))
        if self._profile["availability"]:
            lines.append("Availability: " + ", ".join(self._profile["availability"]))
        return "\n".join(lines) if lines else ""

    def get_cross_session_summary(self) -> str:
        """get_cross_session_summary helper for the portable micro-coaching agent package."""
        if not self._profile["past_meals"]:
            return ""
        lines = []
        for i, meal in enumerate(self._profile["past_meals"], 1):
            meal_type = meal.get("meal_type", "meal")
            summary = meal.get("summary", "")
            lines.append(f"  {i}. {meal_type}: {summary}")
        return "Previous meals in this session:\n" + "\n".join(lines)

    # Internal note

    def update_preferences(self, items: List[str]) -> None:
        """update_preferences helper for the portable micro-coaching agent package."""
        for item in items:
            if item and item not in self._profile["diet_preferences"]:
                self._profile["diet_preferences"].append(item)

    def update_allergies(self, items: List[str]) -> None:
        """update_allergies helper for the portable micro-coaching agent package."""
        for item in items:
            if item and item not in self._profile["allergies"]:
                self._profile["allergies"].append(item)

    def update_restrictions(self, items: List[str]) -> None:
        """update_restrictions helper for the portable micro-coaching agent package."""
        for item in items:
            if item and item not in self._profile["health_concerns"]:
                self._profile["health_concerns"].append(item)

    def update_availability(self, items: List[str]) -> None:
        """update_availability helper for the portable micro-coaching agent package."""
        for item in items:
            if item and item not in self._profile["availability"]:
                self._profile["availability"].append(item)

    def add_past_meal(self, meal_type: str, summary: str, meal_base: str = "") -> None:
        """add_past_meal helper for the portable micro-coaching agent package."""
        self._profile["past_meals"].append({
            "meal_type": meal_type,
            "summary": summary,
            "meal_base": meal_base,
        })

    def set_profile_from_persona(
        self,
        activity_level: str | None = None,
        diet_preferences: List[str] | None = None,
        allergies: List[str] | None = None,
        health_concerns: List[str] | None = None,
    ) -> None:
        """set_profile_from_persona helper for the portable micro-coaching agent package."""
        if activity_level:
            self._profile["activity_level"] = activity_level
        if diet_preferences:
            self._profile["diet_preferences"] = list(diet_preferences)
        if allergies:
            self._profile["allergies"] = list(allergies)
        if health_concerns:
            self._profile["health_concerns"] = list(health_concerns)

    def reset(self) -> None:
        """reset helper for the portable micro-coaching agent package."""
        self._profile = {
            "activity_level": "",
            "diet_preferences": [],
            "allergies": [],
            "health_concerns": [],
            "availability": [],
            "past_meals": [],
        }
