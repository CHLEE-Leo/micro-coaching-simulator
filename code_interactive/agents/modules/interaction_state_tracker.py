"""Track dialogue-operation state for planning."""

from __future__ import annotations

from typing import Any, Dict, List

from ..json_output import JSONOutputError, load_json_object
from ..prompts.roles.interaction_state_tracker import (
    INTERACTION_STATE_FULL_SYSTEM_PROMPT,
    INTERACTION_STATE_INCREMENTAL_SYSTEM_PROMPT,
)


class InteractionStateTracker:
    """Maintain answered/open/rejected/unavailable interaction facts."""

    _FIELDS = (
        "answered_facts",
        "open_questions",
        "rejected_options",
        "unavailable_options",
        "safety_conflicted_options",
        "user_requested_conflicted_options",
        "candidate_options",
        "accepted_options",
        "meal_slots",
    )

    def get_messages(
        self,
        conversation_text: str,
        prev_interaction_state: str = "",
    ) -> List[Dict[str, str]]:
        """Build messages for full or incremental interaction-state tracking."""
        if prev_interaction_state and conversation_text:
            return [
                {"role": "system", "content": INTERACTION_STATE_INCREMENTAL_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Previous interaction_state:\n{prev_interaction_state}\n\n"
                        f"New conversation turns:\n{conversation_text}\n\n"
                        "Return the updated interaction_state JSON:"
                    ),
                },
            ]
        return [
            {"role": "system", "content": INTERACTION_STATE_FULL_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Conversation to analyze:\n{conversation_text}\n\n"
                    "Return the interaction_state JSON:"
                ),
            },
        ]

    def parse_output(self, raw_output: str, fallback: str = "") -> str:
        """Parse JSON output and format it as compact prompt-ready text."""
        try:
            data = load_json_object(raw_output)
        except (JSONOutputError, ValueError, TypeError, AttributeError):
            return fallback.strip()
        return self.format_state(data)

    @classmethod
    def format_state(cls, data: Dict[str, Any]) -> str:
        """Convert structured state into a stable, readable prompt block."""
        lines: List[str] = []
        labels = {
            "answered_facts": "Answered facts",
            "open_questions": "Open questions",
            "rejected_options": "Rejected options",
            "unavailable_options": "Unavailable options",
            "safety_conflicted_options": "Safety-conflicted options",
            "user_requested_conflicted_options": "User-requested conflicted options",
            "candidate_options": "Candidate options",
            "accepted_options": "Accepted options",
            "meal_slots": "Meal slots",
        }
        for field in cls._FIELDS:
            items = cls._as_list(data.get(field))
            if items:
                lines.append(f"{labels[field]}:")
                lines.extend(f"- {item}" for item in items)
        latest = str(data.get("latest_user_position", "") or "").strip()
        if latest:
            lines.append("Latest user position:")
            lines.append(f"- {latest}")
        active_issue = str(data.get("active_issue", "") or "").strip()
        if active_issue:
            lines.append("Active issue:")
            lines.append(f"- {active_issue}")
        return "\n".join(lines).strip()

    @classmethod
    def parse_formatted_state(cls, text: str) -> Dict[str, Any]:
        """Parse the prompt-ready state text back into structured sections."""
        labels = {
            "Answered facts": "answered_facts",
            "Open questions": "open_questions",
            "Rejected options": "rejected_options",
            "Unavailable options": "unavailable_options",
            "Safety-conflicted options": "safety_conflicted_options",
            "User-requested conflicted options": "user_requested_conflicted_options",
            "Candidate options": "candidate_options",
            "Accepted options": "accepted_options",
            "Meal slots": "meal_slots",
            "Latest user position": "latest_user_position",
            "Active issue": "active_issue",
        }
        data: Dict[str, Any] = {field: [] for field in cls._FIELDS}
        data["latest_user_position"] = ""
        data["active_issue"] = ""
        current_key = ""
        for line in (text or "").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.endswith(":"):
                current_key = labels.get(stripped.rstrip(":"), "")
                continue
            if not current_key or not stripped.startswith("-"):
                continue
            value = stripped.lstrip("- ").strip()
            if not value:
                continue
            if current_key in ("latest_user_position", "active_issue"):
                data[current_key] = value
            else:
                data.setdefault(current_key, []).append(value)
        return data

    @staticmethod
    def _as_list(value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        cleaned: List[str] = []
        for item in value:
            text = str(item or "").strip()
            if text and text not in cleaned:
                cleaned.append(text)
        return cleaned
