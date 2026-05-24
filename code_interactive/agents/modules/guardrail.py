"""Portable micro-coaching agent package module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

from ..json_output import JSONOutputError, load_json_object
from ..prompts.roles.guardrail import (
    INPUT_GUARD_SYSTEM_PROMPT,
    OUTPUT_GUARD_SYSTEM_PROMPT,
)

if TYPE_CHECKING:
    from ..agent_config import AgentConfig


class Guardrail:
    """Guardrail component for the portable micro-coaching agent package."""

    def __init__(self, config: "AgentConfig"):
        self.config = config

    # -- Input Guard -------------------------------------------------------

    def get_input_guard_messages(
        self, user_input: str, dialog_context: str = "",
    ) -> List[Dict[str, str]]:
        """get_input_guard_messages helper for the portable micro-coaching agent package."""
        content = user_input
        if dialog_context:
            content = (
                f"[Recent conversation context]\n{dialog_context}\n\n"
                f"[Current user message to evaluate]\n{user_input}"
            )
        return [
            {"role": "system", "content": INPUT_GUARD_SYSTEM_PROMPT},
            {"role": "user",   "content": content},
        ]

    def parse_input_guard(self, raw_output: str) -> Dict:
        """parse_input_guard helper for the portable micro-coaching agent package."""
        try:
            data = load_json_object(raw_output)

            # Multi-signal schema (new)
            action = str(data.get("action", "")).lower()
            if action in ("pass", "block", "crisis"):
                flags = data.get("flags", [])
                if not isinstance(flags, list):
                    flags = []
                flags = [str(f) for f in flags]
                return {
                    "action":  action,
                    "passed":  action == "pass",
                    "flags":   flags,
                    "reason":  str(data.get("reason", "")),
                    "message": str(data.get("message", "")),
                }

            # Legacy schema fallback ({"passed": true/false})
            passed = bool(data.get("passed", True))
            return {
                "action":  "pass" if passed else "block",
                "passed":  passed,
                "flags":   [],
                "reason":  str(data.get("reason", "")),
                "message": str(data.get("message", "")),
            }
        except (JSONOutputError, ValueError, TypeError):
            # false positive false negative
            return {"action": "pass", "passed": True, "flags": [], "reason": "", "message": ""}

    # -- Output Guard ------------------------------------------------------

    def get_output_guard_messages(self, orchestrator_response: str) -> List[Dict[str, str]]:
        """get_output_guard_messages helper for the portable micro-coaching agent package."""
        return [
            {"role": "system", "content": OUTPUT_GUARD_SYSTEM_PROMPT},
            {"role": "user",   "content": orchestrator_response},
        ]

    def parse_output_guard(self, raw_output: str) -> Dict:
        """parse_output_guard helper for the portable micro-coaching agent package."""
        try:
            data = load_json_object(raw_output)
            return {
                "passed": bool(data.get("passed", True)),
                "reason": str(data.get("reason", "")),
            }
        except (JSONOutputError, ValueError, TypeError):
            return {"passed": True, "reason": ""}
