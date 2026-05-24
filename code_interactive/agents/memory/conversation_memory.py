"""Conversation memory primitives shared by the coaching agents.

This module owns only memory concerns:
- store the coach/user turn transcript,
- store extracted meal and context summaries,
- convert those records into prompt-ready text/messages for each agent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


class ConversationBuffer:
    """Append-only utterance buffer for one agent role.

    Some modules keep their own local phrasing history in addition to the shared
    turn history.  This small buffer formats that local history for prompts.
    """

    def __init__(self, role: str):
        self.role = role.strip() or "agent"
        self._utterances: List[str] = []

    def add(self, utterance: str) -> None:
        """Store a non-empty utterance."""
        text = utterance.strip()
        if text:
            self._utterances.append(text)

    def get_all(self) -> List[str]:
        """Return all stored utterances without exposing internal state."""
        return list(self._utterances)

    def get_recent(self, n: int) -> List[str]:
        """Return the most recent ``n`` utterances."""
        return self._utterances[-n:] if n > 0 else []

    def __len__(self) -> int:
        return len(self._utterances)

    def to_prompt_text(self, header: str | None = None) -> str:
        """Format the buffer as a numbered prompt block."""
        if not self._utterances:
            return "(none yet)"

        label = header or f"Your previous {self.role} utterances"
        lines = [f"{label}:"]
        for i, utt in enumerate(self._utterances, start=1):
            lines.append(f"  {i}. {utt}")
        return "\n".join(lines)

    def reset(self) -> None:
        """Clear all stored utterances."""
        self._utterances.clear()


@dataclass
class _ConversationTurn:
    """One paired coach/user turn in chronological order."""

    turn_idx: int
    coach_utterance: str
    user_utterance: str


class SharedConversationHistory:
    """Shared session memory for all coaching agents.

    The transcript is the source of truth for recent dialogue.  The extracted
    memory fields are derived summaries produced by tracker modules and reused
    by estimator, recommender, and response modules.
    """

    # Simulated users can emit this token to end an automated dialogue.
    TERMINATION_TOKEN = "[END]"

    def __init__(self, context_window: int = 5):
        """Create an empty memory with a bounded recent-turn window."""
        self.context_window = context_window
        self._turns: List[_ConversationTurn] = []

        # Published meal facts from MealTracker. Used by AlignmentEstimator and
        # MealRecommender as the cleanest view of what the user plans to eat.
        self.meal_base: str = ""

        # MealTracker's internal notes, including tentative/rejected details.
        # This is useful for debugging and continuity, but is not the primary
        # alignment input.
        self.tracker_state: str = ""

        # Cross-turn user preferences, constraints, and conversation context
        # summarized by ContextTracker.
        self.context_base: str = ""

    def add_turn(
        self,
        turn_idx: int,
        coach_utterance: str,
        user_utterance: str = "",
    ) -> None:
        """Append a coach turn and its optional user reply."""
        self._turns.append(
            _ConversationTurn(
                turn_idx=turn_idx,
                coach_utterance=coach_utterance.strip(),
                user_utterance=user_utterance.strip(),
            )
        )

    def update_last_user_utterance(self, user_utterance: str) -> None:
        """Attach or replace the latest user reply."""
        if self._turns:
            self._turns[-1].user_utterance = user_utterance.strip()

    def _windowed_turns(self) -> List[_ConversationTurn]:
        """Return turns visible to short-context prompt builders."""
        if self.context_window <= 0:
            return list(self._turns)
        return self._turns[-self.context_window:]

    def build_messages(
        self,
        perspective: str,
        system_prompt: str,
    ) -> List[Dict[str, str]]:
        """Build chat messages for coach or simulated-user generation.

        ``perspective="coach"`` maps coach text to assistant messages and user
        text to user messages.  ``perspective="user"`` reverses that mapping so
        the simulated user can answer the coach.
        """
        if perspective not in ("coach", "user"):
            raise ValueError("perspective must be 'coach' or 'user'.")

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]

        for turn in self._windowed_turns():
            if perspective == "coach":
                # Turn 0 is the opening coach question already shown by the UI.
                # For the next coach response, only the user's answer to that
                # opening question should appear in the prompt history.
                if turn.turn_idx == 0:
                    if turn.user_utterance:
                        messages.append({"role": "user", "content": turn.user_utterance})
                else:
                    messages.append({"role": "assistant", "content": turn.coach_utterance})
                    if turn.user_utterance:
                        messages.append({"role": "user", "content": turn.user_utterance})
            else:
                # For simulated users, the coach is the incoming user message
                # and previous user replies are assistant messages.
                messages.append({"role": "user", "content": turn.coach_utterance})
                if turn.user_utterance:
                    messages.append({"role": "assistant", "content": turn.user_utterance})

        return messages

    def update_meal_base(self, new_meal_base: str) -> None:
        """Replace the published meal summary."""
        self.meal_base = new_meal_base.strip()

    def update_tracker_state(self, new_tracker_state: str) -> None:
        """Replace the internal meal-tracking notes."""
        self.tracker_state = new_tracker_state.strip()

    def update_context_base(self, new_context_base: str) -> None:
        """Replace the context summary when the tracker produced content."""
        stripped_context = new_context_base.strip()
        if stripped_context:
            self.context_base = stripped_context

    def to_plain_text(self) -> str:
        """Return the full transcript as readable text."""
        lines: List[str] = []
        for turn in self._turns:
            lines.append(f"Coach: {turn.coach_utterance}")
            if turn.user_utterance:
                lines.append(f"User: {turn.user_utterance}")
        return "\n".join(lines)

    def to_plain_text_from(self, from_turn_idx: int = 0) -> str:
        """Return transcript text from ``from_turn_idx`` onward."""
        lines: List[str] = []
        for turn in self._turns:
            if turn.turn_idx >= from_turn_idx:
                lines.append(f"Coach: {turn.coach_utterance}")
                if turn.user_utterance:
                    lines.append(f"User: {turn.user_utterance}")
        return "\n".join(lines)

    def get_all_coach_questions(self) -> List[str]:
        """Return all non-empty coach utterances."""
        return [t.coach_utterance for t in self._turns if t.coach_utterance.strip()]

    def to_dict_list(self) -> List[Dict]:
        """Serialize turns for UI/debug payloads."""
        return [
            {
                "turn_idx": t.turn_idx,
                "coach_utterance": t.coach_utterance,
                "user_utterance": t.user_utterance,
            }
            for t in self._turns
        ]

    def to_recent_turns_text(self, n: int = 0) -> str:
        """Return the recent transcript window as prompt text."""
        window_size = n or self.context_window
        turns = self._turns[-window_size:] if window_size > 0 else self._turns
        parts: List[str] = []
        for turn in turns:
            parts.append(f"Coach: {turn.coach_utterance}")
            if turn.user_utterance:
                parts.append(f"User: {turn.user_utterance}")
        return "\n".join(parts).strip() or "(no conversation yet)"

    def to_alignment_context(self) -> str:
        """Return the best available meal evidence for alignment scoring."""
        # Prefer extracted meal facts over raw conversation because they remove
        # rejected/tentative details and keep the estimator focused.
        if self.meal_base:
            return self.meal_base

        # Before MealTracker publishes facts, fall back to recent raw turns.
        parts: List[str] = []
        for turn in self._windowed_turns():
            parts.append(f"Coach: {turn.coach_utterance}")
            if turn.user_utterance:
                parts.append(f"User: {turn.user_utterance}")

        result = "\n".join(parts).strip()
        return result if result else "(no conversation yet)"

    def to_certainty_context(self) -> str:
        """Return conversation plus extracted meal facts for certainty scoring."""
        conv_lines: List[str] = []
        for turn in self._windowed_turns():
            conv_lines.append(f"Coach: {turn.coach_utterance}")
            if turn.user_utterance:
                conv_lines.append(f"User: {turn.user_utterance}")
        conversation = "\n".join(conv_lines).strip() or "(no conversation yet)"

        parts: List[str] = [f"[Conversation]\n{conversation}"]

        if self.meal_base:
            parts.append(
                f"[Extracted meal information so far]\n{self.meal_base}"
            )

        return "\n\n".join(parts)

    def is_terminated(self) -> bool:
        """Return whether the latest user reply contains the stop token."""
        if not self._turns:
            return False
        last_user = self._turns[-1].user_utterance
        return self.TERMINATION_TOKEN.lower() in last_user.lower()

    def __len__(self) -> int:
        return len(self._turns)

    def current_turn_idx(self) -> int:
        """Return the next zero-based turn index."""
        return len(self._turns)
