"""Adapters between product-app chat rows and simulator conversation memory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .contracts import ChatMessage, CoachingState
from .opening import build_opening_message

from .memory.conversation_memory import SharedConversationHistory


@dataclass(frozen=True)
class AdaptedHistory:
    history: SharedConversationHistory
    turn_idx: int
    opening_used: str | None = None


def build_shared_history(
    messages: Sequence[ChatMessage],
    current_user_message: str,
    *,
    context_window: int,
    state: CoachingState | None = None,
    opening_message: str | None = None,
    use_opening_fallback: bool = True,
) -> AdaptedHistory:
    """Convert flat chat messages into ``SharedConversationHistory``.

    Product apps persist messages as a flat stream.  The simulator expects
    coach/user turn pairs.  Consecutive assistant messages are folded into one
    coach turn separated by blank lines, preserving the multi-bubble semantics
    while keeping the existing simulator memory shape.
    """

    history = SharedConversationHistory(context_window=context_window)
    if state is not None:
        history.update_meal_base(state.meal_base)
        history.update_tracker_state(state.tracker_state)
        history.update_context_base(state.context_base)

    assistant_buffer: list[str] = []
    turns: list[tuple[str, str]] = []

    def flush_with_user(user_text: str) -> None:
        coach_text = "\n\n".join(assistant_buffer).strip()
        assistant_buffer.clear()
        if not coach_text and not turns and use_opening_fallback:
            coach_text = opening_message or build_opening_message()
        turns.append((coach_text, user_text.strip()))

    for message in messages:
        content = message.content.strip()
        if not content:
            continue
        if message.role == "assistant":
            assistant_buffer.append(content)
        elif message.role == "user":
            flush_with_user(content)

    flush_with_user(current_user_message)

    for idx, (coach_text, user_text) in enumerate(turns):
        history.add_turn(idx, coach_text, user_text)

    opening_used = turns[0][0] if turns and use_opening_fallback else None
    return AdaptedHistory(history=history, turn_idx=max(0, len(turns) - 1), opening_used=opening_used)
