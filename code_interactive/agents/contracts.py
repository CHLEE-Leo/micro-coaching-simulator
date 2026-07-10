"""Stable request/response contracts for the portable coaching engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

Role = Literal["user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    """A minimal product-app chat message.

    Product apps can map their database rows into this structure without
    importing simulator internals.  Consecutive assistant messages are treated
    as one multi-bubble coach turn by the history adapter.
    """

    role: Role
    content: str


@dataclass(frozen=True)
class UserProfileContext:
    """User facts that can personalize the opening and coaching prompts."""

    name: str | None = None
    nutritional_goals: Sequence[str] = ()
    preferences: Sequence[str] = ()
    allergies: Sequence[str] = ()
    activity_level: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MealContext:
    """A recently logged meal supplied by the host app."""

    name: str
    description: str | None = None
    eaten_at: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CoachingState:
    """Optional persisted simulator state.

    The first product integration can omit this and let the engine rebuild
    state from chat history.  Later integrations can persist these fields for
    lower latency and more faithful phase continuity.
    """

    phase: str = "exploration"
    status: str = "active"
    meal_base: str = ""
    tracker_state: str = ""
    context_base: str = ""
    interaction_state: str = ""
    user_preferences: str = ""
    recommendation_history: Sequence[Mapping[str, Any]] = ()
    consecutive_qa_count: int = 0
    stall_count: int = 0
    recommendation_rejection_count: int = 0
    last_intent_summary: str = ""
    last_user_intent: str = "passive"
    last_alignment_score: float | None = None
    last_alignment_reasoning: str | None = None
    last_certainty_score: float | None = None
    last_certainty_reasoning: str | None = None
    safety_clarification_counts: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class CoachingTurnRequest:
    """Input for a single user-message-driven coaching turn."""

    current_message: str
    history: Sequence[ChatMessage] = ()
    profile: UserProfileContext | None = None
    recent_meals: Sequence[MealContext] = ()
    state: CoachingState | None = None
    nutrition_goal: str = "lean_protein"
    meal_type: str = "meal"
    opening_message: str | None = None
    enable_opening_fallback: bool = True
    enable_guardrail: bool = True
    enable_context_tracking: bool = True
    enable_alignment: bool = False
    enable_certainty: bool = False


@dataclass(frozen=True)
class AssistantReply:
    """One assistant bubble returned by the engine."""

    content: str
    kind: str = "reply"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CoachingTurnResult:
    """Result of one coaching turn.

    ``assistant_messages`` is the canonical multi-bubble output.  The
    ``primary_message`` property exists for host apps that still need a
    backwards-compatible single assistant message.
    """

    assistant_messages: Sequence[AssistantReply]
    state: CoachingState
    status: str = "active"
    terminated_by: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def primary_message(self) -> AssistantReply | None:
        if not self.assistant_messages:
            return None
        return self.assistant_messages[-1]
