"""Portable multi-agent chatbot engine.

This package is the import boundary for host applications.  The FastAPI app in
``code_interactive`` can keep using ``SessionManager`` for its UI workflow, while
external apps can call ``ConversationEngine`` through these contracts.
"""

from .contracts import (
    AssistantReply,
    ChatMessage,
    CoachingState,
    CoachingTurnRequest,
    CoachingTurnResult,
    MealContext,
    UserProfileContext,
)
from .agent_config import AgentConfig, SUPPORTED_GOALS
from .engine import ConversationEngine
from .opening import build_opening_message

__all__ = [
    "AssistantReply",
    "AgentConfig",
    "ChatMessage",
    "CoachingState",
    "CoachingTurnRequest",
    "CoachingTurnResult",
    "MealContext",
    "ConversationEngine",
    "SUPPORTED_GOALS",
    "UserProfileContext",
    "build_opening_message",
]
