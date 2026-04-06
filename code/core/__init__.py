"""core package"""
from .memory     import ConversationBuffer, SharedConversationHistory

def __getattr__(name):
    """Lazy import for simulation to avoid circular dependency with models package."""
    if name in ("simulate_conversation", "simulate_conversations_batch", "set_seed"):
        from .simulation import simulate_conversation, simulate_conversations_batch, set_seed
        return {"simulate_conversation": simulate_conversation,
                "simulate_conversations_batch": simulate_conversations_batch,
                "set_seed": set_seed}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ConversationBuffer",
    "SharedConversationHistory",
    "simulate_conversation",
    "simulate_conversations_batch",
    "set_seed",
]
