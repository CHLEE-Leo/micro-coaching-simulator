"""Opening-message policy for product chat integrations."""

from __future__ import annotations

from .contracts import UserProfileContext


def build_opening_message(profile: UserProfileContext | None = None) -> str:
    """Return the assistant's first visible message for an empty chat."""

    name = (profile.name or "").strip() if profile else ""
    greeting = f"Hi, {name}." if name else "Hi."
    return f"{greeting} How can I help you with your meal today?"
