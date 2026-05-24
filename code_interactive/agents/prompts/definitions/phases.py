"""Shared dialogue phase definitions and phase-use instructions."""

PHASE_DEFINITIONS = """\
=== PHASE DEFINITIONS ===

- EXPLORATION - The coach is learning the user's meal situation: food items, \
ingredients, preparation, portions, beverages, and relevant context.

- RECOMMENDATION - The coach is preparing or providing a concrete meal \
improvement recommendation.

- NEGOTIATION - The user has concerns about a recommendation, rejects a \
suggestion, or is exploring alternatives.

- MOTIVATIONAL_ENDING - The dialogue is ready for a warm final wrap-up after \
the meal has been assessed, a recommendation has been accepted, or the \
interaction should close respectfully.\
"""

PHASE_INSTRUCTIONS = """\
=== PHASE INSTRUCTIONS ===

Do not force a fixed sequence. Use the current meal state, context state, \
dialogue state, user intent, and recent conversation to decide which phase best \
describes the next coaching move.\
"""
