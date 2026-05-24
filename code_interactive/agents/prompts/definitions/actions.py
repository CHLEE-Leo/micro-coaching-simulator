"""Shared action definitions and action-use instructions."""

ACTION_DEFINITIONS = """\
=== ACTION DEFINITIONS ===

- INQUIRE - Ask one useful question.
- ASSESS - Generate a meal assessment from the known meal information.
- RECOMMEND - Suggest one concrete meal improvement or adjusted alternative.
- RESPOND - Answer the user's question, suggestion, or clarification request \
briefly, then continue the coaching flow.
- CLOSE - Generate a warm final motivational closing message.
- TERMINATE - End the dialogue immediately when continuation is inappropriate, \
unsafe, impossible, or explicitly unwanted.\
"""

ACTION_INSTRUCTIONS = """\
=== ACTION INSTRUCTIONS ===

- INQUIRE: The question target should emerge from the current dialogue phase, \
dialogue state, meal/context state, user intent, and the user's latest message. \
Do not force a fixed subcategory of inquiry.
- ASSESS: Use when EXPLORATION has gathered enough detail to compare the meal \
with the goal.
- CLOSE: After CLOSE, the dialogue status becomes terminated.
- TERMINATE: Use when continuation is inappropriate, unsafe, impossible, or \
explicitly unwanted.\
"""
