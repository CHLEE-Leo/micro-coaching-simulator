"""Configuration for the portable micro-coaching agent package."""

from dataclasses import dataclass
from typing import Any, Literal


# ------------------------------------------------------------------------------
# Supported nutrition goals
# ------------------------------------------------------------------------------
SUPPORTED_GOALS = [
    "lean_protein",
    "half_fruits_vegetables",
    "one_fourth_carbs",
    "drink_water",
]


@dataclass
class AgentConfig:
    """Portable agent behavior settings.

    This config belongs to the agent package. Host apps should keep server,
    API-client, and model-routing settings outside this class.
    """

    goal: str = "lean_protein"

    # Prompt ablation flags
    coach_use_template_guidance: bool = True
    use_interaction_tracker: bool = True
    dialogue_planner_use_intents: bool = True
    dialogue_planner_use_state_scores: bool = True
    dialogue_planner_use_state_rationales: bool = True

    # Default text-generation options
    max_new_tokens: int = 150
    sampling: Literal["beam", "greedy", "sampling"] = "sampling"
    coach_sampling: Literal["beam", "greedy", "sampling"] = "greedy"

    # Module-specific generation limits
    alignment_min_turn: int = 0
    alignment_max_new_tokens: int = 300
    alignment_sampling: Literal["beam", "greedy", "sampling"] = "greedy"
    alignment_use_goal_def: bool = True
    alignment_use_workflow: bool = True
    alignment_output_format: Literal["binary", "0-1", "0-100"] = "binary"
    alignment_threshold: float = 0.5

    # Conversation state and tracking cadence
    max_turns: int = 15
    context_window: int = 10
    meal_track_every: int = 1
    summarize_every: int = 1
    summarize_max_new_tokens: int = 600 # LLM
    certainty_max_new_tokens: int = 220

    # Structured-output modules
    recommendation_max_new_tokens: int = 900
    planner_max_new_tokens: int = 260
    response_generator_max_new_tokens: int = 500
    guardrail_max_new_tokens: int = 200
    assessment_max_new_tokens: int = 500

    # Conversation exit thresholds
    stall_exit_turns: int = 3
    min_natural_end_turn: int = 3

    def generation_options(self, mode: str) -> dict[str, Any]:
        """Return LLM generation options for an agent module mode."""
        if mode == "alignment":
            return {
                "max_new_tokens": self.alignment_max_new_tokens,
                "sampling": self.alignment_sampling,
                "stop_at_newline": False,
            }
        if mode == "tracker":
            return {
                "max_new_tokens": self.summarize_max_new_tokens,
                "sampling": "greedy",
                "stop_at_newline": False,
            }
        if mode == "certainty":
            return {
                "max_new_tokens": self.certainty_max_new_tokens,
                "sampling": "greedy",
                "stop_at_newline": False,
            }
        if mode == "planner":
            return {
                "max_new_tokens": self.planner_max_new_tokens,
                "sampling": "greedy",
                "stop_at_newline": False,
            }
        if mode == "response_generator":
            return {
                "max_new_tokens": self.response_generator_max_new_tokens,
                "sampling": "greedy",
                "stop_at_newline": False,
            }
        if mode == "assessment":
            return {
                "max_new_tokens": self.assessment_max_new_tokens,
                "sampling": "greedy",
                "stop_at_newline": False,
            }
        if mode == "recommender":
            return {
                "max_new_tokens": self.recommendation_max_new_tokens,
                "sampling": "greedy",
                "stop_at_newline": False,
            }
        if mode == "guardrail":
            return {
                "max_new_tokens": self.guardrail_max_new_tokens,
                "sampling": "greedy",
                "stop_at_newline": False,
            }
        if mode == "coach":
            return {
                "max_new_tokens": self.max_new_tokens,
                "sampling": self.coach_sampling,
                "stop_at_newline": True,
            }
        return {
            "max_new_tokens": self.max_new_tokens,
            "sampling": self.sampling,
            "stop_at_newline": True,
        }
