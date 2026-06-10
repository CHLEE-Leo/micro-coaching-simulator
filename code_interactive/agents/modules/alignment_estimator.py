"""Portable micro-coaching agent package module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

from ..json_output import JSONOutputError, load_json_object
from ..prompts.roles.alignment_estimator import (
    ALIGNMENT_INPUT_TEMPLATE,
    build_alignment_system_prompt,
)
from ..openai_client import generate_response

if TYPE_CHECKING:
    from ..agent_config import AgentConfig
    from ..memory.conversation_memory import SharedConversationHistory


# ------------------------------------------------------------------------------
# Data directory path
# ------------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "additional"


# ------------------------------------------------------------------------------
# Static resource loader
# ------------------------------------------------------------------------------

_GOAL_DEF_CACHE:    Optional[Dict]       = None
_WORKFLOW_CACHE:    Optional[List[Dict]] = None
_OUTPUT_FMT_CACHE:  Dict[str, str]       = {}  # Cache by key


def _load_goal_definitions() -> Dict:
    global _GOAL_DEF_CACHE
    if _GOAL_DEF_CACHE is None:
        path = _DATA_DIR / "goal_def_v2.json"
        with open(path, "r", encoding="utf-8") as f:
            _GOAL_DEF_CACHE = json.load(f)
    return _GOAL_DEF_CACHE


def _load_expert_workflows() -> List[Dict]:
    global _WORKFLOW_CACHE
    if _WORKFLOW_CACHE is None:
        path = _DATA_DIR / "expert_workflow.json"
        with open(path, "r", encoding="utf-8") as f:
            _WORKFLOW_CACHE = json.load(f)
    return _WORKFLOW_CACHE


def _load_output_format(fmt: str = "binary") -> str:
    """_load_output_format helper for the portable micro-coaching agent package."""
    global _OUTPUT_FMT_CACHE
    if fmt not in _OUTPUT_FMT_CACHE:
        filename = f"output_format_inst_{fmt}.txt"
        path = _DATA_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"output format file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            _OUTPUT_FMT_CACHE[fmt] = f.read().strip()
    return _OUTPUT_FMT_CACHE[fmt]


# ------------------------------------------------------------------------------
# Internal note
# ------------------------------------------------------------------------------

_QUALITATIVE_GOALS = {"lean_protein", "drink_water"}


def _get_workflow_text(nutrition_goal: str) -> str:
    """_get_workflow_text helper for the portable micro-coaching agent package."""
    category = "qualitative" if nutrition_goal in _QUALITATIVE_GOALS else "quantitative"
    for wf in _load_expert_workflows():
        if wf.get("goal_category") == category:
            return "\n".join(wf.get("expert_workflow", []))
    return ""


def _get_goal_spec(nutrition_goal: str) -> Dict:
    return _load_goal_definitions().get(nutrition_goal, {})


# ------------------------------------------------------------------------------
# AlignmentEstimator
# ------------------------------------------------------------------------------

class AlignmentEstimator:
    """AlignmentEstimator component for the portable micro-coaching agent package."""

    def __init__(
        self,
        model,
        nutrition_goal: str,
        config: "AgentConfig",
    ):
        self.model          = model
        self.nutrition_goal = nutrition_goal
        self.config         = config

        # 1
        goal_spec = _get_goal_spec(nutrition_goal)

        # config scaffold
        # -> build_alignment_system_prompt
        self._goal_definition = (
            goal_spec.get("definition", "") if config.alignment_use_goal_def else ""
        )
        self._workflow_text = (
            _get_workflow_text(nutrition_goal) if config.alignment_use_workflow else ""
        )
        self._output_fmt = _load_output_format(config.alignment_output_format)

        # System prompt
        self._system_prompt = build_alignment_system_prompt(
            nutrition_goal     = nutrition_goal,
            goal_definition    = self._goal_definition,
            workflow_text      = self._workflow_text,
            output_format_inst = self._output_fmt,
        )

        # Internal note
        # turn_idx int aligned bool score float raw_output str
        self._judgment_history: List[Dict] = []
        self._last_aligned: Optional[bool]  = None
        self._last_score:   Optional[float] = None
        self._last_reasoning: Optional[str]  = None

    # Public interface

    def get_messages(
        self,
        history: "SharedConversationHistory",
    ) -> List[Dict[str, str]]:
        """get_messages helper for the portable micro-coaching agent package."""
        transcript = self._build_transcript(history)

        # Internal note
        if self._last_score is not None:
            prev_score_context = (
                f"\n[previous alignment score]\n"
                f"The alignment score from the previous turn was {self._last_score:.2f}.\n"
                f"In your reasoning, you MUST explain why the score changed, decreased, increased, "
                f"or stayed the same compared to this previous score.\n"
            )
        else:
            prev_score_context = ""  # Internal note

        user_content = ALIGNMENT_INPUT_TEMPLATE.safe_substitute(
            transcript            = transcript,
            nutrition_goal_display = self.nutrition_goal.replace("_", " "),
            prev_score_context    = prev_score_context,
        )
        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user",   "content": user_content},
        ]

    def apply_judgment(self, raw_output: str, turn_idx: int) -> bool:
        """apply_judgment helper for the portable micro-coaching agent package."""
        score = self._parse_answer(raw_output)

        if score is None:
            # -> not aligned
            score   = 0.0
            aligned = False
        elif self.config.alignment_output_format == "binary":
            aligned = (score == 1.0)
        else:
            # Normalization
            aligned = (score >= self.config.alignment_threshold)

        self._last_aligned = aligned
        self._last_score   = score
        self._judgment_history.append({
            "turn_idx"   : turn_idx,
            "aligned"    : aligned,
            "score"      : score,
            "reasoning"  : self._last_reasoning,
            "raw_output" : raw_output.strip(),
        })
        return aligned

    def should_evaluate(self, turn_idx: int) -> bool:
        """should_evaluate helper for the portable micro-coaching agent package."""
        return turn_idx >= self.config.alignment_min_turn

    # Internal note

    def evaluate(
        self,
        history: "SharedConversationHistory",
        turn_idx: int,
    ) -> bool:
        """evaluate helper for the portable micro-coaching agent package."""
        if not self.should_evaluate(turn_idx):
            return False

        messages = self.get_messages(history)
        raw = generate_response(
            self.model,
            messages,
            sampling="greedy",          # greedy
            max_new_tokens=self.config.alignment_max_new_tokens,
            stop_at_newline=False,       # JSON
        )
        return self.apply_judgment(raw, turn_idx)

    # Properties

    @property
    def is_aligned(self) -> Optional[bool]:
        """is_aligned helper for the portable micro-coaching agent package."""
        return self._last_aligned
    @property
    def last_score(self) -> Optional[float]:
        """last_score helper for the portable micro-coaching agent package."""
        return self._last_score
    @property
    def last_reasoning(self) -> Optional[str]:
        """last_reasoning helper for the portable micro-coaching agent package."""
        return self._last_reasoning
    @property
    def judgment_history(self) -> List[Dict]:
        """judgment_history helper for the portable micro-coaching agent package."""
        return list(self._judgment_history)

    # Internal note

    @staticmethod
    def _build_transcript(history: "SharedConversationHistory") -> str:
        """_build_transcript helper for the portable micro-coaching agent package."""
        return history.to_alignment_context()

    def _parse_answer(self, raw: str) -> Optional[float]:
        """_parse_answer helper for the portable micro-coaching agent package."""
        import re

        # JSON
        answer_str = ""
        try:
            data = load_json_object(raw)
            answer_str = str(data.get("answer", "")).strip()
            # reasoning
            self._last_reasoning = str(data.get("reasoning", "")).strip() or None
        except (JSONOutputError, AttributeError, ValueError):
            # fallback
            m = re.search(r'"answer"\s*:\s*"([^"]+)"', raw)
            if m:
                answer_str = m.group(1).strip()
            # reasoning fallback
            m_r = re.search(r'"reasoning"\s*:\s*"([^"]+)"', raw)
            self._last_reasoning = m_r.group(1).strip() if m_r else None

        if not answer_str:
            return None

        fmt = self.config.alignment_output_format
        try:
            if fmt == "binary":
                if answer_str == "1":
                    return 1.0
                elif answer_str == "0":
                    return 0.0
                return None
            elif fmt == "0-1":
                val = float(answer_str)
                return val if 0.0 <= val <= 1.0 else None
            elif fmt == "0-100":
                val = float(answer_str)
                return (val / 100.0) if 0.0 <= val <= 100.0 else None
        except (ValueError, TypeError):
            return None

        return None
