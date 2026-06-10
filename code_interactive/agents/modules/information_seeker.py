"""Portable micro-coaching agent package module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

from ..json_output import JSONOutputError, load_json_object
from ..memory.conversation_memory import ConversationBuffer, SharedConversationHistory
from ..prompts.roles.information_seeker import (
    INFORMATION_SEEKER_ACTION_GUIDELINES,
    INFORMATION_SEEKER_DEAD_END_BLOCK,
    INFORMATION_SEEKER_NATURAL_CLOSE_BLOCK,
    INFORMATION_SEEKER_PHASE_BLOCK,
    INFORMATION_SEEKER_PROFILE_BLOCK,
    INFORMATION_SEEKER_QUESTIONS_BLOCK,
    INFORMATION_SEEKER_ROUTER_BLOCK,
    INFORMATION_SEEKER_STALL_EXIT_BLOCK,
    INFORMATION_SEEKER_STRATEGY_BLOCK,
    INFORMATION_SEEKER_SUMMARY_BLOCK,
    INFORMATION_SEEKER_SYSTEM_PROMPT,
)
from ..openai_client import generate_response

if TYPE_CHECKING:
    from ..agent_config import AgentConfig


# ------------------------------------------------------------------------------
# Duplicate-question detection helper
# ------------------------------------------------------------------------------

_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "do", "does", "did", "will", "would", "could", "should", "can",
    "i", "you", "your", "my", "me", "it", "its", "we", "they",
    "of", "in", "on", "at", "to", "for", "with", "by", "from",
    "and", "or", "not", "no", "but", "if", "so", "as", "than",
    "that", "this", "what", "how", "much", "many", "some", "any",
    "have", "has", "had", "about",
})


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    import re
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9\s]', '', text.lower())).strip()


def _is_duplicate_question(new_q: str, already_asked: list, threshold: float = 0.85) -> bool:
    """Exact normalized match OR Jaccard word-overlap above threshold."""
    norm_new = _normalize(new_q)
    if not norm_new:
        return False
    for prev in already_asked:
        if _normalize(prev) == norm_new:
            return True
        words_new  = {w for w in norm_new.split() if w not in _STOPWORDS}
        words_prev = {w for w in _normalize(prev).split() if w not in _STOPWORDS}
        if not words_new or not words_prev:
            continue
        union = words_new | words_prev
        if len(words_new & words_prev) / len(union) >= threshold:
            return True
    return False


# ------------------------------------------------------------------------------
# InformationSeeker
# ------------------------------------------------------------------------------

class InformationSeeker:
    """InformationSeeker component for the portable micro-coaching agent package."""

    # turn=0
    INITIAL_QUESTION_TEMPLATE = "What are you thinking of having for {meal_type}?"

    def __init__(
        self,
        model,
        nutrition_goal: str,
        meal_type: str,
        config: "AgentConfig",
    ):
        self.model          = model
        self.nutrition_goal = nutrition_goal
        self.meal_type      = meal_type
        self.config         = config

        # Principle 2
        self.own_buffer = ConversationBuffer(role="coach")

    # Public interface

    def first_question(self) -> str:
        """first_question helper for the portable micro-coaching agent package."""
        q = self.INITIAL_QUESTION_TEMPLATE.format(meal_type=self.meal_type)
        self.own_buffer.add(q)
        return q

    def get_messages(
        self,
        shared_history: SharedConversationHistory,
        dead_end_topics: List[str] | None = None,
        stall_exit: bool = False,
        natural_close: bool = False,
        phase: str = "",
        user_preferences: str = "",
        instruction: str = "",
        **_ignored_options,
    ) -> List[Dict[str, str]]:
        """get_messages helper for the portable micro-coaching agent package."""
        prev_questions = shared_history.get_all_coach_questions()
        system_prompt = self._build_system_prompt(
            shared_history.context_base,
            prev_questions=prev_questions,
            dead_end_topics=dead_end_topics,
            stall_exit=stall_exit,
            natural_close=natural_close,
            phase=phase,
            user_preferences=user_preferences,
            instruction=instruction,
        )
        return shared_history.build_messages(
            perspective="coach",
            system_prompt=system_prompt,
        )

    def ask(
        self,
        shared_history: SharedConversationHistory,
        dead_end_topics: List[str] | None = None,
        phase: str = "",
        user_preferences: str = "",
        instruction: str = "",
        **_ignored_options,
    ) -> Dict:
        """ask helper for the portable micro-coaching agent package."""
        messages = self.get_messages(
            shared_history,
            dead_end_topics=dead_end_topics,
            phase=phase,
            user_preferences=user_preferences,
            instruction=instruction,
        )

        raw = generate_response(
            self.model,
            messages,
            max_new_tokens=self.config.max_new_tokens,
            sampling=self.config.sampling,
        )

        template = self._parse_template(raw)

        # Duplicate-question detection helper
        _already_asked = shared_history.get_all_coach_questions()
        question_text = template.get("question_template", "")
        _GENERIC_FALLBACK_TEMPLATE = {
            "question_type": "fallback",
            "target": "meal",
            "reasoning": "Generic follow-up after duplicate detection",
            "question_template": "Could you tell me more about how this meal is put together?",
        }

        for _attempt in range(2):
            if not _is_duplicate_question(question_text, _already_asked):
                break
            _retry_msgs = messages + [{
                "role": "user",
                "content": (
                    "[SYSTEM NOTE: The question you just generated was already asked. "
                    "Please ask about a completely different food item or a new aspect "
                    "that has NOT yet been covered in this conversation.]"
                ),
            }]
            _retry_raw = generate_response(
                self.model,
                _retry_msgs,
                max_new_tokens=self.config.max_new_tokens,
                sampling=self.config.sampling,
            )
            template = self._parse_template(_retry_raw)
            question_text = template.get("question_template", "")
        else:
            if _is_duplicate_question(question_text, _already_asked):
                template = dict(_GENERIC_FALLBACK_TEMPLATE)

        self.own_buffer.add(template.get("question_template", ""))
        return template

    # Internal note

    def _parse_template(self, raw_output: str) -> Dict:
        """_parse_template helper for the portable micro-coaching agent package."""
        fallback = {
            "question_type": "fallback",
            "target": "meal",
            "reasoning": "(parse error)",
            "question_template": "Could you tell me more about your meal?",
        }
        try:
            data = load_json_object(raw_output)
            return {
                "question_type": str(data.get("question_type", "fallback")),
                "target": str(data.get("target", "")),
                "reasoning": str(data.get("reasoning", "")),
                "question_template": str(
                    data.get("question_template", fallback["question_template"])
                ),
            }
        except (JSONOutputError, ValueError, TypeError):
            # JSON raw text question_template
            clean = raw_output.strip()
            if clean and len(clean) < 200:
                fallback["question_template"] = clean
            return fallback

    def _build_system_prompt(
        self,
        dialog_summary: str,
        prev_questions: List[str] | None = None,
        dead_end_topics: List[str] | None = None,
        stall_exit: bool = False,
        natural_close: bool = False,
        phase: str = "",
        user_preferences: str = "",
        instruction: str = "",
    ) -> str:
        """_build_system_prompt helper for the portable micro-coaching agent package."""
        base = INFORMATION_SEEKER_SYSTEM_PROMPT.format(
            nutrition_goal=self.nutrition_goal,
            meal_type=self.meal_type,
        )

        parts = [base]

        if phase:
            parts.append(INFORMATION_SEEKER_PHASE_BLOCK.format(phase=phase))

        # Principle 4
        if dialog_summary:
            parts.append(INFORMATION_SEEKER_SUMMARY_BLOCK.format(dialog_summary=dialog_summary))

        # Principle 2
        if prev_questions:
            pq_lines = "\n".join(f"  {i+1}. {q}" for i, q in enumerate(prev_questions))
            pq_text = f"Questions already asked:\n{pq_lines}"
        else:
            pq_text = self.own_buffer.to_prompt_text(header="Questions already asked")
        parts.append(INFORMATION_SEEKER_QUESTIONS_BLOCK.format(own_buffer=pq_text))

        # User profile
        if user_preferences:
            parts.append(INFORMATION_SEEKER_PROFILE_BLOCK.format(user_preferences=user_preferences))

        if self.config.coach_use_template_guidance:
            parts.append(INFORMATION_SEEKER_STRATEGY_BLOCK.format(
                action_guidelines=INFORMATION_SEEKER_ACTION_GUIDELINES,
            ))

        if instruction.strip():
            parts.append(INFORMATION_SEEKER_ROUTER_BLOCK.format(
                instruction=instruction.strip(),
            ))

        # Dead-end
        if dead_end_topics:
            dead_end_list = "\n".join(f"  - {t}" for t in dead_end_topics)
            parts.append(INFORMATION_SEEKER_DEAD_END_BLOCK.format(dead_end_list=dead_end_list))

        # Stall-exit / Natural-close
        if natural_close:
            parts.append(INFORMATION_SEEKER_NATURAL_CLOSE_BLOCK)
        elif stall_exit:
            parts.append(INFORMATION_SEEKER_STALL_EXIT_BLOCK)

        return "".join(parts)
