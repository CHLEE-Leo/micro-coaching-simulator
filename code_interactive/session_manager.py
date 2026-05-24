"""Session storage and UI adapter for the interactive simulator.

This module keeps web/session concerns separate from agent flow. The public API
used by ``app.py`` is stable, while each user turn is delegated to
``agents.engine.ConversationEngine``.
"""

from __future__ import annotations

import logging
import uuid
import threading
from dataclasses import dataclass, field, replace as _dc_replace
from enum import Enum
from typing import Any, Dict, List, Optional

try:
    from .web_app_config import WebAppConfig
    from .agents.agent_config import AgentConfig
    from .agents.contracts import (
        ChatMessage,
        CoachingState,
        CoachingTurnRequest,
        UserProfileContext,
    )
    from .agents.memory.conversation_memory import SharedConversationHistory
    from .agents.engine import ConversationEngine
    from .agents.modules.alignment_estimator import AlignmentEstimator
    from .agents.modules.certainty_estimator import CertaintyEstimator
    from .agents.modules.context_tracker import ContextTracker
    from .agents.modules.guardrail import Guardrail
    from .agents.modules.information_seeker import InformationSeeker
    from .agents.modules.meal_recommender import MealRecommender
    from .agents.modules.orchestrator import Orchestrator
except ImportError:  # pragma: no cover - script execution via python app.py
    from web_app_config import WebAppConfig
    from agents.agent_config import AgentConfig
    from agents.contracts import (
        ChatMessage,
        CoachingState,
        CoachingTurnRequest,
        UserProfileContext,
    )
    from agents.memory.conversation_memory import SharedConversationHistory
    from agents.engine import ConversationEngine
    from agents.modules.alignment_estimator import AlignmentEstimator
    from agents.modules.certainty_estimator import CertaintyEstimator
    from agents.modules.context_tracker import ContextTracker
    from agents.modules.guardrail import Guardrail
    from agents.modules.information_seeker import InformationSeeker
    from agents.modules.meal_recommender import MealRecommender
    from agents.modules.orchestrator import Orchestrator

logger = logging.getLogger("micro-coach-interactive")


class SessionStatus(str, Enum):
    ACTIVE = "active"
    TERMINATED = "terminated"
    MAX_TURNS = "max_turns"
    ABANDONED = "abandoned"


@dataclass
class TurnRecord:
    turn_idx: int
    coach_utterance: str
    coach_messages: List[str] = field(default_factory=list)
    user_utterance: Optional[str] = None
    alignment_aligned: Optional[bool] = None
    alignment_score: Optional[float] = None
    alignment_reasoning: Optional[str] = None
    certainty_score: Optional[float] = None
    certainty_reasoning: Optional[str] = None
    phase: Optional[str] = None
    orchestrator_action: Optional[str] = None
    orchestrator_reasoning: Optional[str] = None
    orchestrator_instruction: Optional[str] = None
    intent_summary: Optional[str] = None
    user_intent: Optional[str] = None
    guardrail_blocked: bool = False
    input_guard_passed: Optional[bool] = None
    output_guard_passed: Optional[bool] = None
    meal_tracker_output: Optional[str] = None
    context_tracker_output: Optional[str] = None
    recommendation_result: Optional[Dict[str, Any]] = None
    assessment_result: Optional[Dict[str, Any]] = None


@dataclass
class Session:
    session_id: str
    mode: str
    alignment_enabled: bool
    nutrition_goal: str
    meal_description: str
    meal_ingredient: str
    meal_type: str
    agent_config: AgentConfig
    coach: InformationSeeker
    alignment_tracker: AlignmentEstimator
    history: SharedConversationHistory
    coach_conversation_mode: str = "template-based"
    context_tracking: bool = True
    uncertainty_tracking: bool = False
    certainty_tracker: Optional[CertaintyEstimator] = None
    orchestrator: Optional[Orchestrator] = None
    meal_recommender: Optional[MealRecommender] = None
    guardrail: Optional[Guardrail] = None
    context_tracker: Optional[ContextTracker] = None
    phase: str = "exploration"
    user_preferences: str = ""
    stall_count: int = 0
    dead_end_topics: List[str] = field(default_factory=list)
    consecutive_qa_count: int = 0
    recommendation_rejection_count: int = 0
    consecutive_guard_blocks: int = 0
    recent_blocked_messages: List[str] = field(default_factory=list)
    last_meal_track_start: int = 0
    last_summarized_start: int = 0
    last_alignment_meal_hash: Optional[str] = None
    last_certainty_meal_hash: Optional[str] = None
    turns: List[TurnRecord] = field(default_factory=list)
    turn_idx: int = 0
    status: SessionStatus = SessionStatus.ACTIVE
    terminated_by: Optional[str] = None
    final_aligned: Optional[bool] = None
    final_score: Optional[float] = None
    coaching_state: CoachingState = field(default_factory=CoachingState)


class SessionManager:
    """Manage interactive sessions while delegating turns to the portable engine."""

    def __init__(
        self,
        chatgpt_client_pool=None,
        config=None,
        chatgpt_client=None,
        chatgpt_light_client=None,
    ):
        if config is None:
            self._config = WebAppConfig()
        elif isinstance(config, AgentConfig):
            self._config = WebAppConfig(agent=config)
        else:
            self._config = config
        self._agent_config: AgentConfig = getattr(self._config, "agent", self._config)
        if chatgpt_client_pool:
            self._client_pool = chatgpt_client_pool
        else:
            heavy = chatgpt_client
            light = chatgpt_light_client or chatgpt_client
            self._client_pool = {
                self._config.chatgpt_model: heavy,
                self._config.chatgpt_light_model: light,
            }
        self._chatgpt_client = next(iter(self._client_pool.values()))
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()

    def _client_for(self, module: str):
        model_name = self._config.resolve_model_name(module)
        return self._client_pool.get(model_name) or next(iter(self._client_pool.values()))

    def _run_module_inference(
        self,
        *,
        module: str,
        messages,
        mode: str,
        agent_config: AgentConfig | None = None,
    ) -> str:
        try:
            from .agents.openai_client import generate_response as _generate_response
        except ImportError:  # pragma: no cover - script execution via python app.py
            from agents.openai_client import generate_response as _generate_response

        cfg = agent_config or self._agent_config
        options = cfg.generation_options(mode)
        return _generate_response(
            self._client_for(module),
            [dict(message) for message in messages],
            max_new_tokens=options["max_new_tokens"],
            sampling=options["sampling"],
            stop_at_newline=options["stop_at_newline"],
            reasoning_effort=self._config.resolve_reasoning_effort(module),
            reasoning_summary=self._config.resolve_reasoning_summary(module),
        )

    def _session_config(self, **overrides):
        valid = {k: v for k, v in overrides.items() if v is not None}
        if not valid:
            return self._agent_config
        return _dc_replace(self._agent_config, **valid)

    def create_session(
        self,
        nutrition_goal: str,
        meal_description: str,
        meal_ingredient: str,
        meal_type: str = "meal",
        mode: str = "custom",
        alignment_enabled: bool = True,
        coach_conversation_mode: Optional[str] = None,
        alignment_use_goal_def: Optional[bool] = None,
        alignment_use_workflow: Optional[bool] = None,
        alignment_output_format: Optional[str] = None,
        uncertainty_tracking: Optional[bool] = None,
        context_tracking: Optional[bool] = None,
        persona_activity_level: Optional[str] = None,
        persona_diet_preferences: Optional[List[str]] = None,
        persona_allergies: Optional[List[str]] = None,
        persona_health_concerns: Optional[List[str]] = None,
    ) -> Session:
        overrides: Dict[str, Any] = {}
        if alignment_use_goal_def is not None:
            overrides["alignment_use_goal_def"] = alignment_use_goal_def
        if alignment_use_workflow is not None:
            overrides["alignment_use_workflow"] = alignment_use_workflow
        if alignment_output_format in ("binary", "0-1", "0-100"):
            overrides["alignment_output_format"] = alignment_output_format

        conv_mode = (coach_conversation_mode or "template-based").lower()
        if conv_mode == "open-ended":
            overrides["coach_use_template_guidance"] = False
        cfg = self._session_config(**overrides)

        session_id = str(uuid.uuid4())
        history = SharedConversationHistory(context_window=cfg.context_window)
        coach = InformationSeeker(
            model=self._chatgpt_client,
            nutrition_goal=nutrition_goal,
            meal_type=meal_type,
            config=cfg,
        )

        context_tracker = ContextTracker()
        if persona_activity_level or persona_diet_preferences or persona_allergies or persona_health_concerns:
            context_tracker.set_profile_from_persona(
                activity_level=persona_activity_level,
                diet_preferences=persona_diet_preferences,
                allergies=persona_allergies,
                health_concerns=persona_health_concerns,
            )

        session = Session(
            session_id=session_id,
            mode=mode,
            alignment_enabled=alignment_enabled,
            nutrition_goal=nutrition_goal,
            meal_description=meal_description,
            meal_ingredient=meal_ingredient,
            meal_type=meal_type,
            agent_config=cfg,
            coach=coach,
            alignment_tracker=AlignmentEstimator(
                model=self._chatgpt_client,
                nutrition_goal=nutrition_goal,
                config=cfg,
            ),
            history=history,
            coach_conversation_mode=conv_mode,
            context_tracking=(context_tracking if context_tracking is not None else True),
            uncertainty_tracking=bool(uncertainty_tracking),
            certainty_tracker=(
                CertaintyEstimator(nutrition_goal=nutrition_goal, config=cfg)
                if uncertainty_tracking else None
            ),
            orchestrator=Orchestrator(nutrition_goal=nutrition_goal, config=cfg),
            meal_recommender=MealRecommender(nutrition_goal=nutrition_goal, config=cfg),
            guardrail=Guardrail(config=cfg),
            context_tracker=context_tracker,
            phase="exploration",
            coaching_state=CoachingState(phase="exploration", status="active"),
        )

        first_q = coach.first_question()
        history.add_turn(turn_idx=0, coach_utterance=first_q)
        session.turns.append(TurnRecord(turn_idx=0, coach_utterance=first_q))

        with self._lock:
            self._sessions[session_id] = session
        return session

    def continue_session(
        self,
        previous_session_id: str,
        nutrition_goal: str,
        meal_description: str,
        meal_ingredient: str,
        meal_type: str = "meal",
    ) -> Session:
        previous = self.get_session(previous_session_id)
        if previous is None:
            raise KeyError(f"Previous session not found: {previous_session_id}")

        new_session = self.create_session(
            nutrition_goal=nutrition_goal,
            meal_description=meal_description,
            meal_ingredient=meal_ingredient,
            meal_type=meal_type,
            mode=previous.mode,
            alignment_enabled=previous.alignment_enabled,
            coach_conversation_mode=previous.coach_conversation_mode,
            uncertainty_tracking=previous.uncertainty_tracking,
            context_tracking=previous.context_tracking,
        )
        carried_state = _dc_replace(
            new_session.coaching_state,
            context_base=previous.coaching_state.context_base,
            user_preferences=previous.coaching_state.user_preferences,
        )
        new_session.coaching_state = carried_state
        new_session.history.update_context_base(carried_state.context_base)
        if previous.context_tracker is not None:
            new_session.context_tracker = previous.context_tracker
        return new_session

    def submit_reply(self, session_id: str, user_reply: str) -> Dict[str, Any]:
        session = self.get_session(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}")
        if session.status != SessionStatus.ACTIVE:
            raise ValueError(f"Session {session_id} is already {session.status}")

        clean_reply = user_reply.strip()
        if not clean_reply:
            raise ValueError("user_reply must not be empty")

        current_turn = session.turn_idx
        history_messages = self._flat_history(session)
        profile = self._profile_for(session)
        request = CoachingTurnRequest(
            current_message=clean_reply,
            history=history_messages,
            profile=profile,
            state=session.coaching_state,
            nutrition_goal=session.nutrition_goal,
            meal_type=session.meal_type,
            opening_message=session.turns[0].coach_utterance if session.turns else None,
            enable_opening_fallback=True,
            enable_guardrail=True,
            enable_context_tracking=session.context_tracking,
            enable_certainty=session.uncertainty_tracking,
        )
        engine = ConversationEngine(
            generate_response=lambda *, module, messages, mode: self._run_module_inference(
                module=module,
                messages=messages,
                mode=mode,
                agent_config=session.agent_config,
            ),
            config=session.agent_config,
        )
        result = engine.generate_chat_replies(request)

        if session.turns:
            session.turns[-1].user_utterance = clean_reply
        session.history.update_last_user_utterance(clean_reply)

        coach_messages = [m.content for m in result.assistant_messages if m.content]
        coach_question = coach_messages[-1] if coach_messages else None
        next_turn = current_turn + 1

        session.coaching_state = result.state
        session.phase = result.state.phase
        session.user_preferences = result.state.user_preferences
        session.stall_count = result.state.stall_count
        session.consecutive_qa_count = result.state.consecutive_qa_count
        session.recommendation_rejection_count = result.state.recommendation_rejection_count
        session.history.update_meal_base(result.state.meal_base)
        session.history.update_tracker_state(result.state.tracker_state)
        session.history.update_context_base(result.state.context_base)

        if coach_question:
            merged_coach = "\n\n".join(coach_messages)
            session.turn_idx = next_turn
            session.history.add_turn(turn_idx=next_turn, coach_utterance=merged_coach)
            session.turns.append(
                TurnRecord(
                    turn_idx=next_turn,
                    coach_utterance=coach_question,
                    coach_messages=coach_messages,
                )
            )

        if result.status == "terminated":
            session.status = SessionStatus.TERMINATED
            session.terminated_by = result.terminated_by or "orchestrator"
        elif next_turn >= self._agent_config.max_turns:
            session.status = SessionStatus.MAX_TURNS
            session.terminated_by = "max_turns"
        else:
            session.status = SessionStatus.ACTIVE

        metadata = dict(result.metadata)
        alignment_aligned = metadata.get("alignment_aligned")
        alignment_score = result.state.last_alignment_score
        alignment_reasoning = result.state.last_alignment_reasoning
        certainty_score = result.state.last_certainty_score
        certainty_reasoning = result.state.last_certainty_reasoning

        response = {
            "turn_idx": current_turn,
            "coach_question": coach_question,
            "coach_messages": coach_messages,
            "assessment_message": self._first_message_of_kind(result.assistant_messages, "assessment"),
            "user_reply": clean_reply,
            "status": session.status.value,
            "terminated_by": session.terminated_by,
            "phase": session.phase,
            "alignment_aligned": alignment_aligned,
            "alignment_score": alignment_score,
            "alignment_reasoning": alignment_reasoning,
            "aligned_label": _alignment_label(alignment_aligned),
            "certainty_score": certainty_score,
            "certainty_reasoning": certainty_reasoning,
            "input_guard_input": None,
            "input_guard_output": metadata.get("input_guard"),
            "output_guard_input": None,
            "output_guard_output": None,
            "guardrail_blocked": False,
            "alignment_input": None,
            "alignment_raw_output": metadata.get("alignment_raw_output"),
            "meal_tracker_input": None,
            "meal_tracker_output": metadata.get("meal_tracker_output"),
            "context_tracker_input": None,
            "context_tracker_output": metadata.get("context_tracker_output"),
            "certainty_input": None,
            "certainty_output": metadata.get("certainty_raw_output"),
            "orchestrator_input": None,
            "orchestrator_raw_output": metadata.get("orchestrator_raw_output"),
            "orchestrator_decision": metadata.get("orchestrator_decision"),
            "recommendation_result": metadata.get("recommendation_result"),
            "assessment_result": metadata.get("assessment_result"),
            "post_assessment_decision": metadata.get("post_assessment_decision"),
            "engine_metadata": metadata,
        }
        self._populate_turn_monitoring(session, response)
        return response

    def get_session(self, session_id: str) -> Optional[Session]:
        with self._lock:
            return self._sessions.get(session_id)

    def get_history(self, session_id: str) -> Dict[str, Any]:
        session = self.get_session(session_id)
        if session is None:
            return {}

        turns_export = []
        for turn in session.turns:
            entry: Dict[str, Any] = {
                "turn_idx": turn.turn_idx,
                "phase": turn.phase,
                "dialogue": {
                    "coach": turn.coach_utterance,
                    "user": turn.user_utterance,
                },
                "orchestrator": {
                    "action": turn.orchestrator_action,
                    "reasoning": turn.orchestrator_reasoning,
                    "instruction": turn.orchestrator_instruction,
                    "intent_summary": turn.intent_summary,
                    "user_intent": turn.user_intent,
                },
                "guardrail": {
                    "blocked": turn.guardrail_blocked,
                    "input_guard_passed": turn.input_guard_passed,
                    "output_guard_passed": turn.output_guard_passed,
                },
                "trackers": {
                    "meal_tracker": turn.meal_tracker_output,
                    "context_tracker": turn.context_tracker_output,
                },
                "alignment": {
                    "aligned": turn.alignment_aligned,
                    "score": turn.alignment_score,
                    "reasoning": turn.alignment_reasoning,
                    "label": _alignment_label(turn.alignment_aligned),
                },
                "certainty": {
                    "score": turn.certainty_score,
                    "reasoning": turn.certainty_reasoning,
                },
            }
            if turn.recommendation_result:
                entry["recommendation"] = turn.recommendation_result
            if turn.assessment_result:
                entry["assessment"] = turn.assessment_result
            turns_export.append(entry)

        return {
            "session_id": session.session_id,
            "mode": session.mode,
            "nutrition_goal": session.nutrition_goal,
            "meal_description": session.meal_description,
            "meal_ingredient": session.meal_ingredient,
            "meal_type": session.meal_type,
            "status": session.status.value,
            "terminated_by": session.terminated_by,
            "turns": turns_export,
            "final_state": {
                "meal_base": session.coaching_state.meal_base,
                "context_base": session.coaching_state.context_base,
                "final_aligned": session.final_aligned,
                "final_score": session.final_score,
            },
        }

    def abandon_session(self, session_id: str) -> None:
        session = self.get_session(session_id)
        if session is not None:
            session.status = SessionStatus.ABANDONED
            session.terminated_by = "abandoned"

    def remove_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    @staticmethod
    def _first_message_of_kind(messages, kind: str) -> Optional[str]:
        for msg in messages:
            if getattr(msg, "kind", None) == kind:
                return msg.content
        return None

    @staticmethod
    def _flat_history(session: Session) -> List[ChatMessage]:
        messages: List[ChatMessage] = []
        for turn in session.turns:
            coach_messages = turn.coach_messages or ([turn.coach_utterance] if turn.coach_utterance else [])
            for text in coach_messages:
                if text:
                    messages.append(ChatMessage(role="assistant", content=text))
            if turn.user_utterance:
                messages.append(ChatMessage(role="user", content=turn.user_utterance))
        return messages

    @staticmethod
    def _profile_for(session: Session) -> UserProfileContext:
        profile = session.context_tracker._profile if session.context_tracker is not None else {}
        return UserProfileContext(
            preferences=tuple(profile.get("diet_preferences") or ()),
            allergies=tuple(profile.get("allergies") or ()),
            activity_level=profile.get("activity_level") or None,
            extra={"health_concerns": list(profile.get("health_concerns") or [])},
        )

    @staticmethod
    def _populate_turn_monitoring(session: Session, result: Dict[str, Any]) -> None:
        target_idx = result.get("turn_idx")
        target = next((t for t in reversed(session.turns) if t.turn_idx == target_idx), None)
        if target is None:
            return
        target.phase = result.get("phase")
        target.alignment_aligned = result.get("alignment_aligned")
        target.alignment_score = result.get("alignment_score")
        target.alignment_reasoning = result.get("alignment_reasoning")
        target.certainty_score = result.get("certainty_score")
        target.certainty_reasoning = result.get("certainty_reasoning")
        target.meal_tracker_output = result.get("meal_tracker_output")
        target.context_tracker_output = result.get("context_tracker_output")
        target.recommendation_result = result.get("recommendation_result")
        target.assessment_result = result.get("assessment_result")
        target.guardrail_blocked = bool(result.get("guardrail_blocked", False))
        target.input_guard_passed = not target.guardrail_blocked if result.get("input_guard_output") else None
        orch = result.get("orchestrator_decision")
        if isinstance(orch, dict):
            target.orchestrator_action = orch.get("action")
            target.orchestrator_reasoning = orch.get("reasoning")
            target.orchestrator_instruction = orch.get("instruction")
            target.intent_summary = orch.get("intent_summary")
            target.user_intent = orch.get("user_intent")


def _alignment_label(aligned: Optional[bool]) -> str:
    if aligned is True:
        return "aligned"
    if aligned is False:
        return "not_aligned"
    return "pending"
