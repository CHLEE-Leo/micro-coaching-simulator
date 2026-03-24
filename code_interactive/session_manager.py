"""
session_manager.py
─────────────────
Interactive 모드 세션 관리자.
/ Session manager for interactive mode.

각 사용자 세션은 독립적인 Coach, Judge, SharedConversationHistory 를 가집니다.
실제 사용자가 User LLM 역할을 대체하며, Coach LLM이 질문을 생성하고
Judge LLM이 매 턴 alignment를 평가합니다.

/ Each user session holds an independent Coach, Judge, and SharedConversationHistory.
  The real user replaces the User LLM. The Coach LLM generates questions;
  the Judge LLM evaluates alignment every turn.
"""

from __future__ import annotations

import sys
import uuid
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── sys.path 설정 순서가 중요합니다 / sys.path ORDER matters
#
# code_interactive/ 를 먼저 등록 → `from utils.llm_utils import ...` 가
# code_interactive/utils/llm_utils.py (llama-cpp-python) 를 가져갑니다.
# code/ 는 그 다음에 등록 → core/, models/ 를 가져옵니다.
# /code/utils/llm_utils.py (vLLM) 는 절대 호출되지 않습니다.
#
# / Register code_interactive/ FIRST so that CoachModel / JudgeModel's
#   `from utils.llm_utils import generate_response` resolves to the local
#   llama-cpp-python version, not the vLLM version in /code.
_INTERACTIVE_DIR = Path(__file__).resolve().parent      # code_interactive/
_CODE_DIR        = _INTERACTIVE_DIR.parent / "code"     # code/

# uvicorn -m 실행 시 code_interactive/가 이미 sys.path에 있으므로
# `if not in` 가드를 쓰면 insert(0)가 건너뛰어집니다 → 무조건 삽입합니다.
sys.path.insert(0, str(_CODE_DIR))          # code/ 등록
sys.path.insert(0, str(_INTERACTIVE_DIR))   # code_interactive/ → 최종 index 0

from core.memory import SharedConversationHistory
from models.coach import CoachModel
from models.judge import JudgeModel
from models.user  import UserModel
from utils.llm_utils import summarize_conversation   # → code_interactive/utils/llm_utils.py

import re as _re

# Non-answer 패턴: User가 정보를 제공하지 못한 발화 판별
# Patterns indicating the user has no further information to share
_NON_ANSWER_RE = _re.compile(
    r"(i'?m not sure|i haven'?t decided|not sure|just a standard"
    r"|i don'?t know|don'?t know|haven'?t decided|standard portion"
    r"|i'?m unsure|i'?m not really sure|no idea|not decided)",
    _re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# 세션 상태 열거형 / Session state enum
# ─────────────────────────────────────────────────────────────────────────────

class SessionStatus(str, Enum):
    ACTIVE      = "active"        # 진행 중 / in progress
    TERMINATED  = "terminated"    # 정상 종료 (Judge 일치) / clean termination
    MAX_TURNS   = "max_turns"     # 최대 턴 초과 / max turns exceeded
    ABANDONED   = "abandoned"     # 사용자 이탈 / user abandoned


# ─────────────────────────────────────────────────────────────────────────────
# 턴 레코드 / Turn record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TurnRecord:
    turn_idx:        int
    coach_utterance: str
    user_utterance:  Optional[str]   = None  # 사용자 응답 / user reply
    judge_aligned:   Optional[bool]  = None  # Judge 판정 / judge verdict
    judge_score:     Optional[float] = None  # 정규화 점수 / normalised score


# ─────────────────────────────────────────────────────────────────────────────
# 세션 데이터 / Session data
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Session:
    """단일 사용자 인터랙션 세션. / A single user interaction session."""

    session_id:       str
    mode:             str              # "custom" | "simulation"
    judge_enabled:    bool             # Judge AI 활성화 여부
    nutrition_goal:   str
    meal_description: str              # 음식 이름 목록 / food item names
    meal_ingredient:  str              # 재료/조리법 디테일 / ingredient details
    meal_type:        str              # breakfast / lunch / dinner / snack

    coach:   CoachModel
    judge:   JudgeModel
    history: SharedConversationHistory

    # simulation 모드 전용 / simulation mode only
    user:    Optional[UserModel] = None

    # Stall 추적 / Stall tracking (simulation mode)
    stall_count:     int       = 0   # 연속 non-answer 수
    dead_end_topics: List[str] = field(default_factory=list)  # User가 모른다고 한 질문 목록

    # 증분 요약용: 마지막 요약이 포함한 바로 다음 턴 인덱스
    # For incremental summarisation: turn_idx of the first turn NOT yet included in the summary
    last_summarized_start: int = 0

    # 프론트엔드 표시용 대화 기록 / Turn records for frontend display
    turns:    List[TurnRecord] = field(default_factory=list)
    turn_idx: int              = 0
    status:   SessionStatus    = SessionStatus.ACTIVE

    # 종료 정보 / Termination info
    terminated_by:  Optional[str]  = None
    final_aligned:  Optional[bool] = None
    final_score:    Optional[float] = None


# ─────────────────────────────────────────────────────────────────────────────
# 세션 매니저 / Session manager
# ─────────────────────────────────────────────────────────────────────────────

class SessionManager:
    """
    모든 활성 세션을 관리합니다.
    / Manages all active sessions.

    vLLM LLM 객체는 외부에서 한 번 로드되어 이 클래스에 주입됩니다.
    / The vLLM LLM object is loaded once externally and injected into this class.
    """

    def __init__(self, llm, config):
        """
        Parameters
        ----------
        llm    : load_model()로 로드된 vLLM LLM 객체 / vLLM LLM loaded by load_model()
        config : InteractiveConfig 인스턴스 / InteractiveConfig instance
        """
        self._llm     = llm
        self._config  = config
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()  # 멀티 스레드 안전 / thread-safe

    # ── 세션 생성 / Create session ───────────────────────────────────────────

    def create_session(
        self,
        nutrition_goal:   str,
        meal_description: str,
        meal_ingredient:  str,
        meal_type:        str  = "meal",
        mode:             str  = "custom",
        judge_enabled:    bool = True,
    ) -> Session:
        """
        새 세션을 생성하고, Coach의 첫 번째 질문(turn 0)을 생성합니다.
        / Creates a new session and generates the Coach's first question (turn 0).

        Parameters
        ----------
        mode          : "custom" | "simulation"
        judge_enabled : Judge AI 평가 활성화 여부

        Returns
        -------
        Session : 초기화된 세션 (첫 질문 포함) / Initialised session with first question
        """
        session_id = str(uuid.uuid4())

        # ── 에이전트 및 히스토리 초기화 / Initialise agents and history
        history = SharedConversationHistory(context_window=self._config.context_window)
        coach   = CoachModel(
            model=self._llm,
            nutrition_goal=nutrition_goal,
            meal_type=meal_type,
            config=self._config,
        )
        judge   = JudgeModel(
            model=self._llm,
            nutrition_goal=nutrition_goal,
            config=self._config,
        )

        # simulation 모드일 때만 UserModel 생성 / Create UserModel only in simulation mode
        user: Optional[UserModel] = None
        if mode == "simulation":
            user = UserModel(
                model=self._llm,
                nutrition_goal=nutrition_goal,
                meal_description=meal_description,
                meal_ingredient=meal_ingredient,
                config=self._config,
            )

        session = Session(
            session_id=session_id,
            mode=mode,
            judge_enabled=judge_enabled,
            nutrition_goal=nutrition_goal,
            meal_description=meal_description,
            meal_ingredient=meal_ingredient,
            meal_type=meal_type,
            coach=coach,
            judge=judge,
            history=history,
            user=user,
        )

        # ── Turn 0: Coach 첫 질문 생성 / Generate Coach's first question
        first_q = coach.first_question()
        history.add_turn(turn_idx=0, coach_utterance=first_q)
        session.turns.append(TurnRecord(turn_idx=0, coach_utterance=first_q))

        with self._lock:
            self._sessions[session_id] = session

        return session

    # ── 사용자 응답 처리 / Process user reply ────────────────────────────────

    def submit_reply(self, session_id: str, user_reply: str) -> Dict[str, Any]:
        """
        사용자의 응답을 받아 다음 Coach 질문과 Judge 판정을 반환합니다.
        / Receives user reply, returns next Coach question and Judge verdict.

        Returns
        -------
        dict with keys:
            turn_idx       : 현재 턴 인덱스 / current turn index
            coach_question : 다음 Coach 질문 (종료 시 None) / next coach question
            judge_aligned  : Judge 판정 (아직 판정 전이면 None) / judge verdict
            judge_score    : 정규화 점수 / normalised score
            status         : 세션 상태 / session status
            aligned_label  : 판정 레이블 문자열 / alignment label string
        """
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}")
        if session.status != SessionStatus.ACTIVE:
            raise ValueError(f"Session {session_id} is already {session.status}")

        # ── 사용자 응답을 히스토리에 기록 / Record user reply in history
        session.history.update_last_user_utterance(user_reply.strip())
        session.turns[-1].user_utterance = user_reply.strip()

        turn_idx = session.turn_idx
        judge_aligned: Optional[bool]  = None
        judge_score:   Optional[float] = None

        # ── Judge 판정 (judge_min_turn 이후부터, judge_enabled 시에만)
        # Judge evaluation (after judge_min_turn, only when judge_enabled)
        if session.judge_enabled and session.judge.should_judge(turn_idx):
            judge_msgs  = session.judge.get_messages(session.history)
            raw_verdict = _generate_single(self._llm, judge_msgs, self._config, mode="judge")
            aligned     = session.judge.apply_judgment(raw_verdict, turn_idx)
            judge_aligned = aligned
            judge_score   = session.judge.last_score

            session.turns[-1].judge_aligned = aligned
            session.turns[-1].judge_score   = judge_score

            # 종료 조건: pred == true_label
            # Termination condition: pred matches true label
            # (interactive 모드에서는 true_label이 없으므로 aligned == True 시 종료)
            # (in interactive mode there is no true_label, so we terminate on aligned == True)
            if aligned:
                session.status       = SessionStatus.TERMINATED
                session.terminated_by = "judge"
                session.final_aligned = aligned
                session.final_score   = judge_score

        # ── 요약 갱신 스케줄 / Rolling summary update
        completed = turn_idx + 1
        if completed % self._config.summarize_every == 0:
            _new_turns = session.history.to_plain_text_from(session.last_summarized_start)
            _new_summary = summarize_conversation(
                self._llm, _new_turns, prev_summary=session.history.summary
            )
            session.history.update_summary(_new_summary)
            session.last_summarized_start = completed

        # ── 다음 턴 준비 (종료되지 않은 경우) / Prepare next turn
        next_question: Optional[str] = None
        if session.status == SessionStatus.ACTIVE:
            next_turn = turn_idx + 1

            if next_turn >= self._config.max_turns:
                # 최대 턴 초과 / max turns exceeded
                session.status        = SessionStatus.MAX_TURNS
                session.terminated_by = "max_turns"
                session.final_aligned = judge_aligned
                session.final_score   = judge_score
            else:
                session.turn_idx = next_turn
                coach_msgs = session.coach.get_messages(session.history)
                next_question = _generate_single(self._llm, coach_msgs, self._config, mode="coach")

                # TERMINATION_TOKEN 방어 처리 / Guard against termination token
                next_question = next_question.replace(
                    SharedConversationHistory.TERMINATION_TOKEN, ""
                ).strip() or "Could you tell me more about your meal?"

                session.coach.own_buffer.add(next_question)
                session.history.add_turn(turn_idx=next_turn, coach_utterance=next_question)
                session.turns.append(TurnRecord(turn_idx=next_turn, coach_utterance=next_question))

        # ── 종료 시 최종 요약 / Final summary on termination
        if session.status != SessionStatus.ACTIVE:
            _final_new_turns = session.history.to_plain_text_from(session.last_summarized_start)
            if _final_new_turns:
                _new_summary = summarize_conversation(
                    self._llm, _final_new_turns, prev_summary=session.history.summary
                )
                session.history.update_summary(_new_summary)

        return {
            "turn_idx":      turn_idx,
            "coach_question": next_question,
            "judge_aligned":  judge_aligned,
            "judge_score":    judge_score,
            "status":         session.status.value,
            "aligned_label":  aligned_label,
        }

    # ── Simulation 스텝 / Simulation step ───────────────────────────────────

    def sim_step(self, session_id: str) -> Dict[str, Any]:
        """
        Simulation 모드 전용: AI User가 한 번 응답하고 Judge 평가 후 다음 Coach 질문을 반환합니다.
        / Simulation mode only: AI User responds, Judge evaluates, returns next Coach question.

        Returns
        -------
        dict with keys:
            turn_idx       : 현재 턴 / current turn index
            user_reply     : AI User 발화 / AI user utterance
            coach_question : 다음 Coach 질문 (종료 시 None) / next coach question
            judge_aligned  : Judge 판정 / judge verdict
            judge_score    : 정규화 점수 / normalised score
            status         : 세션 상태 / session status
            aligned_label  : 판정 레이블 / alignment label
        """
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}")
        if session.status != SessionStatus.ACTIVE:
            raise ValueError(f"Session {session_id} is already {session.status}")
        if session.user is None:
            raise ValueError(f"Session {session_id} is not in simulation mode")

        # ── AI User 응답 생성 / Generate AI User response
        user_msgs   = session.user.get_messages(session.history)
        user_reply  = _generate_single(self._llm, user_msgs, self._config, mode="user")

        # TERMINATION_TOKEN 처리 / Handle termination token
        natural_end = SharedConversationHistory.TERMINATION_TOKEN in user_reply
        user_reply_clean = user_reply.replace(
            SharedConversationHistory.TERMINATION_TOKEN, ""
        ).strip()

        # User가 TERMINATION_TOKEN만 생성하고 텍스트가 비어있으면 표시용 기본갑 사용
        # If user emitted only the termination token with no preceding text, use a safe default
        if natural_end and not user_reply_clean:
            user_reply_clean = "I think that covers everything about my meal."

        # 히스토리 갱신 / Update history
        session.history.update_last_user_utterance(user_reply_clean)
        session.turns[-1].user_utterance = user_reply_clean
        # 실질적 정보가 있는 발화만 own_buffer에 기록 (non-answer는 제외)
        # Only add to own_buffer if the reply contains actual meal information
        if not _is_non_answer(user_reply_clean):
            session.user.own_buffer.add(user_reply_clean)

        turn_idx      = session.turn_idx

        # 최소 턴 수 미달 시 자연 종료를 무시 — 너무 이른 시점에 TERMINATION_TOKEN이 생성된 경우 억제
        # Suppress natural_end if the conversation has not yet reached the minimum required turns
        _min_natural_end_turn = getattr(self._config, 'min_natural_end_turn', 3)
        if natural_end and turn_idx < _min_natural_end_turn:
            natural_end = False

        judge_aligned: Optional[bool]  = None
        judge_score:   Optional[float] = None

        # ── Judge 평가 / Judge evaluation
        if session.judge_enabled and session.judge.should_judge(turn_idx):
            judge_msgs  = session.judge.get_messages(session.history)
            raw_verdict = _generate_single(self._llm, judge_msgs, self._config, mode="judge")
            aligned     = session.judge.apply_judgment(raw_verdict, turn_idx)
            judge_aligned = aligned
            judge_score   = session.judge.last_score

            session.turns[-1].judge_aligned = aligned
            session.turns[-1].judge_score   = judge_score

            if aligned:
                session.status        = SessionStatus.TERMINATED
                session.terminated_by = "judge"
                session.final_aligned = aligned
                session.final_score   = judge_score

        # ── 요약 갱신 / Rolling summary
        completed = turn_idx + 1
        if completed % self._config.summarize_every == 0:
            _new_turns = session.history.to_plain_text_from(session.last_summarized_start)
            _new_summary = summarize_conversation(
                self._llm, _new_turns, prev_summary=session.history.summary
            )
            session.history.update_summary(_new_summary)
            session.last_summarized_start = completed

        # ── 자연 종료: 직접 TERMINATED 하지 않고 Coach 마무리 발화 생성 후 종료
        # Natural end: do NOT terminate immediately — let Coach generate a closing message first
        closing_for_natural = (natural_end and session.status == SessionStatus.ACTIVE)

        # ── Stall 감지 / Detect stall (non-answer or real answer)
        _stall_exit_turns = getattr(self._config, 'stall_exit_turns', 3)
        if _is_non_answer(user_reply_clean) and not natural_end:
            session.stall_count += 1
            # Coach가 방금 한 질문을 dead-end 토픽으로 기록
            if session.turns and session.turns[-1].coach_utterance:
                session.dead_end_topics.append(session.turns[-1].coach_utterance)
        else:
            session.stall_count = 0   # 실질적 답변 시 리셋 / reset on real answer

        stall_exit_now = (
            session.status == SessionStatus.ACTIVE
            and not closing_for_natural
            and session.stall_count >= _stall_exit_turns
        )

        # ── 다음 Coach 질문 준비 / Prepare next Coach question
        next_question: Optional[str] = None
        if session.status == SessionStatus.ACTIVE:
            next_turn = turn_idx + 1
            if next_turn >= self._config.max_turns:
                session.status        = SessionStatus.MAX_TURNS
                session.terminated_by = "max_turns"
                session.final_aligned = judge_aligned
                session.final_score   = judge_score
            else:
                session.turn_idx = next_turn
                coach_msgs = session.coach.get_messages(
                    session.history,
                    dead_end_topics=session.dead_end_topics if session.dead_end_topics else None,
                    stall_exit=stall_exit_now,
                    natural_close=closing_for_natural,
                )
                next_question = _generate_single(
                    self._llm, coach_msgs, self._config, mode="coach"
                )

                # ── 중복 질문 탐지 + 재시도 / Duplicate question detection + retry
                # greedy 샘플링에서도 모델이 own_buffer 지시를 무시하는 경우를 프로그래매틱으로 차단
                # Programmatic guard against repeated questions — use history as authoritative source
                _already_asked = session.history.get_all_coach_questions()
                if _is_duplicate_question(next_question, _already_asked):
                    _retry_msgs = coach_msgs + [{
                        "role": "user",
                        "content": (
                            "[SYSTEM NOTE: The question you just generated was already asked. "
                            "Please ask about a completely different food item or a new aspect "
                            "that has NOT yet been covered in this conversation.]"
                        ),
                    }]
                    _retry = _generate_single(self._llm, _retry_msgs, self._config, mode="coach")
                    if _retry.strip():
                        next_question = _retry

                # Stall-exit / natural-close fallback when LLM returns empty
                _fallback = (
                    "Thanks for sharing all of that — I think I have a good picture of your meal!"
                    if (stall_exit_now or closing_for_natural) else
                    "Could you tell me more about your meal?"
                )
                next_question = next_question.replace(
                    SharedConversationHistory.TERMINATION_TOKEN, ""
                ).strip() or _fallback

                # Closing 모드에서 모델이 질문을 생성한 경우 (? 포함) 폴백으로 교체
                # If Coach generated a question in closing mode, replace with the safe fallback
                if (stall_exit_now or closing_for_natural) and '?' in next_question:
                    next_question = _fallback

                session.coach.own_buffer.add(next_question)
                session.history.add_turn(
                    turn_idx=next_turn, coach_utterance=next_question
                )
                session.turns.append(
                    TurnRecord(turn_idx=next_turn, coach_utterance=next_question)
                )

                # Coach 마무리 발화 생성 후 세션 종료
                if stall_exit_now:
                    session.status        = SessionStatus.TERMINATED
                    session.terminated_by = "stall_exit"
                    session.final_aligned = judge_aligned
                    session.final_score   = judge_score
                elif closing_for_natural:
                    session.status        = SessionStatus.TERMINATED
                    session.terminated_by = "user_natural"
                    session.final_aligned = judge_aligned
                    session.final_score   = judge_score

        # ── 종료 시 최종 요약 / Final summary
        if session.status != SessionStatus.ACTIVE:
            _final_new_turns = session.history.to_plain_text_from(session.last_summarized_start)
            if _final_new_turns:
                _new_summary = summarize_conversation(
                    self._llm, _final_new_turns, prev_summary=session.history.summary
                )
                session.history.update_summary(_new_summary)

        return {
            "turn_idx":       turn_idx,
            "user_reply":     user_reply_clean,
            "coach_question": next_question,
            "judge_aligned":  judge_aligned,
            "judge_score":    judge_score,
            "status":         session.status.value,
            "aligned_label":  _alignment_label(judge_aligned),
        }

    # ── 세션 조회 / Get session ───────────────────────────────────────────────

    def get_session(self, session_id: str) -> Optional[Session]:
        """세션 객체 반환 / Return session object."""
        with self._lock:
            return self._sessions.get(session_id)

    def get_history(self, session_id: str) -> Dict[str, Any]:
        """
        프론트엔드 표시용 전체 대화 기록을 반환합니다.
        / Returns full conversation history for frontend display.
        """
        session = self.get_session(session_id)
        if session is None:
            return {}
        return {
            "session_id":      session.session_id,
            "mode":            session.mode,
            "judge_enabled":   session.judge_enabled,
            "nutrition_goal":  session.nutrition_goal,
            "meal_description": session.meal_description,
            "meal_ingredient": session.meal_ingredient,
            "meal_type":       session.meal_type,
            "turns": [
                {
                    "turn_idx":        t.turn_idx,
                    "coach_utterance": t.coach_utterance,
                    "user_utterance":  t.user_utterance,
                    "judge_aligned":   t.judge_aligned,
                    "judge_score":     t.judge_score,
                    "aligned_label":   _alignment_label(t.judge_aligned),
                }
                for t in session.turns
            ],
            "summary":        session.history.summary,
            "status":         session.status.value,
            "terminated_by":  session.terminated_by,
            "final_aligned":  session.final_aligned,
            "final_score":    session.final_score,
        }

    def abandon_session(self, session_id: str) -> None:
        """세션을 ABANDONED 상태로 표시합니다. / Mark session as ABANDONED."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session and session.status == SessionStatus.ACTIVE:
                session.status        = SessionStatus.ABANDONED
                session.terminated_by = "abandoned"

    def remove_session(self, session_id: str) -> None:
        """메모리에서 세션을 삭제합니다. / Remove session from memory."""
        with self._lock:
            self._sessions.pop(session_id, None)


# ─────────────────────────────────────────────────────────────────────────────
# 내부 헬퍼 / Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _generate_single(llm, messages: list, config, mode: str = "coach") -> str:
    """
    단일 응답 생성 (batch_generate 대신 단건 호출).
    / Generate a single response (single-call instead of batch_generate).

    mode: "coach" | "user" | "judge"
    """
    from utils.llm_utils import generate_response as _gen

    if mode == "judge":
        return _gen(
            llm, messages,
            max_new_tokens=config.judge_max_new_tokens,
            sampling=config.judge_sampling,
            stop_at_newline=False,
        )
    if mode == "coach":
        # coach_sampling 이 설정되어 있으면 우선 사용, 없으면 config.sampling 폴백
        # If coach_sampling is configured use it, otherwise fall back to config.sampling
        _sampling = getattr(config, 'coach_sampling', config.sampling)
        return _gen(
            llm, messages,
            max_new_tokens=config.max_new_tokens,
            sampling=_sampling,
        )
    # mode == "user"
    return _gen(
        llm, messages,
        max_new_tokens=config.max_new_tokens,
        sampling=config.sampling,
    )


def _is_duplicate_question(new_q: str, already_asked: list, threshold: float = 0.55) -> bool:
    """
    Jaccard 기반 단어 중복 판단: 새 질문이 이미 물어본 질문들과 너무 비슷하면 True.
    / Jaccard word-overlap: returns True if new_q is too similar to any already-asked question.
    """
    # 3자 이하 단어 제외 (a, an, of ... 등 기능어 필터링)
    # Ignore short stopwords to focus on content words
    words_new = {w.lower().strip('?!.,') for w in new_q.split() if len(w) > 3}
    if not words_new:
        return False
    for prev in already_asked:
        words_prev = {w.lower().strip('?!.,') for w in prev.split() if len(w) > 3}
        if not words_prev:
            continue
        union = words_new | words_prev
        overlap = len(words_new & words_prev) / len(union)
        if overlap >= threshold:
            return True
    return False


def _is_non_answer(text: str) -> bool:
    """
    User 발화가 실질적 정보가 없는 non-answer 인지 판별합니다.
    / Determine whether a user utterance carries no real meal information.
    """
    stripped = text.strip()
    if not stripped:
        return True
    return bool(_NON_ANSWER_RE.search(stripped))


def _alignment_label(aligned: Optional[bool]) -> str:
    """
    Judge 판정을 사람이 읽기 좋은 레이블로 변환합니다.
    / Convert judge verdict to a human-readable label.
    """
    if aligned is None:
        return "pending"
    return "aligned" if aligned else "not_aligned"
