"""
session_manager.py
─────────────────
Interactive 모드 세션 관리자.
/ Session manager for interactive mode.

에이전트 상호작용 플로우 (Orchestrator-centric)
───────────────────────────────────────────────

  사용자 ↔ [Guardrail] ↔ Orchestrator ↔ {InformationSeeker, MealRecommender, Estimators}

  Orchestrator가 대화의 유일한 커뮤니케이션 허브입니다.
  InformationSeeker와 MealRecommender는 구조화된 템플릿을 제공하는 서브 에이전트이며,
  사용자와 직접 소통하지 않습니다. 모든 사용자 대면 텍스트는 Orchestrator의
  TextGen LLM 호출을 통해 생성됩니다.

 [Custom 모드]  사용자가 직접 응답
    User(사람) → [Guardrail Input Guard]
               → MealTracker + DialogSummarizer
               → Orchestrator.route() → action 결정
                 ├─ seek_meal_info → IS.ask() → Orchestrator TextGen(질문)
                 ├─ seek_recommendation_info → IS.ask() → Orchestrator TextGen(질문)
                 ├─ assess_meal → Orchestrator.assess() → Orchestrator TextGen(피드백)
                 ├─ recommend → MR.recommend() → Orchestrator TextGen(추천)
                 ├─ motivational_close → Orchestrator.assess() → Orchestrator TextGen(동기부여 마무리)
                 └─ terminate → Orchestrator.render_closing()
               → [Guardrail Output Guard] → 사용자에게 전달

 [Simulation 모드]  AI User 가 응답
    UserModel → sim_step() → (동일한 Orchestrator 흐름)

  에이전트 10종:
    Orchestrator      → 중앙 허브: action 결정, intent 분석, LLM 기반 사용자 대면 텍스트 생성
    InformationSeeker → Orchestrator에 구조화된 질문 템플릿 제공
    MealRecommender   → Orchestrator에 구조화된 추천 템플릿 제공
    UserModel         → AI 사용자 응답 생성 (Simulation 모드 전용)
    MealTracker       → User 의 응답들을 구조화된 Meal Fact Sheet 로 누적 추출
    DialogSummarizer  → 대화 흐름을 서술형으로 요약
    AlignmentEstimator → 영양 목표 달성도 평가 (점수 + reasoning)
    UncertaintyEstimator → 평가 확실성 추정 (certainty 점수 + reasoning)
    Guardrail         → LLM 기반 양방향 안전 필터 (Input Guard + Output Guard)
    Memorizer         → 다중 식사 세션 간 사용자 프로필 유지

  Turn 0 :
    InformationSeeker → 고정 첫 질문 (create_session)
    User              → 응답 대기

  Turn t (≥ 1) — submit_reply() / sim_step() 내부 :
    Step 0  Guardrail         Input Guard (Custom 모드만, 사용자 입력 검증)
    Step 1  User              사용자 응답 기록 (또는 AI User 발화 생성)
    Step 2a MealTracker       User 응답을 기존 Fact Sheet 에 누적 반영
    Step 2b DialogSummarizer  대화 흐름을 서술형으로 요약 갱신
    Step 3c Orchestrator      route() → 다음 행동 결정
    Step 3  AlignmentEstimator + UncertaintyEstimator (terminate가 아닌 경우만)
    Step 4  서브 에이전트 호출 (Orchestrator 결정에 따라):
            IS.ask() → Orchestrator TextGen(질문) / MR.recommend() → Orchestrator TextGen(추천)
            / assess() → Orchestrator TextGen(피드백) / motivational_close → TextGen(동기부여 마무리)
            / render_closing()
    Step 5  Guardrail         Output Guard (Coach 응답 검증)

  종료 조건 (안전장치 — Orchestrator보다 우선) :
    - max_turns 초과            → terminated_by = "max_turns"
    - 연속 non-answer (stall)   → info_seeking/rec_info_seeking 이면 assess/recommend 강제 전환,
                                  recommending/negotiation 이면 motivational_close 강제 전환,
                                  그 외 phase 이면 terminated_by = "stall_exit"
    - User 자연 종료 토큰       → terminated_by = "user_natural"
  Orchestrator 결정에 의한 종료 :
    - Orchestrator terminate / motivational_close → terminated_by = "orchestrator"

책임 분리
  - 세션 저장소   : SessionManager (이 파일)
  - 중앙 허브     : models/orchestrator.py  (action 결정 + 텍스트 렌더링)
  - 질문 템플릿   : models/information_seeker.py
  - 추천 템플릿   : models/meal_recommender.py
  - AI 유저 응답  : models/user.py
  - 목표 달성 평가: models/alignment_estimator.py
  - 확실성 추정   : models/uncertainty_estimator.py
  - 안전 필터     : models/guardrail.py
  - 사용자 프로필 : models/memorizer.py
  - 식사 정보 추출: models/meal_tracker.py       (→ Meal Fact Sheet)
  - 대화 흐름 요약: models/dialog_summarizer.py  (→ 시스템 프롬프트)
  - LLM 추론      : utils/llm_utils.py (llama-cpp-python 백엔드)
                     utils/llm_chatgpt.py (LangGraph + ChatGPT API 백엔드)
  - 메모리 관리   : core/memory.py (SharedConversationHistory, ConversationBuffer)
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
# / Register code_interactive/ FIRST so that InformationSeeker / AlignmentEstimator's
#   `from utils.llm_utils import generate_response` resolves to the local
#   llama-cpp-python version, not the vLLM version in /code.
_INTERACTIVE_DIR = Path(__file__).resolve().parent      # code_interactive/
_CODE_DIR        = _INTERACTIVE_DIR.parent / "code"     # code/

# uvicorn -m 실행 시 code_interactive/가 이미 sys.path에 있으므로
# `if not in` 가드를 쓰면 insert(0)가 건너뛰어집니다 → 무조건 삽입합니다.
sys.path.insert(0, str(_CODE_DIR))          # code/ 등록
sys.path.insert(0, str(_INTERACTIVE_DIR))   # code_interactive/ → 최종 index 0

from core.memory import SharedConversationHistory
from models.information_seeker import InformationSeeker
from models.alignment_estimator import AlignmentEstimator
from models.user  import UserModel
from models.meal_tracker           import MealTrackerModel
from models.dialog_summarizer      import DialogSummarizerModel
from models.uncertainty_estimator   import UncertaintyEstimator, CERTAINTY_THRESHOLD
from models.orchestrator           import Orchestrator, PHASES
from models.meal_recommender       import MealRecommender
from models.guardrail              import Guardrail
from models.memorizer              import Memorizer

import re as _re
from dataclasses import replace as _dc_replace

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
    TERMINATED  = "terminated"    # 정상 종료 (Alignment Tracker 일치) / clean termination
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
    alignment_aligned:   Optional[bool]  = None  # Alignment Tracker 판정 / alignment verdict
    alignment_score:     Optional[float] = None  # 정규화 점수 / normalised score


# ─────────────────────────────────────────────────────────────────────────────
# 세션 데이터 / Session data
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Session:
    """단일 사용자 인터랙션 세션. / A single user interaction session."""

    session_id:       str
    mode:             str              # "custom" | "simulation"
    alignment_enabled:    bool             # Alignment Tracker 활성화 여부
    llm_provider:     str              # Coach/User LLM: "gemma" | "chatgpt"
    alignment_llm_provider: str            # Alignment Tracker LLM: "gemma" | "chatgpt"
    user_llm_provider:  str            # User LLM: "gemma" | "chatgpt" (simulation only)
    nutrition_goal:   str
    meal_description: str              # 음식 이름 목록 / food item names
    meal_ingredient:  str              # 재료/조리법 디테일 / ingredient details
    meal_type:        str              # breakfast / lunch / dinner / snack

    coach:   InformationSeeker
    alignment_tracker: AlignmentEstimator
    history: SharedConversationHistory

    # simulation 모드 전용 / simulation mode only
    user:    Optional[UserModel] = None

    # Coach 대화 모드 / Coach conversation mode
    coach_conversation_mode: str = "template-based"  # "open-ended" | "template-based"

    # Dialogue Summarization
    dialog_summarization: bool = True

    # Uncertainty Tracking
    uncertainty_tracking: bool = False
    uncertainty_tracker: Optional["UncertaintyEstimator"] = None

    # Orchestrator + MealRecommender
    orchestrator: Optional["Orchestrator"] = None
    meal_recommender: Optional["MealRecommender"] = None
    orchestrator_llm_provider: str = ""  # "" → llm_provider와 동일

    # Guardrail + Memorizer (new architecture)
    guardrail: Optional["Guardrail"] = None
    memorizer: Optional["Memorizer"] = None

    # 대화 Phase (new architecture)
    # info_seeking → assessment → rec_info_seeking → recommending
    #   → [accept] → motivational_ending → terminated
    #   → [reject] → negotiation → motivational_ending → terminated
    phase: str = "info_seeking"

    # 사용자 선호도 컨텍스트 (rec_info_seeking 에서 수집)
    user_preferences: str = ""

    # Stall 추적 / Stall tracking (simulation mode)
    stall_count:     int       = 0   # 연속 non-answer 수
    dead_end_topics: List[str] = field(default_factory=list)  # User가 모른다고 한 질문 목록

    # 증분 요약용: 마지막으로 처리된 턴의 다음 인덱스
    # For incremental processing: turn_idx of the first turn NOT yet included
    last_meal_track_start: int = 0     # MealTracker 용
    last_summarized_start: int = 0     # DialogSummarizer 용

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

    LLM 객체는 외부에서 한 번 로드되어 이 클래스에 주입됩니다.
     - llm          : llama-cpp Llama 객체 (Gemma GGUF)
     - chatgpt_client: ChatGPTClient 객체 (LangGraph + OpenAI API)
    / LLM objects are loaded once externally and injected into this class.
    """

    def __init__(self, llm, config, chatgpt_client=None):
        """
        Parameters
        ----------
        llm             : load_model()로 로드된 llama-cpp Llama 객체
        config          : InteractiveConfig 인스턴스
        chatgpt_client  : ChatGPTClient 인스턴스 (None이면 chatgpt 사용 불가)
        """
        self._llm     = llm
        self._chatgpt_client = chatgpt_client
        self._config  = config
        self._meal_tracker      = MealTrackerModel(model=llm, config=config)
        self._dialog_summarizer = DialogSummarizerModel(model=llm, config=config)
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
        alignment_enabled:    bool = True,
        llm_provider:     Optional[str] = None,
        user_llm_provider: Optional[str] = None,
        alignment_llm_provider: Optional[str] = None,
        coach_conversation_mode: Optional[str] = None,
        alignment_use_goal_def:  Optional[bool] = None,
        alignment_use_workflow:  Optional[bool] = None,
        alignment_output_format: Optional[str]  = None,
        uncertainty_tracking: Optional[bool] = None,
        dialog_summarization: Optional[bool] = None,
        persona_preferences: Optional[List[str]] = None,
        persona_allergies:   Optional[List[str]] = None,
        persona_restrictions: Optional[List[str]] = None,
    ) -> Session:
        """
        새 세션을 생성하고, Coach의 첫 번째 질문(turn 0)을 생성합니다.
        / Creates a new session and generates the Coach's first question (turn 0).

        Parameters
        ----------
        mode          : "custom" | "simulation"
        alignment_enabled : Alignment Tracker 평가 활성화 여부

        Returns
        -------
        Session : 초기화된 세션 (첫 질문 포함) / Initialised session with first question
        """
        session_id = str(uuid.uuid4())

        # ── LLM 프로바이더 결정 / Determine LLM provider ────────────────────
        provider = (llm_provider or self._config.llm_provider).lower()
        if provider == "chatgpt" and self._chatgpt_client is None:
            raise ValueError(
                "ChatGPT가 사용 불가합니다. .env 파일에 유효한 OPENAI_API_KEY를 입력하세요."
            )

        # User LLM 프로바이더 (미지정 시 Coach와 동일)
        user_provider = (user_llm_provider or provider).lower()
        if user_provider == "chatgpt" and self._chatgpt_client is None:
            raise ValueError(
                "User용 ChatGPT가 사용 불가합니다. .env 파일에 유효한 OPENAI_API_KEY를 입력하세요."
            )

        # Alignment Tracker LLM 프로바이더 (미지정 시 Coach/User와 동일)
        alignment_provider = (alignment_llm_provider or self._config.alignment_llm_provider).lower()
        if alignment_provider == "chatgpt" and self._chatgpt_client is None:
            raise ValueError(
                "Alignment Tracker용 ChatGPT가 사용 불가합니다. .env 파일에 유효한 OPENAI_API_KEY를 입력하세요."
            )

        # ── Alignment Tracker 옵션 오버라이드 / Alignment Tracker option overrides ────────────
        cfg = self._config
        overrides = {}
        if alignment_use_goal_def is not None:
            overrides["alignment_use_goal_def"] = alignment_use_goal_def
        if alignment_use_workflow is not None:
            overrides["alignment_use_workflow"] = alignment_use_workflow
        if alignment_output_format is not None and alignment_output_format in ("binary", "0-1", "0-100"):
            overrides["alignment_output_format"] = alignment_output_format

        # Coach 대화 모드: open-ended → ACTION_GUIDELINES 미사용
        conv_mode = (coach_conversation_mode or "template-based").lower()
        if conv_mode == "open-ended":
            overrides["coach_use_template_guidance"] = False

        if overrides:
            cfg = _dc_replace(cfg, **overrides)

        # ── 에이전트 및 히스토리 초기화 / Initialise agents and history
        # model 인자: gemma → self._llm (Llama), chatgpt → self._chatgpt_client
        # InformationSeeker/UserModel의 self.model은 _generate_single에서 직접 사용하지 않고
        # session_manager._generate_for_session()이 provider에 따라 분기합니다.
        # 그러나 AlignmentEstimator은 별도 alignment_llm_provider 를 사용합니다.
        # MealTracker/DialogSummarizer 도 _generate_single(mode="tracker") 를 통해
        # Coach/User 와 동일한 프로바이더를 사용합니다.
        _agent_model = self._llm  # agent model ref (used for get_messages only)
        history = SharedConversationHistory(context_window=cfg.context_window)
        coach   = InformationSeeker(
            model=_agent_model,
            nutrition_goal=nutrition_goal,
            meal_type=meal_type,
            config=cfg,
        )
        alignment_tracker = AlignmentEstimator(
            model=_agent_model,
            nutrition_goal=nutrition_goal,
            config=cfg,
        )

        # simulation 모드일 때만 UserModel 생성 / Create UserModel only in simulation mode
        user: Optional[UserModel] = None
        if mode == "simulation":
            user = UserModel(
                model=_agent_model,
                nutrition_goal=nutrition_goal,
                meal_description=meal_description,
                meal_ingredient=meal_ingredient,
                config=cfg,
                persona_preferences=persona_preferences,
                persona_allergies=persona_allergies,
                persona_restrictions=persona_restrictions,
            )

        session = Session(
            session_id=session_id,
            mode=mode,
            alignment_enabled=alignment_enabled,
            llm_provider=provider,
            alignment_llm_provider=alignment_provider,
            user_llm_provider=user_provider,
            nutrition_goal=nutrition_goal,
            meal_description=meal_description,
            meal_ingredient=meal_ingredient,
            meal_type=meal_type,
            coach=coach,
            alignment_tracker=alignment_tracker,
            history=history,
            user=user,
            coach_conversation_mode=conv_mode,
            dialog_summarization=(dialog_summarization if dialog_summarization is not None else True),
            uncertainty_tracking=bool(uncertainty_tracking),
            uncertainty_tracker=(
                UncertaintyEstimator(nutrition_goal=nutrition_goal, config=cfg)
                if uncertainty_tracking else None
            ),
            orchestrator=Orchestrator(nutrition_goal=nutrition_goal, config=cfg),
            meal_recommender=MealRecommender(nutrition_goal=nutrition_goal, config=cfg),
            orchestrator_llm_provider=(
                cfg.orchestrator_llm_provider if cfg.orchestrator_llm_provider else provider
            ),
            guardrail=Guardrail(config=cfg),
            memorizer=Memorizer(),
            phase="info_seeking",
        )

        # ── Memorizer: 페르소나 설정에서 초기 프로필 세팅
        # / Initialize Memorizer profile from persona settings
        if persona_preferences or persona_allergies or persona_restrictions:
            session.memorizer.set_profile_from_persona(
                preferences=persona_preferences,
                allergies=persona_allergies,
                restrictions=persona_restrictions,
            )

        # ── Turn 0: Coach 첫 질문 생성 / Generate Coach's first question
        first_q = coach.first_question()
        history.add_turn(turn_idx=0, coach_utterance=first_q)
        session.turns.append(TurnRecord(turn_idx=0, coach_utterance=first_q))

        with self._lock:
            self._sessions[session_id] = session

        return session

    # ── 세션 이어하기 (Multi-meal) / Continue session for next meal ───────

    def continue_session(
        self,
        previous_session_id: str,
        nutrition_goal:   str,
        meal_description: str,
        meal_ingredient:  str,
        meal_type:        str = "meal",
    ) -> Session:
        """
        이전 세션의 Memorizer(사용자 프로필/식사 이력)를 이어받아 새 식사 세션을 생성합니다.
        mode, LLM 프로바이더, 페르소나 설정은 이전 세션에서 자동 이어받습니다.
        / Creates a new meal session carrying over the Memorizer from a previous session.
        Mode, LLM providers, and persona settings are inherited automatically.

        Parameters
        ----------
        previous_session_id : 이전 세션 ID (Memorizer 이어받기 용도)
        nutrition_goal      : 새 식사의 영양 목표
        meal_description    : 새 식사의 음식 이름
        meal_ingredient     : 새 식사의 재료/조리법
        meal_type           : breakfast / lunch / dinner / snack

        Returns
        -------
        Session : 새 세션 (이전 Memorizer 포함, 첫 질문 포함)
        """
        with self._lock:
            prev_session = self._sessions.get(previous_session_id)
        if prev_session is None:
            raise KeyError(f"Previous session not found: {previous_session_id}")

        # 이전 세션에서 페르소나 데이터 추출 (simulation 모드 UserModel에서)
        # / Extract persona data from previous session's UserModel
        persona_prefs = None
        persona_allergy = None
        persona_restrict = None
        if prev_session.user is not None:
            persona_prefs   = prev_session.user.persona_preferences or None
            persona_allergy = prev_session.user.persona_allergies or None
            persona_restrict = prev_session.user.persona_restrictions or None

        # 이전 세션의 설정을 이어받기 / Carry over settings from previous session
        new_session = self.create_session(
            nutrition_goal=nutrition_goal,
            meal_description=meal_description,
            meal_ingredient=meal_ingredient,
            meal_type=meal_type,
            mode=prev_session.mode,
            alignment_enabled=prev_session.alignment_enabled,
            llm_provider=prev_session.llm_provider,
            user_llm_provider=prev_session.user_llm_provider,
            alignment_llm_provider=prev_session.alignment_llm_provider,
            coach_conversation_mode=prev_session.coach_conversation_mode,
            uncertainty_tracking=prev_session.uncertainty_tracking,
            dialog_summarization=prev_session.dialog_summarization,
            persona_preferences=persona_prefs,
            persona_allergies=persona_allergy,
            persona_restrictions=persona_restrict,
        )

        # Memorizer 이어받기 / Carry over Memorizer
        if prev_session.memorizer:
            new_session.memorizer = prev_session.memorizer

        return new_session

    # ── 사용자 응답 처리 / Process user reply ────────────────────────────────

    def submit_reply(self, session_id: str, user_reply: str) -> Dict[str, Any]:
        """
        사용자의 응답을 받아 다음 Coach 질문과 Alignment Tracker 판정을 반환합니다.
        / Receives user reply, returns next Coach question and Alignment Tracker verdict.

        Returns
        -------
        dict with keys:
            turn_idx       : 현재 턴 인덱스 / current turn index
            coach_question : 다음 Coach 질문 (종료 시 None) / next coach question
            alignment_aligned  : Alignment Tracker 판정 (아직 판정 전이면 None) / alignment verdict
            alignment_score    : 정규화 점수 / normalised score
            status         : 세션 상태 / session status
            aligned_label  : 판정 레이블 문자열 / alignment label string
        """
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}")
        if session.status != SessionStatus.ACTIVE:
            raise ValueError(f"Session {session_id} is already {session.status}")

        # ChatGPT 디스패치: 세션 LLM provider 에 따라 chatgpt_client 결정
        _gpt = self._chatgpt_client if session.llm_provider == "chatgpt" else None
        _gpt_alignment = self._chatgpt_client if session.alignment_llm_provider == "chatgpt" else None
        _guardrail_provider = (self._config.guardrail_llm_provider or session.llm_provider)
        _gpt_guardrail = self._chatgpt_client if _guardrail_provider == "chatgpt" else None

        # ── Guardrail Input Guard: 사용자 입력 검증 / Validate user input
        if session.guardrail:
            _ig_msgs = session.guardrail.get_input_guard_messages(user_reply.strip())
            _ig_raw = _generate_single(
                self._llm, _ig_msgs, self._config,
                mode="guardrail", chatgpt_client=_gpt_guardrail,
            )
            _ig_result = session.guardrail.parse_input_guard(_ig_raw)
            if not _ig_result["passed"]:
                _redirect = (
                    _ig_result["message"]
                    or "Let's keep our conversation focused on your meal. "
                       "Could you tell me about what you're eating?"
                )
                return {
                    "turn_idx":       session.turn_idx,
                    "coach_question":  _redirect,
                    "guardrail_blocked": True,
                    "alignment_aligned":   None,
                    "alignment_score":     None,
                    "alignment_reasoning": None,
                    "alignment_input":     None,
                    "alignment_raw_output": None,
                    "certainty_score":    None,
                    "certainty_reasoning": None,
                    "meal_tracker_input":  None,
                    "meal_tracker_output": None,
                    "certainty_input":     None,
                    "certainty_output":    None,
                    "orchestrator_decision": None,
                    "recommendation_result": None,
                    "assessment_result":   None,
                    "phase":               session.phase,
                    "dialog_summary":  session.history.dialog_summary or None,
                    "meal_fact_sheet": session.history.meal_fact_sheet or None,
                    "status":          session.status.value,
                    "terminated_by":   None,
                    "aligned_label":   "pending",
                }

        # ── Step 1: 사용자 응답을 히스토리에 기록 / Record user reply in history
        session.history.update_last_user_utterance(user_reply.strip())
        session.turns[-1].user_utterance = user_reply.strip()

        # ── Memorizer: rec_info_seeking 페이즈에서 사용자 선호도 수집
        # / Accumulate user preferences during rec_info_seeking phase
        if session.phase == "rec_info_seeking" and user_reply.strip():
            _pref_text = user_reply.strip()
            if session.user_preferences:
                session.user_preferences += "\n" + _pref_text
            else:
                session.user_preferences = _pref_text
            if session.memorizer:
                session.memorizer.update_preferences([_pref_text])

        turn_idx = session.turn_idx
        alignment_aligned: Optional[bool]  = None
        alignment_score:   Optional[float] = None
        alignment_reasoning: Optional[str]  = None
        alignment_input_ctx:  Optional[str] = None
        alignment_raw_output: Optional[str] = None
        certainty_score:    Optional[float] = None
        certainty_reasoning: Optional[str]  = None
        meal_tracker_input:   Optional[str] = None
        meal_tracker_output:  Optional[str] = None
        certainty_input:      Optional[str] = None
        certainty_output:     Optional[str] = None
        orchestrator_input:   Optional[str] = None
        orchestrator_raw_output: Optional[str] = None

        # Output Guard 재생성용 TextGen 컨텍스트 / For Output Guard re-prompt
        _last_tg_msgs: Optional[list] = None
        _last_tg_template: Optional[dict] = None
        _assessment_message: Optional[str] = None

        # ── Step 2a: MealTracker — Meal Fact Sheet 갱신 (매 meal_track_every 턴)
        # Runs BEFORE Alignment Tracker so it sees the freshest Meal Fact Sheet
        completed = turn_idx + 1
        if completed % self._config.meal_track_every == 0:
            _mt_new = session.history.to_plain_text_from(session.last_meal_track_start)
            _mt_msgs = self._meal_tracker.get_messages(_mt_new, prev_fact_sheet=session.history.meal_fact_sheet)
            meal_tracker_input = _mt_msgs[1]["content"] if len(_mt_msgs) > 1 else ""
            _mt_result = _generate_single(self._llm, _mt_msgs, self._config, mode="tracker", chatgpt_client=_gpt)
            meal_tracker_output = _mt_result
            session.history.update_meal_fact_sheet(_mt_result)
            session.last_meal_track_start = completed

        # ── Step 2b: DialogSummarizer — 대화 요약 갱신 (매 summarize_every 턴)
        if session.dialog_summarization and completed % self._config.summarize_every == 0:
            _ds_new = session.history.to_plain_text_from(session.last_summarized_start)
            _ds_msgs = self._dialog_summarizer.get_messages(_ds_new, prev_summary=session.history.dialog_summary)
            session.history.update_dialog_summary(
                _generate_single(self._llm, _ds_msgs, self._config, mode="tracker", chatgpt_client=_gpt)
            )
            session.last_summarized_start = completed

        # ── Step 3: Estimator Bundle 은 Orchestrator 결정 이후 필요 시에만 실행
        # Estimator Bundle runs ONLY when Orchestrator decides IS/Recommender/Assessment
        # → terminate 시에는 실행하지 않아 레이턴시 절감

        # ── Stall 감지 / Detect stall (Custom Chat 에서도 non-answer 추적)
        _stall_exit_turns = getattr(self._config, 'stall_exit_turns', 3)
        if _is_non_answer(user_reply.strip()):
            session.stall_count += 1
            if session.turns and session.turns[-1].coach_utterance:
                session.dead_end_topics.append(session.turns[-1].coach_utterance)
        else:
            session.stall_count = 0

        stall_exit_now = (
            session.status == SessionStatus.ACTIVE
            and session.stall_count >= _stall_exit_turns
        )

        # ── Step 3c: Orchestrator Router — 다음 행동 결정 / Decide next action
        orchestrator_decision: Optional[dict] = None
        recommendation_result: Optional[dict] = None
        assessment_result: Optional[dict] = None
        next_question: Optional[str] = None
        _gpt_orch = self._chatgpt_client if session.orchestrator_llm_provider == "chatgpt" else None

        if session.status == SessionStatus.ACTIVE:
            next_turn = turn_idx + 1

            # 안전장치: max_turns 도달 시 강제 종료 (Orchestrator보다 우선)
            if next_turn >= self._config.max_turns:
                session.status        = SessionStatus.MAX_TURNS
                session.terminated_by = "max_turns"
                session.final_aligned = alignment_aligned
                session.final_score   = alignment_score
            else:
                # ── stall_exit 이면 Orchestrator Router LLM 호출 없이 강제 action 결정
                _stall_forced = False
                if stall_exit_now and session.phase in ("info_seeking", "rec_info_seeking", "recommending", "negotiation"):
                    _stall_forced = True
                    session.stall_count = 0
                    if session.phase == "info_seeking":
                        orchestrator_decision = {
                            "action": "assess_meal",
                            "reasoning": "(stall-exit: user cannot provide more info → assess with available data)",
                            "instruction": "",
                        }
                    elif session.phase == "rec_info_seeking":
                        orchestrator_decision = {
                            "action": "recommend",
                            "reasoning": "(stall-exit: user cannot provide preferences → recommend without them)",
                            "instruction": "",
                        }
                    else:  # recommending / negotiation
                        orchestrator_decision = {
                            "action": "motivational_close",
                            "reasoning": "(stall-exit: user unresponsive in recommendation/negotiation → wrap up with motivation)",
                            "instruction": "",
                        }
                elif stall_exit_now:
                    # 이미 핵심 흐름을 마친 phase에서 stall → 종료
                    # (info_seeking/rec_info_seeking/recommending/negotiation만 강제 전환 대상)
                    session.turn_idx = next_turn
                    _fallback_close = "Thanks for sharing all of that — I think I have a good picture of your meal!"
                    session.coach.own_buffer.add(_fallback_close)
                    session.history.add_turn(turn_idx=next_turn, coach_utterance=_fallback_close)
                    session.turns.append(TurnRecord(turn_idx=next_turn, coach_utterance=_fallback_close))
                    next_question = _fallback_close
                    session.status        = SessionStatus.TERMINATED
                    session.terminated_by = "stall_exit"
                    session.final_aligned = alignment_aligned
                    session.final_score   = alignment_score

                # ── Orchestrator Router 호출 (stall 강제 전환이 아닌 경우만)
                if not _stall_forced and session.status == SessionStatus.ACTIVE:
                    _rec_history = (
                        session.meal_recommender.recommendation_history
                        if session.meal_recommender else []
                    )
                    _orch_msgs = session.orchestrator.get_routing_messages(
                        history=session.history,
                        turn_idx=turn_idx,
                        phase=session.phase,
                        recommendation_history=_rec_history,
                    )
                    orchestrator_input = _orch_msgs[1]["content"] if len(_orch_msgs) > 1 else ""
                    _orch_raw = _generate_single(
                        self._llm, _orch_msgs, self._config,
                        mode="orchestrator", chatgpt_client=_gpt_orch,
                    )
                    orchestrator_raw_output = _orch_raw
                    if not _orch_raw or not _orch_raw.strip():
                        print(f"[Router] WARNING: empty output at turn {turn_idx}, phase={session.phase}")
                    orchestrator_decision = session.orchestrator.parse_routing(
                        _orch_raw, turn_idx=turn_idx, phase=session.phase,
                    )

                # ── Orchestrator 결정 실행
                if orchestrator_decision and session.status == SessionStatus.ACTIVE:
                    action = orchestrator_decision.get("action", "seek_meal_info")
                    orch_instruction = orchestrator_decision.get("instruction", "")

                    # ── Estimator Bundle: IS/Recommender/Assessment/Motivational 에만 실행
                    needs_estimators = action in (
                        "seek_meal_info", "seek_recommendation_info",
                        "assess_meal", "recommend", "motivational_close",
                    )
                    if needs_estimators:
                        alignment_aligned, alignment_score, alignment_reasoning, \
                            alignment_input_ctx, alignment_raw_output = \
                            _run_alignment(self, session, _gpt_alignment, turn_idx)
                        certainty_score, certainty_reasoning, certainty_input, certainty_output = \
                            _run_uncertainty(self, session, _gpt, turn_idx)

                    # ── Step 4: Orchestrator 결정에 따른 분기
                    if action == "terminate":
                        session.turn_idx = next_turn
                        next_question = session.orchestrator.render_closing(orch_instruction)
                        session.coach.own_buffer.add(next_question)
                        session.history.add_turn(turn_idx=next_turn, coach_utterance=next_question)
                        session.turns.append(TurnRecord(turn_idx=next_turn, coach_utterance=next_question))
                        session.status        = SessionStatus.TERMINATED
                        session.terminated_by = "orchestrator"
                        session.final_aligned = alignment_aligned
                        session.final_score   = alignment_score
                        session.phase         = "terminated"

                    elif action == "motivational_close":
                        # ── Motivational Ending: assessment + health tip + encouraging close
                        _assess_msgs = session.orchestrator.get_assessment_messages(
                            history=session.history,
                            alignment_score=alignment_score,
                            alignment_reasoning=alignment_reasoning,
                        )
                        _assess_raw = _generate_single(
                            self._llm, _assess_msgs, self._config,
                            mode="orchestrator", chatgpt_client=_gpt_orch,
                        )
                        assessment_result = session.orchestrator.parse_assessment(_assess_raw)

                        _tg_motiv_msgs = session.orchestrator.get_textgen_motivational_messages(
                            assessment_result, session.history,
                        )
                        _tg_motiv_raw = _generate_single(
                            self._llm, _tg_motiv_msgs, self._config,
                            mode="orchestrator", chatgpt_client=_gpt_orch,
                        )
                        next_question = session.orchestrator.parse_textgen(
                            _tg_motiv_raw, assessment_result,
                        )
                        if not next_question.strip():
                            next_question = session.orchestrator.render_motivational_close(assessment_result)

                        _last_tg_msgs = _tg_motiv_msgs
                        _last_tg_template = assessment_result
                        session.turn_idx = next_turn
                        session.coach.own_buffer.add(next_question)
                        session.history.add_turn(turn_idx=next_turn, coach_utterance=next_question)
                        session.turns.append(TurnRecord(turn_idx=next_turn, coach_utterance=next_question))
                        session.phase         = "motivational_ending"
                        session.status        = SessionStatus.TERMINATED
                        session.terminated_by = "orchestrator"
                        session.final_aligned = alignment_aligned
                        session.final_score   = alignment_score

                    elif action == "assess_meal":
                        _assess_msgs = session.orchestrator.get_assessment_messages(
                            history=session.history,
                            alignment_score=alignment_score,
                            alignment_reasoning=alignment_reasoning,
                        )
                        _assess_raw = _generate_single(
                            self._llm, _assess_msgs, self._config,
                            mode="orchestrator", chatgpt_client=_gpt_orch,
                        )
                        assessment_result = session.orchestrator.parse_assessment(_assess_raw)

                        overall = assessment_result.get("overall", "partially_aligned")
                        _needs_rec = (overall != "aligned")
                        # Orchestrator TextGen: Assessment 결과 → 순수 피드백 (선호도 질문 미포함)
                        _tg_assess_msgs = session.orchestrator.get_textgen_assessment_messages(
                            assessment_result, needs_recommendation=_needs_rec,
                            history=session.history,
                        )
                        _tg_assess_raw = _generate_single(
                            self._llm, _tg_assess_msgs, self._config,
                            mode="orchestrator", chatgpt_client=_gpt_orch,
                        )
                        _assessment_text = session.orchestrator.parse_textgen(
                            _tg_assess_raw, assessment_result,
                        )
                        # Record assessment as Turn N
                        session.turn_idx = next_turn
                        session.coach.own_buffer.add(_assessment_text)
                        session.history.add_turn(turn_idx=next_turn, coach_utterance=_assessment_text)
                        session.turns.append(TurnRecord(turn_idx=next_turn, coach_utterance=_assessment_text))
                        if overall == "aligned":
                            next_question = _assessment_text
                            session.status        = SessionStatus.TERMINATED
                            session.terminated_by = "orchestrator"
                            session.final_aligned = True
                            session.final_score   = alignment_score
                            session.phase         = "terminated"
                        else:
                            session.phase = "rec_info_seeking"
                            _assessment_message = _assessment_text
                            # ── Double-turn: 즉시 선호도 질문 생성 (사용자 대기 없이)
                            _pref_turn = next_turn + 1
                            _pref_is_msgs = session.coach.get_messages(
                                session.history,
                                dead_end_topics=session.dead_end_topics if session.dead_end_topics else None,
                                mode="recommendation_info",
                            )
                            _pref_is_raw = _generate_single(
                                self._llm, _pref_is_msgs, self._config,
                                mode="coach", chatgpt_client=_gpt,
                            )
                            _pref_template = session.coach._parse_template(_pref_is_raw)
                            _pref_tg_msgs = session.orchestrator.get_textgen_question_messages(
                                _pref_template, session.history,
                            )
                            _pref_tg_raw = _generate_single(
                                self._llm, _pref_tg_msgs, self._config,
                                mode="orchestrator", chatgpt_client=_gpt_orch,
                            )
                            next_question = session.orchestrator.parse_textgen(
                                _pref_tg_raw, _pref_template,
                            )
                            _last_tg_msgs = _pref_tg_msgs
                            _last_tg_template = _pref_template
                            # Record preference question as Turn N+1
                            session.turn_idx = _pref_turn
                            session.coach.own_buffer.add(next_question)
                            session.history.add_turn(turn_idx=_pref_turn, coach_utterance=next_question)
                            session.turns.append(TurnRecord(turn_idx=_pref_turn, coach_utterance=next_question))

                    elif action == "recommend":
                        _user_prefs = ""
                        if session.memorizer:
                            _user_prefs = session.memorizer.get_preferences_text()
                        if session.user_preferences:
                            _user_prefs = (_user_prefs + "\n" + session.user_preferences).strip()

                        # 이전 추천 이력 + 사용자 피드백 수집 / Collect recommendation history + user feedback
                        _rec_history = session.meal_recommender.recommendation_history
                        _user_feedback = ""
                        if _rec_history and session.turns:
                            for t in reversed(session.turns):
                                if t.user_utterance:
                                    _user_feedback = f"Turn {t.turn_idx}: {t.user_utterance}"
                                    break

                        _rec_msgs = session.meal_recommender.get_messages(
                            meal_fact_sheet=session.history.meal_fact_sheet or "",
                            alignment_score=alignment_score if alignment_score is not None else 0.0,
                            alignment_reasoning=alignment_reasoning or "",
                            instruction=orch_instruction,
                            user_preferences=_user_prefs,
                            recommendation_history=_rec_history,
                            user_feedback=_user_feedback,
                        )
                        _rec_raw = _generate_single(
                            self._llm, _rec_msgs, self._config,
                            mode="recommender", chatgpt_client=_gpt,
                        )
                        recommendation_result = session.meal_recommender.parse_output(
                            _rec_raw, turn_idx=turn_idx,
                        )

                        # Orchestrator TextGen: MR 추천 템플릿 → 자연어 추천
                        _tg_msgs = session.orchestrator.get_textgen_recommendation_messages(
                            recommendation_result, session.history,
                            recommendation_history=_rec_history,
                        )
                        _tg_raw = _generate_single(
                            self._llm, _tg_msgs, self._config,
                            mode="orchestrator", chatgpt_client=_gpt_orch,
                        )
                        next_question = session.orchestrator.parse_textgen(
                            _tg_raw, recommendation_result,
                        )
                        _last_tg_msgs = _tg_msgs
                        _last_tg_template = recommendation_result
                        session.turn_idx = next_turn
                        session.coach.own_buffer.add(next_question)
                        session.history.add_turn(turn_idx=next_turn, coach_utterance=next_question)
                        session.turns.append(TurnRecord(turn_idx=next_turn, coach_utterance=next_question))
                        # negotiation 중이면 phase 유지, 아니면 recommending
                        if session.phase != "negotiation":
                            session.phase = "recommending"

                    elif action == "seek_recommendation_info":
                        is_msgs = session.coach.get_messages(
                            session.history,
                            dead_end_topics=session.dead_end_topics if session.dead_end_topics else None,
                            mode="recommendation_info",
                        )
                        _is_raw = _generate_single(
                            self._llm, is_msgs, self._config, mode="coach", chatgpt_client=_gpt,
                        )
                        is_template = session.coach._parse_template(_is_raw)
                        # Orchestrator TextGen: IS 질문 템플릿 → 자연어 질문
                        _tg_msgs = session.orchestrator.get_textgen_question_messages(
                            is_template, session.history,
                        )
                        _tg_raw = _generate_single(
                            self._llm, _tg_msgs, self._config,
                            mode="orchestrator", chatgpt_client=_gpt_orch,
                        )
                        next_question = session.orchestrator.parse_textgen(
                            _tg_raw, is_template,
                        )
                        _last_tg_msgs = _tg_msgs
                        _last_tg_template = is_template

                        session.turn_idx = next_turn
                        session.coach.own_buffer.add(next_question)
                        session.history.add_turn(turn_idx=next_turn, coach_utterance=next_question)
                        session.turns.append(TurnRecord(turn_idx=next_turn, coach_utterance=next_question))
                        # recommending에서 seek_recommendation_info가 왔으면 negotiation으로 전환
                        if session.phase == "recommending":
                            session.phase = "negotiation"

                    else:
                        # action == "seek_meal_info" (기본값)
                        is_msgs = session.coach.get_messages(
                            session.history,
                            dead_end_topics=session.dead_end_topics if session.dead_end_topics else None,
                            mode="meal_info",
                        )
                        _is_raw = _generate_single(
                            self._llm, is_msgs, self._config, mode="coach", chatgpt_client=_gpt,
                        )
                        is_template = session.coach._parse_template(_is_raw)
                        # Orchestrator TextGen: IS 질문 템플릿 → 자연어 질문
                        _tg_msgs = session.orchestrator.get_textgen_question_messages(
                            is_template, session.history,
                        )
                        _tg_raw = _generate_single(
                            self._llm, _tg_msgs, self._config,
                            mode="orchestrator", chatgpt_client=_gpt_orch,
                        )
                        next_question = session.orchestrator.parse_textgen(
                            _tg_raw, is_template,
                        )

                        # 중복 질문 탐지 + 재시도
                        _already_asked = session.history.get_all_coach_questions()
                        if _is_duplicate_question(next_question, _already_asked):
                            _retry_msgs = is_msgs + [{
                                "role": "user",
                                "content": (
                                    "[SYSTEM NOTE: The question you just generated was already asked. "
                                    "Please ask about a completely different food item or a new aspect "
                                    "that has NOT yet been covered in this conversation.]"
                                ),
                            }]
                            _retry_raw = _generate_single(
                                self._llm, _retry_msgs, self._config, mode="coach", chatgpt_client=_gpt,
                            )
                            _retry_tmpl = session.coach._parse_template(_retry_raw)
                            _tg_retry_msgs = session.orchestrator.get_textgen_question_messages(
                                _retry_tmpl, session.history,
                            )
                            _tg_retry_raw = _generate_single(
                                self._llm, _tg_retry_msgs, self._config,
                                mode="orchestrator", chatgpt_client=_gpt_orch,
                            )
                            _retry_q = session.orchestrator.parse_textgen(
                                _tg_retry_raw, _retry_tmpl,
                            )
                            if _retry_q.strip():
                                next_question = _retry_q
                                _tg_msgs = _tg_retry_msgs
                                is_template = _retry_tmpl

                        _fallback = "Could you tell me more about your meal?"
                        next_question = next_question.replace(
                            SharedConversationHistory.TERMINATION_TOKEN, ""
                        ).strip() or _fallback
                        _last_tg_msgs = _tg_msgs
                        _last_tg_template = is_template

                        session.turn_idx = next_turn
                        session.coach.own_buffer.add(next_question)
                        session.history.add_turn(turn_idx=next_turn, coach_utterance=next_question)
                        session.turns.append(TurnRecord(turn_idx=next_turn, coach_utterance=next_question))

        # ── Guardrail Output Guard: Coach 응답 검증 / Validate coach response
        if session.guardrail and next_question:
            _og_msgs = session.guardrail.get_output_guard_messages(next_question)
            _og_raw = _generate_single(
                self._llm, _og_msgs, self._config,
                mode="guardrail", chatgpt_client=_gpt_guardrail,
            )
            _og_result = session.guardrail.parse_output_guard(_og_raw)
            if not _og_result["passed"]:
                _regenerated = False
                _guard_reason = _og_result.get("reason", "")
                # Re-prompt Orchestrator TextGen with guard feedback (max 2 retries)
                if _last_tg_msgs and _last_tg_template:
                    for _retry_i in range(2):
                        _retry_tg_msgs = _last_tg_msgs + [{
                            "role": "user",
                            "content": (
                                "[SAFETY NOTE: Your previous response was flagged. "
                                f"Reason: {_guard_reason}. "
                                "Please rephrase while keeping the conversation focused on "
                                "the user's meal and nutrition. Avoid medical advice, "
                                "calorie prescriptions, or any content outside meal coaching.]"
                            ),
                        }]
                        _retry_raw = _generate_single(
                            self._llm, _retry_tg_msgs, self._config,
                            mode="orchestrator", chatgpt_client=_gpt_orch,
                        )
                        _retry_q = session.orchestrator.parse_textgen(
                            _retry_raw, _last_tg_template,
                        )
                        _og2_msgs = session.guardrail.get_output_guard_messages(_retry_q)
                        _og2_raw = _generate_single(
                            self._llm, _og2_msgs, self._config,
                            mode="guardrail", chatgpt_client=_gpt_guardrail,
                        )
                        _og2_result = session.guardrail.parse_output_guard(_og2_raw)
                        if _og2_result["passed"]:
                            next_question = _retry_q
                            _regenerated = True
                            break
                        _guard_reason = _og2_result.get("reason", _guard_reason)
                if not _regenerated:
                    _safe = "Could you tell me more about what you're having for your meal?"
                    next_question = _safe
                if session.turns:
                    session.turns[-1].coach_utterance = next_question

        # ── 종료 시 최종 요약 / Final summary on termination
        if session.status != SessionStatus.ACTIVE:
            _mt_final = session.history.to_plain_text_from(session.last_meal_track_start)
            if _mt_final:
                _mt_msgs = self._meal_tracker.get_messages(_mt_final, prev_fact_sheet=session.history.meal_fact_sheet)
                session.history.update_meal_fact_sheet(
                    _generate_single(self._llm, _mt_msgs, self._config, mode="tracker", chatgpt_client=_gpt)
                )
            _ds_final = session.history.to_plain_text_from(session.last_summarized_start)
            if _ds_final and session.dialog_summarization:
                _ds_msgs = self._dialog_summarizer.get_messages(_ds_final, prev_summary=session.history.dialog_summary)
                session.history.update_dialog_summary(
                    _generate_single(self._llm, _ds_msgs, self._config, mode="tracker", chatgpt_client=_gpt)
                )
            # Memorizer: 종료 시 식사 요약을 past_meals 에 저장
            if session.memorizer and session.history.meal_fact_sheet:
                session.memorizer.add_past_meal(
                    meal_type=session.meal_type,
                    summary=session.history.dialog_summary or "",
                    fact_sheet=session.history.meal_fact_sheet or "",
                )

        return {
            "turn_idx":       turn_idx,
            "coach_question":  next_question,
            "guardrail_blocked": False,
            "alignment_aligned":   alignment_aligned,
            "alignment_score":     alignment_score,
            "alignment_reasoning": alignment_reasoning,
            "alignment_input":     alignment_input_ctx,
            "alignment_raw_output": alignment_raw_output,
            "certainty_score":    certainty_score,
            "certainty_reasoning": certainty_reasoning,
            "meal_tracker_input":  meal_tracker_input,
            "meal_tracker_output": meal_tracker_output,
            "certainty_input":     certainty_input,
            "certainty_output":    certainty_output,
            "orchestrator_decision": orchestrator_decision,
            "orchestrator_input":  orchestrator_input,
            "orchestrator_raw_output": orchestrator_raw_output,
            "recommendation_result": recommendation_result,
            "assessment_result":   assessment_result,
            "assessment_message":  _assessment_message,
            "phase":               session.phase,
            "dialog_summary":  session.history.dialog_summary or None,
            "meal_fact_sheet": session.history.meal_fact_sheet or None,
            "status":          session.status.value,
            "terminated_by":   session.terminated_by,
            "aligned_label":   _alignment_label(alignment_aligned),
        }

    # ── Simulation 스텝 / Simulation step ───────────────────────────────────

    def sim_step(self, session_id: str) -> Dict[str, Any]:
        """
        Simulation 모드 전용: AI User가 한 번 응답하고 Alignment Tracker 평가 후 다음 Coach 질문을 반환합니다.
        / Simulation mode only: AI User responds, Alignment Tracker evaluates, returns next Coach question.

        Returns
        -------
        dict with keys:
            turn_idx       : 현재 턴 / current turn index
            user_reply     : AI User 발화 / AI user utterance
            coach_question : 다음 Coach 질문 (종료 시 None) / next coach question
            alignment_aligned  : Alignment Tracker 판정 / alignment verdict
            alignment_score    : 정규화 점수 / normalised score
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

        # ChatGPT 디스패치: 세션 LLM provider 에 따라 chatgpt_client 결정
        _gpt = self._chatgpt_client if session.llm_provider == "chatgpt" else None
        _gpt_user = self._chatgpt_client if session.user_llm_provider == "chatgpt" else None
        _gpt_alignment = self._chatgpt_client if session.alignment_llm_provider == "chatgpt" else None
        _guardrail_provider = (self._config.guardrail_llm_provider or session.llm_provider)
        _gpt_guardrail = self._chatgpt_client if _guardrail_provider == "chatgpt" else None

        # ── Step 1: User Agent — AI User 응답 생성 / Generate AI User response
        user_msgs   = session.user.get_messages(session.history)
        user_reply  = _generate_single(self._llm, user_msgs, self._config, mode="user", chatgpt_client=_gpt_user)

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

        # ── Memorizer: rec_info_seeking 페이즈에서 사용자 선호도 수집
        # / Accumulate user preferences during rec_info_seeking phase
        if session.phase == "rec_info_seeking" and user_reply_clean:
            if session.user_preferences:
                session.user_preferences += "\n" + user_reply_clean
            else:
                session.user_preferences = user_reply_clean
            if session.memorizer:
                session.memorizer.update_preferences([user_reply_clean])

        turn_idx      = session.turn_idx

        # 최소 턴 수 미달 시 자연 종료를 무시 — 너무 이른 시점에 TERMINATION_TOKEN이 생성된 경우 억제
        # Suppress natural_end if the conversation has not yet reached the minimum required turns
        _min_natural_end_turn = getattr(self._config, 'min_natural_end_turn', 3)
        if natural_end and turn_idx < _min_natural_end_turn:
            natural_end = False

        alignment_aligned: Optional[bool]  = None
        alignment_score:   Optional[float] = None
        alignment_reasoning: Optional[str]  = None
        alignment_input_ctx:  Optional[str] = None
        alignment_raw_output: Optional[str] = None
        certainty_score:    Optional[float] = None
        certainty_reasoning: Optional[str]  = None
        meal_tracker_input:   Optional[str] = None
        meal_tracker_output:  Optional[str] = None
        certainty_input:      Optional[str] = None
        certainty_output:     Optional[str] = None
        orchestrator_input:   Optional[str] = None
        orchestrator_raw_output: Optional[str] = None

        # Output Guard 재생성용 TextGen 컨텍스트 / For Output Guard re-prompt
        _last_tg_msgs: Optional[list] = None
        _last_tg_template: Optional[dict] = None
        _assessment_message: Optional[str] = None

        # ── Step 2a: MealTracker — Meal Fact Sheet 갱신 (매 meal_track_every 턴)
        # Runs BEFORE Alignment Tracker so it sees the freshest Meal Fact Sheet
        completed = turn_idx + 1
        if completed % self._config.meal_track_every == 0:
            _mt_new = session.history.to_plain_text_from(session.last_meal_track_start)
            _mt_msgs = self._meal_tracker.get_messages(_mt_new, prev_fact_sheet=session.history.meal_fact_sheet)
            meal_tracker_input = _mt_msgs[1]["content"] if len(_mt_msgs) > 1 else ""
            _mt_result = _generate_single(self._llm, _mt_msgs, self._config, mode="tracker", chatgpt_client=_gpt)
            meal_tracker_output = _mt_result
            session.history.update_meal_fact_sheet(_mt_result)
            session.last_meal_track_start = completed

        # ── Step 2b: DialogSummarizer — 대화 요약 갱신 (매 summarize_every 턴)
        if session.dialog_summarization and completed % self._config.summarize_every == 0:
            _ds_new = session.history.to_plain_text_from(session.last_summarized_start)
            _ds_msgs = self._dialog_summarizer.get_messages(_ds_new, prev_summary=session.history.dialog_summary)
            session.history.update_dialog_summary(
                _generate_single(self._llm, _ds_msgs, self._config, mode="tracker", chatgpt_client=_gpt)
            )
            session.last_summarized_start = completed

        # ── Step 3: Estimator Bundle 은 Orchestrator 결정 이후 필요 시에만 실행
        # Estimator Bundle runs ONLY when Orchestrator decides IS/Recommender/Assessment
        # → terminate 시에는 실행하지 않아 레이턴시 절감

        # ── 자연 종료: Orchestrator에게 신호로 전달 (직접 종료하지 않음)
        # Natural end: passed as a signal to Orchestrator (not terminated directly)
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

        # ── Step 3c: Orchestrator Router — 다음 행동 결정 / Decide next action
        orchestrator_decision: Optional[dict] = None
        recommendation_result: Optional[dict] = None
        assessment_result: Optional[dict] = None
        next_question: Optional[str] = None
        _gpt_orch = self._chatgpt_client if session.orchestrator_llm_provider == "chatgpt" else None

        if session.status == SessionStatus.ACTIVE:
            next_turn = turn_idx + 1

            # 안전장치: max_turns 도달 시 강제 종료 (Orchestrator보다 우선)
            if next_turn >= self._config.max_turns:
                session.status        = SessionStatus.MAX_TURNS
                session.terminated_by = "max_turns"
                session.final_aligned = alignment_aligned
                session.final_score   = alignment_score
            # 안전장치: natural_end — Coach 마무리 발화 후 종료
            elif closing_for_natural:
                session.turn_idx = next_turn
                _fallback_close = "Thanks for sharing all of that — I think I have a good picture of your meal!"
                next_question = _fallback_close
                session.coach.own_buffer.add(next_question)
                session.history.add_turn(turn_idx=next_turn, coach_utterance=next_question)
                session.turns.append(TurnRecord(turn_idx=next_turn, coach_utterance=next_question))
                session.status        = SessionStatus.TERMINATED
                session.terminated_by = "user_natural"
                session.final_aligned = alignment_aligned
                session.final_score   = alignment_score
            else:
                # ── stall_exit 이면 Orchestrator Router LLM 호출 없이 강제 action 결정
                # Stall → force assess/recommend transition instead of terminating
                _stall_forced = False
                if stall_exit_now and session.phase in ("info_seeking", "rec_info_seeking", "recommending", "negotiation"):
                    _stall_forced = True
                    session.stall_count = 0
                    if session.phase == "info_seeking":
                        orchestrator_decision = {
                            "action": "assess_meal",
                            "reasoning": "(stall-exit: user cannot provide more info → assess with available data)",
                            "instruction": "",
                        }
                    elif session.phase == "rec_info_seeking":
                        orchestrator_decision = {
                            "action": "recommend",
                            "reasoning": "(stall-exit: user cannot provide preferences → recommend without them)",
                            "instruction": "",
                        }
                    else:  # recommending / negotiation
                        orchestrator_decision = {
                            "action": "motivational_close",
                            "reasoning": "(stall-exit: user unresponsive in recommendation/negotiation → wrap up with motivation)",
                            "instruction": "",
                        }
                elif stall_exit_now:
                    # 이미 핵심 흐름을 마친 phase에서 stall → 종료
                    # (info_seeking/rec_info_seeking/recommending/negotiation만 강제 전환 대상)
                    session.turn_idx = next_turn
                    _fallback_close = "Thanks for sharing all of that — I think I have a good picture of your meal!"
                    session.coach.own_buffer.add(_fallback_close)
                    session.history.add_turn(turn_idx=next_turn, coach_utterance=_fallback_close)
                    session.turns.append(TurnRecord(turn_idx=next_turn, coach_utterance=_fallback_close))
                    next_question = _fallback_close
                    session.status        = SessionStatus.TERMINATED
                    session.terminated_by = "stall_exit"
                    session.final_aligned = alignment_aligned
                    session.final_score   = alignment_score

                # ── Orchestrator Router 호출 (stall 강제 전환이 아닌 경우만)
                if not _stall_forced and session.status == SessionStatus.ACTIVE:
                    _rec_history = (
                        session.meal_recommender.recommendation_history
                        if session.meal_recommender else []
                    )
                    _orch_msgs = session.orchestrator.get_routing_messages(
                        history=session.history,
                        turn_idx=turn_idx,
                        phase=session.phase,
                        recommendation_history=_rec_history,
                    )
                    orchestrator_input = _orch_msgs[1]["content"] if len(_orch_msgs) > 1 else ""
                    _orch_raw = _generate_single(
                        self._llm, _orch_msgs, self._config,
                        mode="orchestrator", chatgpt_client=_gpt_orch,
                    )
                    orchestrator_raw_output = _orch_raw
                    if not _orch_raw or not _orch_raw.strip():
                        print(f"[Router] WARNING: empty output at turn {turn_idx}, phase={session.phase}")
                    orchestrator_decision = session.orchestrator.parse_routing(
                        _orch_raw, turn_idx=turn_idx, phase=session.phase,
                    )

                # ── Orchestrator 결정 실행 (Router 결과든 stall 강제든 동일 경로)
                if orchestrator_decision and session.status == SessionStatus.ACTIVE:
                    action = orchestrator_decision.get("action", "seek_meal_info")
                    orch_instruction = orchestrator_decision.get("instruction", "")

                    # ── Estimator Bundle: IS/Recommender/Assessment/Motivational 에만 실행
                    needs_estimators = action in (
                        "seek_meal_info", "seek_recommendation_info",
                        "assess_meal", "recommend", "motivational_close",
                    )
                    if needs_estimators:
                        alignment_aligned, alignment_score, alignment_reasoning, \
                            alignment_input_ctx, alignment_raw_output = \
                            _run_alignment(self, session, _gpt_alignment, turn_idx)
                        certainty_score, certainty_reasoning, certainty_input, certainty_output = \
                            _run_uncertainty(self, session, _gpt, turn_idx)

                    # ── Step 4: Orchestrator 결정에 따른 분기
                    if action == "terminate":
                        session.turn_idx = next_turn
                        next_question = session.orchestrator.render_closing(orch_instruction)
                        session.coach.own_buffer.add(next_question)
                        session.history.add_turn(turn_idx=next_turn, coach_utterance=next_question)
                        session.turns.append(TurnRecord(turn_idx=next_turn, coach_utterance=next_question))
                        session.status        = SessionStatus.TERMINATED
                        session.terminated_by = "orchestrator"
                        session.final_aligned = alignment_aligned
                        session.final_score   = alignment_score
                        session.phase         = "terminated"

                    elif action == "motivational_close":
                        # ── Motivational Ending: assessment + health tip + encouraging close
                        _assess_msgs = session.orchestrator.get_assessment_messages(
                            history=session.history,
                            alignment_score=alignment_score,
                            alignment_reasoning=alignment_reasoning,
                        )
                        _assess_raw = _generate_single(
                            self._llm, _assess_msgs, self._config,
                            mode="orchestrator", chatgpt_client=_gpt_orch,
                        )
                        assessment_result = session.orchestrator.parse_assessment(_assess_raw)

                        _tg_motiv_msgs = session.orchestrator.get_textgen_motivational_messages(
                            assessment_result, session.history,
                        )
                        _tg_motiv_raw = _generate_single(
                            self._llm, _tg_motiv_msgs, self._config,
                            mode="orchestrator", chatgpt_client=_gpt_orch,
                        )
                        next_question = session.orchestrator.parse_textgen(
                            _tg_motiv_raw, assessment_result,
                        )
                        if not next_question.strip():
                            next_question = session.orchestrator.render_motivational_close(assessment_result)

                        _last_tg_msgs = _tg_motiv_msgs
                        _last_tg_template = assessment_result
                        session.turn_idx = next_turn
                        session.coach.own_buffer.add(next_question)
                        session.history.add_turn(turn_idx=next_turn, coach_utterance=next_question)
                        session.turns.append(TurnRecord(turn_idx=next_turn, coach_utterance=next_question))
                        session.phase         = "motivational_ending"
                        session.status        = SessionStatus.TERMINATED
                        session.terminated_by = "orchestrator"
                        session.final_aligned = alignment_aligned
                        session.final_score   = alignment_score

                    elif action == "assess_meal":
                        _assess_msgs = session.orchestrator.get_assessment_messages(
                            history=session.history,
                            alignment_score=alignment_score,
                            alignment_reasoning=alignment_reasoning,
                        )
                        _assess_raw = _generate_single(
                            self._llm, _assess_msgs, self._config,
                            mode="orchestrator", chatgpt_client=_gpt_orch,
                        )
                        assessment_result = session.orchestrator.parse_assessment(_assess_raw)

                        overall = assessment_result.get("overall", "partially_aligned")
                        _needs_rec = (overall != "aligned")
                        # Orchestrator TextGen: Assessment 결과 → 순수 피드백 (선호도 질문 미포함)
                        _tg_assess_msgs = session.orchestrator.get_textgen_assessment_messages(
                            assessment_result, needs_recommendation=_needs_rec,
                            history=session.history,
                        )
                        _tg_assess_raw = _generate_single(
                            self._llm, _tg_assess_msgs, self._config,
                            mode="orchestrator", chatgpt_client=_gpt_orch,
                        )
                        _assessment_text = session.orchestrator.parse_textgen(
                            _tg_assess_raw, assessment_result,
                        )
                        # Record assessment as Turn N
                        session.turn_idx = next_turn
                        session.coach.own_buffer.add(_assessment_text)
                        session.history.add_turn(turn_idx=next_turn, coach_utterance=_assessment_text)
                        session.turns.append(TurnRecord(turn_idx=next_turn, coach_utterance=_assessment_text))
                        if overall == "aligned":
                            next_question = _assessment_text
                            session.status        = SessionStatus.TERMINATED
                            session.terminated_by = "orchestrator"
                            session.final_aligned = True
                            session.final_score   = alignment_score
                            session.phase         = "terminated"
                        else:
                            session.phase = "rec_info_seeking"
                            _assessment_message = _assessment_text
                            # ── Double-turn: 즉시 선호도 질문 생성 (사용자 대기 없이)
                            _pref_turn = next_turn + 1
                            _pref_is_msgs = session.coach.get_messages(
                                session.history,
                                dead_end_topics=session.dead_end_topics if session.dead_end_topics else None,
                                mode="recommendation_info",
                            )
                            _pref_is_raw = _generate_single(
                                self._llm, _pref_is_msgs, self._config,
                                mode="coach", chatgpt_client=_gpt,
                            )
                            _pref_template = session.coach._parse_template(_pref_is_raw)
                            _pref_tg_msgs = session.orchestrator.get_textgen_question_messages(
                                _pref_template, session.history,
                            )
                            _pref_tg_raw = _generate_single(
                                self._llm, _pref_tg_msgs, self._config,
                                mode="orchestrator", chatgpt_client=_gpt_orch,
                            )
                            next_question = session.orchestrator.parse_textgen(
                                _pref_tg_raw, _pref_template,
                            )
                            _last_tg_msgs = _pref_tg_msgs
                            _last_tg_template = _pref_template
                            # Record preference question as Turn N+1
                            session.turn_idx = _pref_turn
                            session.coach.own_buffer.add(next_question)
                            session.history.add_turn(turn_idx=_pref_turn, coach_utterance=next_question)
                            session.turns.append(TurnRecord(turn_idx=_pref_turn, coach_utterance=next_question))

                    elif action == "recommend":
                        _user_prefs = ""
                        if session.memorizer:
                            _user_prefs = session.memorizer.get_preferences_text()
                        if session.user_preferences:
                            _user_prefs = (_user_prefs + "\n" + session.user_preferences).strip()

                        # 이전 추천 이력 + 사용자 피드백 수집 / Collect recommendation history + user feedback
                        _rec_history = session.meal_recommender.recommendation_history
                        _user_feedback = ""
                        if _rec_history and session.turns:
                            for t in reversed(session.turns):
                                if t.user_utterance:
                                    _user_feedback = f"Turn {t.turn_idx}: {t.user_utterance}"
                                    break

                        _rec_msgs = session.meal_recommender.get_messages(
                            meal_fact_sheet=session.history.meal_fact_sheet or "",
                            alignment_score=alignment_score if alignment_score is not None else 0.0,
                            alignment_reasoning=alignment_reasoning or "",
                            instruction=orch_instruction,
                            user_preferences=_user_prefs,
                            recommendation_history=_rec_history,
                            user_feedback=_user_feedback,
                        )
                        _rec_raw = _generate_single(
                            self._llm, _rec_msgs, self._config,
                            mode="recommender", chatgpt_client=_gpt,
                        )
                        recommendation_result = session.meal_recommender.parse_output(
                            _rec_raw, turn_idx=turn_idx,
                        )

                        # Orchestrator TextGen: MR 추천 템플릿 → 자연어 추천
                        _tg_msgs = session.orchestrator.get_textgen_recommendation_messages(
                            recommendation_result, session.history,
                            recommendation_history=_rec_history,
                        )
                        _tg_raw = _generate_single(
                            self._llm, _tg_msgs, self._config,
                            mode="orchestrator", chatgpt_client=_gpt_orch,
                        )
                        next_question = session.orchestrator.parse_textgen(
                            _tg_raw, recommendation_result,
                        )
                        _last_tg_msgs = _tg_msgs
                        _last_tg_template = recommendation_result
                        session.turn_idx = next_turn
                        session.coach.own_buffer.add(next_question)
                        session.history.add_turn(turn_idx=next_turn, coach_utterance=next_question)
                        session.turns.append(TurnRecord(turn_idx=next_turn, coach_utterance=next_question))
                        # negotiation 중이면 phase 유지, 아니면 recommending
                        if session.phase != "negotiation":
                            session.phase = "recommending"

                    elif action == "seek_recommendation_info":
                        is_msgs = session.coach.get_messages(
                            session.history,
                            dead_end_topics=session.dead_end_topics if session.dead_end_topics else None,
                            mode="recommendation_info",
                        )
                        _is_raw = _generate_single(
                            self._llm, is_msgs, self._config, mode="coach", chatgpt_client=_gpt,
                        )
                        is_template = session.coach._parse_template(_is_raw)
                        # Orchestrator TextGen: IS 질문 템플릿 → 자연어 질문
                        _tg_msgs = session.orchestrator.get_textgen_question_messages(
                            is_template, session.history,
                        )
                        _tg_raw = _generate_single(
                            self._llm, _tg_msgs, self._config,
                            mode="orchestrator", chatgpt_client=_gpt_orch,
                        )
                        next_question = session.orchestrator.parse_textgen(
                            _tg_raw, is_template,
                        )
                        _last_tg_msgs = _tg_msgs
                        _last_tg_template = is_template

                        session.turn_idx = next_turn
                        session.coach.own_buffer.add(next_question)
                        session.history.add_turn(turn_idx=next_turn, coach_utterance=next_question)
                        session.turns.append(TurnRecord(turn_idx=next_turn, coach_utterance=next_question))
                        # recommending에서 seek_recommendation_info가 왔으면 negotiation으로 전환
                        if session.phase == "recommending":
                            session.phase = "negotiation"

                    else:
                        # action == "seek_meal_info" (기본값)
                        is_msgs = session.coach.get_messages(
                            session.history,
                            dead_end_topics=session.dead_end_topics if session.dead_end_topics else None,
                            mode="meal_info",
                        )
                        _is_raw = _generate_single(
                            self._llm, is_msgs, self._config, mode="coach", chatgpt_client=_gpt,
                        )
                        is_template = session.coach._parse_template(_is_raw)
                        # Orchestrator TextGen: IS 질문 템플릿 → 자연어 질문
                        _tg_msgs = session.orchestrator.get_textgen_question_messages(
                            is_template, session.history,
                        )
                        _tg_raw = _generate_single(
                            self._llm, _tg_msgs, self._config,
                            mode="orchestrator", chatgpt_client=_gpt_orch,
                        )
                        next_question = session.orchestrator.parse_textgen(
                            _tg_raw, is_template,
                        )

                        # 중복 질문 탐지 + 재시도
                        _already_asked = session.history.get_all_coach_questions()
                        if _is_duplicate_question(next_question, _already_asked):
                            _retry_msgs = is_msgs + [{
                                "role": "user",
                                "content": (
                                    "[SYSTEM NOTE: The question you just generated was already asked. "
                                    "Please ask about a completely different food item or a new aspect "
                                    "that has NOT yet been covered in this conversation.]"
                                ),
                            }]
                            _retry_raw = _generate_single(
                                self._llm, _retry_msgs, self._config, mode="coach", chatgpt_client=_gpt,
                            )
                            _retry_tmpl = session.coach._parse_template(_retry_raw)
                            _tg_retry_msgs = session.orchestrator.get_textgen_question_messages(
                                _retry_tmpl, session.history,
                            )
                            _tg_retry_raw = _generate_single(
                                self._llm, _tg_retry_msgs, self._config,
                                mode="orchestrator", chatgpt_client=_gpt_orch,
                            )
                            _retry_q = session.orchestrator.parse_textgen(
                                _tg_retry_raw, _retry_tmpl,
                            )
                            if _retry_q.strip():
                                next_question = _retry_q
                                _tg_msgs = _tg_retry_msgs
                                is_template = _retry_tmpl

                        _fallback = "Could you tell me more about your meal?"
                        next_question = next_question.replace(
                            SharedConversationHistory.TERMINATION_TOKEN, ""
                        ).strip() or _fallback
                        _last_tg_msgs = _tg_msgs
                        _last_tg_template = is_template

                        session.turn_idx = next_turn
                        session.coach.own_buffer.add(next_question)
                        session.history.add_turn(turn_idx=next_turn, coach_utterance=next_question)
                        session.turns.append(TurnRecord(turn_idx=next_turn, coach_utterance=next_question))

        # ── Guardrail Output Guard: Coach 응답 검증 / Validate coach response
        if session.guardrail and next_question:
            _og_msgs = session.guardrail.get_output_guard_messages(next_question)
            _og_raw = _generate_single(
                self._llm, _og_msgs, self._config,
                mode="guardrail", chatgpt_client=_gpt_guardrail,
            )
            _og_result = session.guardrail.parse_output_guard(_og_raw)
            if not _og_result["passed"]:
                _regenerated = False
                _guard_reason = _og_result.get("reason", "")
                # Re-prompt Orchestrator TextGen with guard feedback (max 2 retries)
                if _last_tg_msgs and _last_tg_template:
                    for _retry_i in range(2):
                        _retry_tg_msgs = _last_tg_msgs + [{
                            "role": "user",
                            "content": (
                                "[SAFETY NOTE: Your previous response was flagged. "
                                f"Reason: {_guard_reason}. "
                                "Please rephrase while keeping the conversation focused on "
                                "the user's meal and nutrition. Avoid medical advice, "
                                "calorie prescriptions, or any content outside meal coaching.]"
                            ),
                        }]
                        _retry_raw = _generate_single(
                            self._llm, _retry_tg_msgs, self._config,
                            mode="orchestrator", chatgpt_client=_gpt_orch,
                        )
                        _retry_q = session.orchestrator.parse_textgen(
                            _retry_raw, _last_tg_template,
                        )
                        _og2_msgs = session.guardrail.get_output_guard_messages(_retry_q)
                        _og2_raw = _generate_single(
                            self._llm, _og2_msgs, self._config,
                            mode="guardrail", chatgpt_client=_gpt_guardrail,
                        )
                        _og2_result = session.guardrail.parse_output_guard(_og2_raw)
                        if _og2_result["passed"]:
                            next_question = _retry_q
                            _regenerated = True
                            break
                        _guard_reason = _og2_result.get("reason", _guard_reason)
                if not _regenerated:
                    _safe = "Could you tell me more about what you're having for your meal?"
                    next_question = _safe
                if session.turns:
                    session.turns[-1].coach_utterance = next_question

        # ── 종료 시 최종 요약 / Final summary on termination
        if session.status != SessionStatus.ACTIVE:
            _mt_final = session.history.to_plain_text_from(session.last_meal_track_start)
            if _mt_final:
                _mt_msgs = self._meal_tracker.get_messages(_mt_final, prev_fact_sheet=session.history.meal_fact_sheet)
                session.history.update_meal_fact_sheet(
                    _generate_single(self._llm, _mt_msgs, self._config, mode="tracker", chatgpt_client=_gpt)
                )
            _ds_final = session.history.to_plain_text_from(session.last_summarized_start)
            if _ds_final and session.dialog_summarization:
                _ds_msgs = self._dialog_summarizer.get_messages(_ds_final, prev_summary=session.history.dialog_summary)
                session.history.update_dialog_summary(
                    _generate_single(self._llm, _ds_msgs, self._config, mode="tracker", chatgpt_client=_gpt)
                )
            # Memorizer: 종료 시 식사 요약을 past_meals 에 저장
            if session.memorizer and session.history.meal_fact_sheet:
                session.memorizer.add_past_meal(
                    meal_type=getattr(session, 'meal_type', 'meal'),
                    summary=session.history.dialog_summary or "",
                    fact_sheet=session.history.meal_fact_sheet or "",
                )

        return {
            "turn_idx":        turn_idx,
            "user_reply":      user_reply_clean,
            "coach_question":  next_question,
            "guardrail_blocked": False,
            "alignment_aligned":   alignment_aligned,
            "alignment_score":     alignment_score,
            "alignment_reasoning": alignment_reasoning,
            "alignment_input":     alignment_input_ctx,
            "alignment_raw_output": alignment_raw_output,
            "certainty_score":    certainty_score,
            "certainty_reasoning": certainty_reasoning,
            "meal_tracker_input":  meal_tracker_input,
            "meal_tracker_output": meal_tracker_output,
            "certainty_input":     certainty_input,
            "certainty_output":    certainty_output,
            "orchestrator_decision": orchestrator_decision,
            "orchestrator_input":  orchestrator_input,
            "orchestrator_raw_output": orchestrator_raw_output,
            "recommendation_result": recommendation_result,
            "assessment_result":   assessment_result,
            "assessment_message":  _assessment_message,
            "phase":               session.phase,
            "dialog_summary":  session.history.dialog_summary or None,
            "meal_fact_sheet": session.history.meal_fact_sheet or None,
            "status":          session.status.value,
            "terminated_by":   session.terminated_by,
            "aligned_label":   _alignment_label(alignment_aligned),
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
            "alignment_enabled":   session.alignment_enabled,
            "nutrition_goal":  session.nutrition_goal,
            "meal_description": session.meal_description,
            "meal_ingredient": session.meal_ingredient,
            "meal_type":       session.meal_type,
            "turns": [
                {
                    "turn_idx":        t.turn_idx,
                    "coach_utterance": t.coach_utterance,
                    "user_utterance":  t.user_utterance,
                    "alignment_aligned":   t.alignment_aligned,
                    "alignment_score":     t.alignment_score,
                    "aligned_label":   _alignment_label(t.alignment_aligned),
                }
                for t in session.turns
            ],
            "summary":          session.history.dialog_summary,
            "meal_fact_sheet":  session.history.meal_fact_sheet,
            "dialog_summary":   session.history.dialog_summary,
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

def _run_alignment(mgr, session, _gpt_alignment, turn_idx):
    """Alignment Estimator 실행 → (aligned, score, reasoning, input_ctx, raw_output) 반환."""
    if session.alignment_enabled and session.alignment_tracker.should_evaluate(turn_idx):
        alignment_msgs = session.alignment_tracker.get_messages(session.history)
        raw_verdict = _generate_single(
            mgr._llm, alignment_msgs, mgr._config,
            mode="alignment", chatgpt_client=_gpt_alignment,
        )
        aligned = session.alignment_tracker.apply_judgment(raw_verdict, turn_idx)
        score = session.alignment_tracker.last_score
        reasoning = session.alignment_tracker.last_reasoning
        input_ctx = alignment_msgs[1]["content"] if len(alignment_msgs) > 1 else ""
        if session.turns:
            session.turns[-1].alignment_aligned = aligned
            session.turns[-1].alignment_score = score
        return aligned, score, reasoning, input_ctx, raw_verdict
    return None, None, None, None, None


def _run_uncertainty(mgr, session, _gpt, turn_idx):
    """Uncertainty Estimator 실행 → (score, reasoning, input_ctx, raw_output) 반환."""
    if (session.uncertainty_tracking
            and session.uncertainty_tracker
            and session.status == SessionStatus.ACTIVE):
        _ut_msgs = session.uncertainty_tracker.get_messages(session.history)
        _ut_raw = _generate_single(
            mgr._llm, _ut_msgs, mgr._config,
            mode="certainty", chatgpt_client=_gpt,
        )
        _input = _ut_msgs[1]["content"] if len(_ut_msgs) > 1 else ""
        reasoning, score = session.uncertainty_tracker.parse_output(_ut_raw)
        return score, reasoning, _input, _ut_raw
    return None, None, None, None


def _generate_single(llm, messages: list, config, mode: str = "coach",
                     chatgpt_client=None) -> str:
    """
    단일 응답 생성.  chatgpt_client 가 주어지면 ChatGPT API,
    None 이면 llama-cpp (Gemma) 를 사용합니다.

    / Generate a single response.
      Uses ChatGPT API when chatgpt_client is provided,
      otherwise falls back to llama-cpp (Gemma).

    Parameters
    ----------
    llm            : llama-cpp Llama 객체 (gemma 폴백용)
    messages       : [{role, content}, ...]
    config         : InteractiveConfig
    mode           : "coach" | "user" | "alignment" | "tracker" | "certainty" | "orchestrator" | "recommender" | "guardrail"
    chatgpt_client : ChatGPTClient 또는 None
    """
    # ── 백엔드 선택 / Backend selection ──
    if chatgpt_client is not None:
        from utils.llm_chatgpt import generate_response as _gen
        _model = chatgpt_client
    else:
        from utils.llm_utils import generate_response as _gen
        _model = llm

    # ── 모드별 파라미터 / Mode-specific parameters ──
    if mode == "alignment":
        return _gen(
            _model, messages,
            max_new_tokens=config.alignment_max_new_tokens,
            sampling=config.alignment_sampling,
            stop_at_newline=False,
        )
    if mode == "tracker":
        return _gen(
            _model, messages,
            max_new_tokens=config.summarize_max_new_tokens,
            sampling="greedy",
            stop_at_newline=False,
        )
    if mode == "certainty":
        return _gen(
            _model, messages,
            max_new_tokens=config.certainty_max_new_tokens,
            sampling="greedy",
            stop_at_newline=False,
        )
    if mode == "orchestrator":
        return _gen(
            _model, messages,
            max_new_tokens=config.orchestrator_max_new_tokens,
            sampling="greedy",
            stop_at_newline=False,
        )
    if mode == "recommender":
        return _gen(
            _model, messages,
            max_new_tokens=config.recommendation_max_new_tokens,
            sampling="greedy",
            stop_at_newline=False,
        )
    if mode == "guardrail":
        return _gen(
            _model, messages,
            max_new_tokens=config.guardrail_max_new_tokens,
            sampling="greedy",
            stop_at_newline=False,
        )
    if mode == "coach":
        _sampling = getattr(config, 'coach_sampling', config.sampling)
        return _gen(
            _model, messages,
            max_new_tokens=config.max_new_tokens,
            sampling=_sampling,
        )
    # mode == "user"
    return _gen(
        _model, messages,
        max_new_tokens=config.max_new_tokens,
        sampling=config.sampling,
    )


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
    return _re.sub(r'\s+', ' ', _re.sub(r'[^a-z0-9\s]', '', text.lower())).strip()


def _is_duplicate_question(new_q: str, already_asked: list, threshold: float = 0.85) -> bool:
    """
    Exact normalized match OR Jaccard word-overlap above threshold.
    / 정규화된 정확 일치 또는 Jaccard 단어 유사도가 threshold 이상이면 True.
    """
    norm_new = _normalize(new_q)
    if not norm_new:
        return False
    for prev in already_asked:
        # 1) exact normalized match
        if _normalize(prev) == norm_new:
            return True
        # 2) Jaccard on content words (stopwords excluded)
        words_new  = {w for w in norm_new.split() if w not in _STOPWORDS}
        words_prev = {w for w in _normalize(prev).split() if w not in _STOPWORDS}
        if not words_new or not words_prev:
            continue
        union = words_new | words_prev
        if len(words_new & words_prev) / len(union) >= threshold:
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
    Alignment Tracker 판정을 사람이 읽기 좋은 레이블로 변환합니다.
    / Convert alignment verdict to a human-readable label.
    """
    if aligned is None:
        return "pending"
    return "aligned" if aligned else "not_aligned"
