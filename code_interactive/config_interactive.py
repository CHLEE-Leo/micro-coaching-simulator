"""
config_interactive.py
─────────────────────
Interactive (웹 UI) 모드 전용 설정.
/ Configuration for interactive (web UI) mode.

Batch 모드(code/config.py)와 달리 UserModel이 없으므로
관련 설정(user_llm_repo, batch_mode 등)이 제거됩니다.
/ Unlike batch mode (code/config.py), UserModel is absent,
  so user-llm settings are removed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
import sys
from pathlib import Path

# 부모 디렉터리(code/)를 import 경로에 추가
# Add parent code/ directory to import path
_CODE_DIR = Path(__file__).resolve().parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from config import SUPPORTED_GOALS, ACTION_GUIDELINES, SimulationConfig as _SimCfg  # noqa: F401  (재사용 / re-export)

# 공유 대화 제어 파라미터는 code/config.py SimulationConfig 에서 가져와 단일 소스 유지
# Shared conversation-control params read from SimulationConfig — single source of truth
_sim = _SimCfg()


@dataclass
class InteractiveConfig:
    """
    Interactive 세션 하나를 구성하는 모든 설정.
    / All settings for a single interactive session.

    Server
    ------
    host            : FastAPI 서버 바인드 주소 / FastAPI server bind address
    port            : FastAPI 서버 포트 / FastAPI server port
    reload          : uvicorn hot-reload (개발 시 True) / hot-reload for dev

    Model
    -----
    gguf_path       : 로컬 .gguf 파일 경로 (llama-cpp-python)
                      예) ~/.cache/models/gemma-3-12b-it-Q4_K_M.gguf
                      / Path to local .gguf model file for llama-cpp-python
    n_ctx           : 최대 컨텍스트 토큰 수 / max context length
    n_gpu_layers    : GPU에 올릴 레이어 수. 0=CPU 전용, -1=전체 GPU
                      / Layers to offload to GPU. 0=CPU only, -1=all GPU

    Coach
    -----
    coach_use_template_guidance : Action 가이드라인 프롬프트 포함 여부
                                  / Include action guidelines in coach prompt
    max_new_tokens  : 발화당 최대 생성 토큰 / max tokens per utterance
    sampling        : 생성 전략 / generation strategy

    Alignment Tracker
    -----
    alignment_min_turn          : 판정 시작 최소 턴 / min turn to start judging
    alignment_max_new_tokens    : Alignment Tracker 최대 출력 토큰 / alignment max output tokens
    alignment_sampling          : Alignment Tracker 생성 전략 / alignment generation strategy
    alignment_output_format     : 출력 포맷 / output format
    alignment_threshold   : aligned 임계값 (binary 제외) / alignment threshold
    alignment_use_goal_def      : goal_definition 블록 포함 여부 / use goal_def block
    alignment_use_workflow      : workflow 블록 포함 여부 / use workflow block

    Conversation
    ------------
    max_turns               : 최대 턴 수 (안전 상한) / max turns (safety ceiling)
    context_window          : 공유 히스토리 슬라이딩 윈도우 / sliding context window
    meal_track_every        : MealTracker 실행 주기 / meal tracker update interval
    summarize_every         : DialogSummarizer 요약 갱신 주기 / summary update interval
    summarize_max_new_tokens: 요약 최대 토큰 / summary max tokens
    """

    # ── 서버 / Server ─────────────────────────────────────────────────────────
    host:   str  = "0.0.0.0"
    port:   int  = 8000
    reload: bool = False   # 개발 시 True / set True for development

    # ── LLM 프로바이더 / LLM provider ────────────────────────────────────────
    # "gemma" → 로컬 GGUF (llama-cpp-python)
    # "chatgpt" → OpenAI ChatGPT API (LangGraph 경유)
    llm_provider: Literal["gemma", "chatgpt"] = "gemma"

    # Alignment Tracker 전용 LLM 프로바이더 (None이면 llm_provider를 따름)
    # / Alignment Tracker-specific LLM provider (None follows llm_provider)
    alignment_llm_provider: Literal["gemma", "chatgpt"] = "gemma"

    # ChatGPT 모델명 (llm_provider="chatgpt" 일 때만 사용)
    # / OpenAI model name (only used when llm_provider="chatgpt")
    chatgpt_model: str = "gpt-5.2"

    # ── 모델 / Model (llm_provider="gemma" 일 때 사용) ────────────────────────
    # 사용자의 로컬 GGUF 파일 경로를 지정하세요.
    # 예) ~/.cache/models/gemma-3-12b-it-Q4_K_M.gguf
    # HF에서 다운로드: hf download unsloth/gemma-3-12b-it-GGUF --include "*Q4_K_M*" --local-dir ~/models
    gguf_path:     str = "/home/messy92/models/gemma-3-12b-it-Q4_K_M.gguf"
    n_ctx:         int = 4096    # 최대 컨텍스트 길이 / max context length
    n_gpu_layers:  int = -1       # 0=CPU 전용 / -1=전체 레이어 GPU 오프로드
    n_threads:     int = None      # CPU 추론 스레드 수 (물리 코어 수 권장, None=자동)

    # ── Coach ─────────────────────────────────────────────────────────────────
    coach_use_template_guidance: bool = True
    max_new_tokens: int = _sim.max_new_tokens
    sampling: Literal["beam", "greedy", "sampling"] = "sampling"   # User 생성 전략
    # Coach는 greedy로 질문 반복을 방지. User는 sampling으로 자연스러웄 답변 유지.
    # Coach uses greedy to avoid repeating questions; User keeps sampling for naturalness.
    coach_sampling: Literal["beam", "greedy", "sampling"] = "greedy"

    # ── Alignment Tracker ─────────────────────────────────────────────────────────────────
    # ⚠️  alignment_min_turn, alignment_max_new_tokens, alignment_output_format 은
    #    code/config.py 의 SimulationConfig 값을 자동으로 읽어옵니다.
    #    → 웹 서버 설정을 바꾸려면 code/config.py 의 SimulationConfig 를 수정하세요.
    alignment_min_turn:       int   = _sim.alignment_min_turn
    alignment_max_new_tokens: int   = _sim.alignment_max_new_tokens
    alignment_sampling: Literal["beam", "greedy", "sampling"] = "greedy"
    alignment_output_format: Literal["binary", "0-1", "0-100"] = _sim.alignment_output_format
    alignment_threshold: float = 0.5
    alignment_use_goal_def:  bool = True
    alignment_use_workflow:  bool = True

    # ── 대화 제어 / Conversation control ─────────────────────────────────────
    # ⚠️  max_turns, context_window, summarize_every, stall_exit_turns, min_natural_end_turn 은
    #    code/config.py 의 SimulationConfig 값을 자동으로 읽어옵니다.
    #    → 웹 서버 설정을 바꾸려면 code/config.py 의 SimulationConfig 를 수정하세요.
    max_turns:                int = _sim.max_turns
    context_window:           int = _sim.context_window
    meal_track_every:         int = _sim.meal_track_every
    summarize_every:          int = _sim.summarize_every
    summarize_max_new_tokens: int = _sim.summarize_max_new_tokens
    certainty_max_new_tokens:  int = _sim.certainty_max_new_tokens
    recommendation_max_new_tokens: int = _sim.recommendation_max_new_tokens
    orchestrator_max_new_tokens:   int = _sim.orchestrator_max_new_tokens
    orchestrator_llm_provider:     str = _sim.orchestrator_llm_provider
    guardrail_max_new_tokens:      int = _sim.guardrail_max_new_tokens
    guardrail_llm_provider:        str = _sim.guardrail_llm_provider
    assessment_max_new_tokens:     int = _sim.assessment_max_new_tokens
    stall_exit_turns:         int = _sim.stall_exit_turns
    min_natural_end_turn:     int = _sim.min_natural_end_turn

    # ── 리포 이름 (말풍선 레이블 표시용) / Repo labels shown above chat bubbles ──
    coach_llm_repo: str = _sim.coach_llm_repo
    user_llm_repo:  str = _sim.user_llm_repo

    # ── 지원 목표 (읽기 전용) / Supported goals (read-only) ─────────────────
    supported_goals: list = field(default_factory=lambda: list(SUPPORTED_GOALS))
