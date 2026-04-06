"""
config.py
─────────
전체 시뮬레이션 파이프라인의 하이퍼파라미터·경로를 한 곳에서 관리합니다.

사용법
  run_simulation.py 는 argparse 인자를 받지 않으며, 이 파일을 직접 수정해서 실행합니다.
  변경이 필요한 항목만 아래에서 값을 바꾼 뒤 `python run_simulation.py` 를 실행하세요.
"""

from dataclasses import dataclass
from typing import Literal


# ──────────────────────────────────────────────────────────────────────────────
# 1. Coach 프롬프트에 주입할 Action 가이드라인 (Principle 1)
# ──────────────────────────────────────────────────────────────────────────────
ACTION_GUIDELINES = """\
To learn about a meal, you might explore questions like:

  • What specific ingredients or components are in a food
  • How something is prepared or cooked
  • Approximate portion sizes or amounts
  • What kind or variety of a food (e.g. whole wheat vs. white bread)
  • What else might be inside a composite food (sandwich, bowl, wrap, etc.)
  • Anything else that is nutritionally relevant and currently unknown

These are examples, not a rigid checklist.
Ask whatever is most useful given the current conversation context and the nutritional goal.\
"""


# ──────────────────────────────────────────────────────────────────────────────
# 2. 지원하는 영양 목표
# ──────────────────────────────────────────────────────────────────────────────
SUPPORTED_GOALS = [
    "lean_protein",
    "half_fruits_vegetables",
    "one_fourth_carbs",
    "drink_water",
]


@dataclass
class SimulationConfig:
    """
    한 번의 시뮬레이션 실행에 필요한 모든 설정.

    실행 환경
      num_gpus            : vLLM tensor_parallel_size (사용할 GPU 수)
      max_model_len       : vLLM 최대 컨텍스트 길이 (메모리 절감 목적으로 축소 가능)
      dtype               : 모델 가중치 데이터 타입 ("bfloat16" | "float16" | "auto")
      seed                : 재현성을 위한 랜덤 시드

    실행 모드
      batch_mode          : True → 병렬 배치 처리 / False → 순차 단일 처리

    데이터
      goal                : 영양 목표 ("lean_protein" | "half_fruits_vegetables" |
                            "one_fourth_carbs" | "drink_water")
      data_path           : 학습 데이터 CSV 경로

    모델
      coach_llm_repo      : HuggingFace Coach 모델 경로
      user_llm_repo       : HuggingFace User 모델 경로 (coach와 동일 가능)
      alignment_llm_repo      : HuggingFace Alignment Tracker 모델 경로
                            "" (빈 문자열) → coach_llm_repo 와 동일 모델 공유

    Coach 설계 (Principle 1)
      coach_use_template_guidance : Action 가이드라인을 프롬프트에 포함

    생성 파라미터 (Coach / User)
      max_new_tokens      : 한 발화당 최대 생성 토큰 수
      sampling            : "beam" | "greedy" | "sampling"

    Alignment Tracker
      alignment_min_turn       : Alignment Tracker 가 판정을 시작하는 최소 턴 인덱스 (0-based)
      alignment_max_new_tokens : Alignment Tracker LLM 최대 출력 토큰 수 (JSON + reasoning)
      alignment_sampling       : Alignment Tracker 생성 전략 (판정 재현성을 위해 "greedy" 권장)

    대화 제어
      max_turns            : 안전 상한 턴 수 (Alignment Tracker 정상 종료 전 강제 종료 방지용)
      context_window       : Shared history 에서 유지할 최근 턴 수 (0 = 전체)
      meal_track_every     : MealTracker 실행 주기 (기본 매턴)
      summarize_every      : N 턴마다 DialogSummarizer 요약 갱신
      summarize_max_new_tokens : 요약 LLM 최대 출력 토큰 수

    출력
      output_dir           : 결과 JSON 저장 루트 디렉토리
    """

    # ── 실행 환경 ─────────────────────────────────────────────────────────────
    num_gpus:      int  = 1
    max_model_len: int  = 4096
    dtype:         str  = "bfloat16"   # "bfloat16" | "float16" | "auto"
    seed:          int  = 42

    # ── 실행 모드 ─────────────────────────────────────────────────────────────
    batch_mode: bool = True   # True: 병렬 배치 / False: 순차 단일

    # ── 데이터 ────────────────────────────────────────────────────────────────
    goal:      str = "lean_protein"
    data_path: str = "../data/df_normal_without_test_string.csv"

    # ── 모델 ──────────────────────────────────────────────────────────────────
    coach_llm_repo: str = "google/gemma-3-12b-it"
    user_llm_repo:  str = "google/gemma-3-12b-it"
    alignment_llm_repo: str = "google/gemma-3-12b-it"   # "" → coach_llm_repo 와 동일 모델 공유

    # ── Coach 설계 (Principle 1) ───────────────────────────────────────────────
    coach_use_template_guidance: bool = True

    # ── 생성 파라미터 (Coach / User) ─────────────────────────────────────────
    max_new_tokens: int = 150
    sampling: Literal["beam", "greedy", "sampling"] = "sampling"  # User 생성 전략
    # Coach는 greedy로 반복 방지, User는 sampling으로 자연스러웄 표현 유지
    coach_sampling: Literal["beam", "greedy", "sampling"] = "greedy"

    # ── Alignment Tracker ─────────────────────────────────────────────────────────────────
    alignment_min_turn:       int = 0       # turn 0: food 이름 이미 검 / turn 1이후: ingredient 답변 누적 시작 → turn 3부터 판정
    alignment_max_new_tokens: int = 300     # JSON + reasoning 출력
    alignment_sampling: Literal["beam", "greedy", "sampling"] = "greedy"

    # Alignment Tracker scaffold 사용 여부 (False 로 설정 시 해당 블록이 프롬프트에서 완전히 제거됨)
    alignment_use_goal_def:  bool = True   # goal_definition 블록 포함 여부
    alignment_use_workflow:  bool = True   # WORKFLOW OF EXPERT NUTRITIONIST 블록 포함 여부
    # output_format_inst 는 항상 포함되며, 아래 세 가지 포맷 중 하나를 선택
    alignment_output_format: Literal["binary", "0-1", "0-100"] = "binary"
    # 0-1 / 0-100 포맷 사용 시 aligned 판정 임계값 (정규화 후 [0, 1] 기준)
    # 0-100 점수는 /100 정규화 후 이 값과 비교합니다. binary 포맷에서는 무시됩니다.
    alignment_threshold: float = 0.5

    # ── 대화 제어 ─────────────────────────────────────────────────────────────
    # ⚠️  웹 UI 서버(code_interactive/)는 이 값들을 직접 읽습니다.
    #    아래 값을 바꾸면 `uvicorn` 서버를 재시작할 때 자동으로 반영됩니다.
    max_turns:                int = 15  # 비상 상한
    context_window:           int = 10   # Principle 3 : 최근 N 턴만 shared history로 참조
    meal_track_every:         int = 1   # MealTracker 실행 주기 (기본 매턴)
    summarize_every:          int = 3   # DialogSummarizer 실행 주기 (N 턴마다 요약 갱신)
    summarize_max_new_tokens: int = 250 # 요약 LLM 최대 출력 토큰 수
    certainty_max_new_tokens:  int = 400 # UncertaintyEstimator JSON 출력 최대 토큰 수

    # ── MealRecommender ──────────────────────────────────────────────────────
    recommendation_max_new_tokens: int = 500  # MealRecommender JSON 출력 최대 토큰 수

    # ── Orchestrator ─────────────────────────────────────────────────────────
    orchestrator_max_new_tokens:  int = 500   # Orchestrator JSON 출력 최대 토큰 수
    orchestrator_llm_provider:    str = ""    # "" → llm_provider와 동일, "gemma" | "chatgpt"

    # ── Guardrail ────────────────────────────────────────────────────────
    guardrail_max_new_tokens:     int = 200   # Guardrail JSON 출력 최대 토큰 수
    guardrail_llm_provider:       str = ""    # "" → llm_provider와 동일

    # ── Assessment ───────────────────────────────────────────────────────
    assessment_max_new_tokens:    int = 500   # Orchestrator Assessment 출력 최대 토큰 수

    # 연속 non-answer N회 이상 시 Coach가 마무리 발화와 함께 종료
    # After N consecutive non-answers Coach generates a closing message and terminates
    stall_exit_turns: int = 3
    # AI User가 최소 이 턴 수 이전에 TERMINATION_TOKEN을 내보내면 무시
    min_natural_end_turn: int = 3
    # ── 출력 ──────────────────────────────────────────────────────────────────
    output_dir: str = "../results"
