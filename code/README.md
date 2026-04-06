# Micro-Coaching Simulator — 배치 시뮬레이션 (`code/`)

> **"[Paper Title]"** (under review)

vLLM 기반 멀티턴 대화 시뮬레이션 프레임워크로, 크라우드소싱된 식사 데이터셋 위에서 합성 영양 코칭 대화를 자동 생성합니다.

---

## 핵심 구조

### 에이전트 아키텍처

이 프로젝트의 전체 아키텍처는 10개 에이전트로 구성되어 있으며, Orchestrator가 중앙 허브로서 모든 사용자 대면 텍스트를 생성합니다 (자세한 내용은 [루트 README](../README.md) 참조).

**배치 시뮬레이션(`code/`)에서는 간소화된 흐름을 사용합니다:**

Orchestrator/MealRecommender/Guardrail/Memorizer는 사용되지 않으며, InformationSeeker가 직접 질문을 생성하고 AlignmentEstimator가 종료를 판단합니다.

```
┌──────────────────┐   질문    ┌──────────┐
│ InformationSeeker│ ───────→ │   User   │
└────────┬─────────┘          └────┬─────┘
         │                         │ 응답 (partial info)
         │                ┌────────┴─────────────┐
         │                │                      │
         │       ┌────────────────┐  ┌──────────────────┐
         │       │  MealTracker   │  │ DialogSummarizer │
         │       │ → Fact Sheet   │  │ → 대화흐름 요약   │
         │       └───────┬────────┘  └────────┬─────────┘
         │               │                    │
         │               ▼                    ▼
         │    ┌────────────────────────────────────┐
         │    │      Shared Conversation History   │
         │    └──────────────┬─────────────────────┘
         │                   │ meal_fact_sheet
         │                   ▼
         │          ┌─────────────────┐
         │          │AlignmentEstimator│ ← aligned 판정 시 종료
         │          │ → 목표 정렬 신호 │
         │          └─────────────────┘
         │
    IS 질문 반복 (max_turns까지)
```

**배치 모드에서 사용되는 에이전트 (6종):**

| 에이전트 | 역할 | 파일 |
|----------|------|------|
| **InformationSeeker** | 질문 직접 생성 (Orchestrator 없이) | `models/information_seeker.py` |
| **User** | 식사 정보를 점진적으로 공개, 페르소나 반영, 자연스러운 종료 | `models/user.py` |
| **MealTracker** | 대화에서 Fact Sheet 구조화 추출 | `models/meal_tracker.py` |
| **DialogSummarizer** | 대화 흐름 서술형 요약 | `models/dialog_summarizer.py` |
| **AlignmentEstimator** | 영양 목표 정렬 판정 + 종료 결정 | `models/alignment_estimator.py` |
| **UncertaintyEstimator** | 정보 충분성 확신도 (배치 모드에서 참조용) | `models/uncertainty_estimator.py` |

**코드에 정의되어 있으나 배치 시뮬레이션에서 미사용 (4종):**

| 에이전트 | 역할 | 사용처 |
|----------|------|--------|
| **Orchestrator** | 중앙 허브 — user intent 분석, action 결정, 텍스트 렌더링, 식사 평가 | `code_interactive/` 전용 |
| **MealRecommender** | 구조화된 추천 템플릿 생성 | `code_interactive/` 전용 |
| **Guardrail** | LLM 기반 양방향 안전 필터 (Output Guard 실패 시 최대 2회 재생성) | `code_interactive/` 전용 |
| **Memorizer** | 다중 세션 사용자 프로필 유지 | `code_interactive/` 전용 |

### 두 종류의 요약 에이전트

```
┌─────────────────────────┬─────────────────────────┐
│   MealTracker           │   DialogSummarizer      │
├─────────────────────────┼─────────────────────────┤
│ 식사 정보 구조화 추출    │  대화 흐름 서술형 요약   │
│ 주 소비자: AlignmentEstimator        │  주 소비자: InformationSeeker, User  │
│ "무엇을 먹는가"         │  "무엇을 물었/답했는가"   │
│ 정확한 영양 판정 지원    │  반복 질문/응답 방지      │
└─────────────────────────┴─────────────────────────┘
```

---

## 2단계 대화 설계

```
Turn 0   Coach: "What are you thinking of having for {meal_type}?"
         User : 음식 이름 전체 공개 (meal_description)

Turn 1+  Coach: 각 음식의 재료·조리법·양을 하나씩 질문
         User : 상세 정보를 점진적으로 공개 (meal_ingredient)
         MealTracker: 식사 정보를 Fact Sheet로 구조화 (매 meal_track_every 턴)
         DialogSummarizer: 대화 흐름을 서술형 요약 (매 summarize_every 턴)
         AlignmentEstimator: Fact Sheet 기반 목표 정렬 판정 → aligned 시 종료
```

---

## 디렉토리 구조

```
code/
├── config.py                   # SimulationConfig — 모든 설정의 단일 소스 (CLI 없음)
├── run_simulation.py           # 실행 진입점
├── run_simulation.sh           # GPU/Conda 환경 설정 셸 스크립트
│
├── core/
│   ├── __init__.py             # 패키지 re-export
│   ├── memory.py               # ConversationBuffer (Principle 2)
│   │                           # SharedConversationHistory (Principle 3+4)
│   └── simulation.py           # simulate_conversation() — 단일 순차 처리
│                               # simulate_conversations_batch() — N건 병렬 배치 처리
│
├── models/
│   ├── __init__.py             # 패키지 re-export
│   ├── information_seeker.py   # InformationSeeker — 질문 생성, 중복 감지, Action Guidelines
│   ├── user.py                 # UserModel — 점진적 정보 공개, 페르소나 반영, [END] 태그 자연 종료
│   ├── alignment_estimator.py  # AlignmentEstimator — 영양 목표 정렬 신호 (binary/0-1/0-100)
│   ├── uncertainty_estimator.py # UncertaintyEstimator — 정보 충분성 기반 certainty 신호
│   ├── meal_tracker.py         # MealTrackerModel — 구조화된 Meal Fact Sheet 추출
│   ├── dialog_summarizer.py    # DialogSummarizerModel — 대화 흐름 서술형 요약
│   ├── orchestrator.py         # Orchestrator — 다음 행동 종합 결정
│   ├── meal_recommender.py     # MealRecommender — 식사 개선 추천
│   ├── guardrail.py            # Guardrail — LLM 기반 양방향 안전 필터
│   └── memorizer.py            # Memorizer — 다중 세션 사용자 프로필 관리
│
├── utils/
│   ├── __init__.py
│   ├── llm_utils.py            # load_model(), batch_generate(), generate_response()
│   │                           # build_sampling_params() — vLLM SamplingParams 빌더
│   └── io_utils.py             # load_meal_data(), save_results(), make_output_path()
│
└── _*.py                       # 레거시 스크립트 (이전 버전, 참조용)
    ├── _create_simulated_dialogs.py
    ├── _functions_for_general.py
    ├── _functions_for_simulation.py
    └── _functions_for_simulation_baseline.py
```

---

## 실행 방법

`config.py`를 수정한 뒤 실행합니다:

```bash
# GPU 지정 실행
bash run_simulation.sh 6,7

# 또는 직접 실행
python run_simulation.py
```

### 주요 설정 항목 (`config.py`)

```python
# 데이터
goal                = "lean_protein"    # lean_protein | half_fruits_vegetables | one_fourth_carbs | drink_water
data_path           = "../data/df_normal_without_test_string.csv"

# 모델
coach_llm_repo      = "google/gemma-3-12b-it"
user_llm_repo       = "google/gemma-3-12b-it"
alignment_llm_repo  = "google/gemma-3-12b-it"   # "" → coach_llm과 동일 모델 공유
num_gpus            = 1                          # vLLM tensor_parallel_size

# 실행 모드
batch_mode          = True   # True: 병렬 배치 / False: 순차 단일

# 생성 전략
coach_sampling      = "greedy"     # Coach: greedy (반복 방지)
sampling            = "sampling"   # User: sampling (자연스러운 표현)
alignment_sampling  = "greedy"     # AlignmentEstimator: greedy (재현성)

# Coach 설계
coach_use_template_guidance = True   # Action Guidelines 포함

# AlignmentEstimator 설정
alignment_min_turn       = 0       # 판정 시작 최소 턴 (0-based)
alignment_max_new_tokens = 120     # AlignmentEstimator JSON 출력 최대 토큰
alignment_output_format  = "binary"   # binary | 0-1 | 0-100
alignment_threshold      = 0.5   # 0-1/0-100 포맷일 때 aligned 임계값
alignment_use_goal_def   = True   # goal_definition 블록 포함 여부
alignment_use_workflow   = True   # expert workflow 블록 포함 여부

# UncertaintyEstimator 설정
certainty_max_new_tokens = 200   # UncertaintyEstimator JSON 출력 최대 토큰

# Orchestrator 설정
orchestrator_max_new_tokens = 200   # Orchestrator JSON 출력 최대 토큰

# MealRecommender 설정
recommendation_max_new_tokens = 300   # MealRecommender 출력 최대 토큰

# 대화 제어
max_turns                = 15    # 안전 상한 턴 수
context_window           = 10    # 최근 N턴만 shared history로 참조
meal_track_every         = 1     # MealTracker 실행 주기 (기본 매턴)
summarize_every          = 3     # DialogSummarizer 요약 갱신 주기
summarize_max_new_tokens = 120   # 요약 최대 토큰 수
stall_exit_turns         = 3     # 연속 non-answer N회 시 종료
min_natural_end_turn     = 3     # AI User [END] 태그 최소 허용 턴
```

> **단일 설정 소스** — `code/config.py`(`SimulationConfig`)는 배치 파이프라인과 인터랙티브 웹 서버(`code_interactive/`) 양쪽에서 읽힙니다. 여기서 값을 수정하고 서버를 재시작하면 됩니다.

---

## 출력 형식

결과는 `results/goal={goal}/model={model_name}/` 하위에 JSON으로 저장됩니다.

```json
{
  "id": 42,
  "goal_id": 1,
  "meal_id": 100,
  "nutrition_goal": "lean_protein",
  "meal_type": "lunch",
  "meal_description": "Grilled chicken salad",
  "turns": [
    {"turn_idx": 0, "coach_utterance": "...", "user_utterance": "..."},
    {"turn_idx": 1, "coach_utterance": "...", "user_utterance": "..."}
  ],
  "meal_fact_sheet": "- Food items: grilled chicken salad\n- Ingredients: ...",
  "dialog_summary": "The coach asked about ... The user mentioned ...",
  "terminated_by": "alignment",
  "pred_alignment": true,
  "pred_score": 1.0,
  "true_alignment": true,
  "alignment_correct": true,
  "alignment_history": [
    {"turn_idx": 3, "aligned": false, "score": 0.0, "reasoning": "...", "raw_output": "..."},
    {"turn_idx": 4, "aligned": true,  "score": 1.0, "reasoning": "...", "raw_output": "..."}
  ]
}
```

### 종료 조건

| `terminated_by` | 설명 |
|-----------------|---------|
| `"alignment"` | AlignmentEstimator가 aligned 판정 (정상 종료) |
| `"max_turns"` | `max_turns` 소진 (안전 상한 도달) |
| `"natural_end"` | AI User가 자연 종료 |

---

## 설계 원칙 상세

> 코어 설계 원칙(D1–D9) 전체는 [루트 README](../README.md#설계-원칙-design-principles)를 참조하세요. 아래는 배치 시뮬레이션(`code/`)에서 직접 적용되는 대화 설계 원칙을 상세히 설명합니다.

### P1 — Action Guidelines

`coach_use_template_guidance=True`이면 Coach(InformationSeeker) 시스템 프롬프트에 질문 전략 가이드라인이 포함됩니다:

- `WHAT_ELSE` — 추가 음식 항목 질문
- `WHAT_ELSE_IN` — 컨테이너 음식(샌드위치 등) 내부 재료 질문
- `WHAT_KIND` — 음식의 구체적 종류/품종 질문
- `HOW_PREPARED` — 조리법 질문
- `HOW_MUCH` — 양/분량 질문
- `FALLBACK` — 불명확한 응답 시 재질문

### P2 — Own Buffer

`ConversationBuffer`로 각 에이전트가 자신의 발화 이력만 별도 관리합니다. InformationSeeker는 이전에 물어본 질문 목록을, User는 이미 공개한 정보를 추적하여 반복을 방지합니다. InformationSeeker는 `history.get_all_coach_questions()`를 통해 완전한 질문 목록을 직접 추출하여 own_buffer의 누락 가능성을 보완합니다.

### P3 — Sliding Context Window

`SharedConversationHistory`의 `context_window`(기본 5)만큼만 최근 턴을 LLM 컨텍스트에 포함합니다. 오래된 턴은 두 종류의 요약(Principle 4)으로 대체됩니다.

### P4 — 이중 요약 전략

| 요약 에이전트 | 역할 | 소비자 | 주기 |
|--------------|------|--------|------|
| **MealTracker** | 식사 정보를 `Food items / Ingredients / Preparation methods / Portions / Beverages` 형태로 구조화 | AlignmentEstimator | `meal_track_every` 턴 |
| **DialogSummarizer** | 대화 흐름(누가 무엇을 물었고 답했는지)을 2-4문장으로 서술 | InformationSeeker, User | `summarize_every` 턴 |

두 요약 모두 전체 요약(full)과 증분 요약(incremental) 모드를 지원합니다. 배치 시뮬레이션에서는 전체 요약, 인터랙티브 모드에서는 증분 요약이 주로 사용됩니다.

---

## 중복 질문 방지

Coach(InformationSeeker) 발화 생성 후 Jaccard 유사도(임계값 0.85) 기반 중복 감지를 수행합니다. 중복이 감지되면 `[SYSTEM NOTE]`를 추가하여 최대 2회 재생성을 시도하고, 그래도 중복이면 제네릭 폴백 질문을 사용합니다.

---

## Dead-End 토픽 및 Stall 종료

- User가 `"I'm not sure"` 등 non-answer로 응답하면 해당 Coach 질문이 dead-end 토픽으로 기록됩니다.
- 다음 턴 Coach(InformationSeeker) 프롬프트에 `[Topics the user already said they are NOT SURE about]` 블록이 주입되어 같은 토픽 재질문을 방지합니다.
- `stall_exit_turns`(기본 3)회 연속 non-answer 시 Coach(InformationSeeker)가 마무리 발화를 생성하고 대화가 종료됩니다.

---

## 지원 영양 목표

| 목표 | 카테고리 | 설명 |
|------|----------|------|
| `lean_protein` | qualitative | 닭고기·생선·콩류 등 저지방 단백질 중심 |
| `half_fruits_vegetables` | quantitative | 식판의 절반을 과일·채소로 채우기 |
| `one_fourth_carbs` | quantitative | 식판의 1/4을 복합 탄수화물로 구성 |
| `drink_water` | qualitative | 주요 음료를 물로 선택 |

AlignmentEstimator의 expert workflow는 목표 카테고리(qualitative/quantitative)에 따라 자동으로 선택됩니다.

---

## 레거시 스크립트

`_`로 시작하는 파일들은 이전 버전의 시뮬레이션 코드입니다:

| 파일 | 설명 |
|------|------|
| `_create_simulated_dialogs.py` | 초기 대화 생성 스크립트 (argparse 기반) |
| `_functions_for_general.py` | 범용 유틸리티 (GOALS, 템플릿 등) |
| `_functions_for_simulation.py` | 구버전 시뮬레이션 함수 (transformers 직접 사용) |
| `_functions_for_simulation_baseline.py` | 베이스라인 시뮬레이션 함수 |

현재 코드는 vLLM 기반으로 완전히 리팩터링되었으며, 이 파일들은 참조용으로만 보존됩니다.

---

## Citation

```bibtex
@misc{micro-coaching-simulator-2026,
  author = {},
  title  = {},
  year   = {2026},
  url    = {}
}
```
