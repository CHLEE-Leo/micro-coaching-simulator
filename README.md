# Micro-Coaching Simulator

> **"[Paper Title]"** (under review)

영양 코칭 대화를 자동으로 생성·평가하고, 실제 사용자와 AI 코치 간의 실시간 상호작용을 실험하기 위한 연구용 프레임워크입니다.

---

## 이 레포지토리의 목적

식사 목표 정렬(Nutritional Goal Alignment) 연구를 위해 두 가지 핵심 기능을 제공합니다.

1. **대화 데이터 생성** — InformationSeeker, User, AlignmentEstimator, UncertaintyEstimator, MealTracker, DialogSummarizer, Orchestrator, MealRecommender, Guardrail, Memorizer 열 개의 에이전트를 조합하여 크라우드소싱 식사 데이터셋 위에서 대량의 합성 코칭 대화를 자동 생성합니다.
2. **실시간 인터랙션 실험** — 실제 사람이 AI 코치와 대화하면서, AlignmentEstimator가 매 발화마다 목표 정렬 여부를, UncertaintyEstimator가 정보 충분성(확신도)을 실시간으로 평가합니다. Guardrail이 입출력 안전성을 검증하고, Memorizer가 다중 식사 세션 간 사용자 프로필을 유지합니다.

두 기능은 동일한 모델 구조와 대화 설계를 공유하되, **목적과 운영 방식이 다릅니다.**

---

## 두 폴더 한눈에 비교

|  | `code/` | `code_interactive/` |
|---|---|---|
| **역할** | 배치 시뮬레이션 (논문 실험 데이터 생성) | 웹 기반 실시간 코칭 UI |
| **사용자** | 없음 — LLM 에이전트들이 자율적으로 대화 | 실제 사람이 코치와 직접 대화 (또는 AI 시뮬레이션 관찰) |
| **LLM 백엔드** | vLLM (GPU 배치 추론) | llama-cpp-python (GGUF) 또는 ChatGPT API (LangGraph) |
| **실행 방식** | Python 스크립트 / 셸 배치 | FastAPI 서버 + 브라우저 SPA |
| **AlignmentEstimator** | 정렬 신호를 Orchestrator에 전달 | 매 발화마다 실시간 정렬 칩 표시 |
| **UncertaintyEstimator** | 확신도 신호를 Orchestrator에 전달 | 매 발화마다 확신도 점수 실시간 표시 |
| **Orchestrator** | 모든 신호를 종합하여 다음 행동 결정 | 모든 신호를 종합하여 다음 행동 결정 |
| **목적** | 합성 데이터셋 생성, 오프라인 분석 | 사용자 경험 실험, 프로토타입 검증 |
| **진입점** | `code/run_simulation.py` | `code_interactive/app.py` |

---

## 10-에이전트 아키텍처

인터랙티브 모드(`code_interactive/`)에서의 실제 에이전트 상호작용 흐름입니다.
SessionManager가 전체 턴 루프를 조율하며, Orchestrator는 action 결정과 사용자 대면 텍스트 생성을 담당합니다.
Estimator Bundle은 Orchestrator가 아닌 **SessionManager가 직접** 호출하며, 그 결과를 MealRecommender에 전달합니다.

```
  ┌──────────┐        ┌──────────────┐        ┌──────────────┐
  │   User   │ ←────→ │  Guardrail   │ ←────→ │ Orchestrator │
  │ (사람/AI) │        │ Input/Output │        │  (중앙 허브)  │
  └──────────┘        └──────────────┘        └──────┬───────┘
                                                      │
                    ┌─────────────────────────────────┘
                    │
              ┌─────┴─────┐
              │           │
              ▼           ▼
       ┌────────────┐  ┌──────────────┐
       │ Information │  │    Meal      │
       │   Seeker    │  │ Recommender  │
       │(질문 템플릿) │  │ (추천 템플릿) │
       └──────┬─────┘  └──────┬───────┘
              │  템플릿 반환    │  템플릿 반환
              └───────┬────────┘
                      │
                      ▼
               Orchestrator.render_*()
               → 사용자 대면 텍스트 생성

  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
  SessionManager가 직접 관리 (Orchestrator 경유하지 않음):

  ┌────────────────────────────────────────────────┐
  │            Estimator Bundle                    │
  │                                                │
  │  MealTracker ──→ meal_fact_sheet               │
  │                      │                         │
  │                      ▼                         │
  │  AlignmentEstimator (score + reasoning)        │
  │  UncertaintyEstimator (certainty score)        │
  │                                                │
  │  ※ alignment 결과 → MR.get_messages() 입력     │
  │  ※ IS에는 estimator 출력이 전달되지 않음        │
  └────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────┐
  │       Shared Conversation History        │
  │                                          │
  │  meal_fact_sheet  ← MealTracker (매 N턴) │
  │  dialog_summary   ← DialogSummarizer     │
  │  + 최근 context_window 원문              │
  └──────────────────────────────────────────┘
                        │
                 ┌──────┴──────┐
                 │  Memorizer  │
                 │ 사용자 프로필│
                 │ (다중 세션)  │
                 └─────────────┘
```

**Orchestrator의 역할:**
1. **Router** — 사용자 발화의 intent를 명시적으로 분석한 뒤(`user_intent` 필드), 대화 상태를 종합하여 다음 행동(Action)을 결정
2. **Assessor** — 식사 평가를 수행하고, 평가 결과를 사용자 대면 텍스트로 렌더링 (assessment는 순수 피드백만 포함, 선호도 질문은 별도 턴)
3. **TextGen** — IS/MR가 반환한 구조화 템플릿을 자연어로 렌더링
4. **Phase 관리** — `info_seeking → assessment → rec_info_seeking → recommending → [수락] motivational_ending → terminated` / `→ [거절] negotiation → motivational_ending → terminated`

**Phase별 에이전트 호출 흐름:**
```
[info_seeking 페이즈]
  seek_meal_info → IS.ask(mode="meal_info") → 질문 템플릿 → Orchestrator TextGen → 사용자 대면 질문

[assessment → rec_info_seeking 전환 (Double-turn)]
  assess_meal → Orchestrator.assess() → TextGen(순수 assessment 피드백)
    ├─ aligned   → terminate
    └─ 개선 필요 → Turn N: assessment 피드백 출력
                  → Turn N+1: IS(recommendation_info) → 선호도 질문 즉시 생성
                  → phase: rec_info_seeking

[rec_info_seeking 페이즈]  ← IS가 추천을 위한 사용자 선호도/제약을 수집
  seek_recommendation_info → IS.ask(mode="recommendation_info") → 질문 템플릿 → Orchestrator TextGen

[recommending 페이즈]  ← MR이 식사 개선 제안 전달
  recommend → MR.recommend(alignment_score, alignment_reasoning) → 추천 템플릿 → Orchestrator TextGen
    ├─ 사용자 수락 → motivational_close → phase: motivational_ending
    └─ 사용자 거절 → seek_recommendation_info / recommend → phase: negotiation

[negotiation 페이즈]  ← 사용자가 추천을 거절; Orchestrator가 맥락에 따라 IS 또는 MR 호출
  seek_recommendation_info → IS.ask(mode="recommendation_info") → 거절 사유 파악
  recommend → MR.recommend(...) → 대안 추천 제시
    └─ 사용자 수락 (또는 탐색 종료) → motivational_close → phase: motivational_ending

[motivational_ending 페이즈]  ← 동기부여 마무리
  motivational_close → Orchestrator.assess() + TextGen(동기부여 마무리 메시지)
    → (개선된) 식사 assessment + 보편적 건강 팁 + 동기부여 마무리 → terminated
```

**핵심 데이터 흐름:**
- `IS → Orchestrator`: 질문 템플릿 반환 (양방향)
- `MR → Orchestrator`: 추천 템플릿 반환 (양방향)
- `MealTracker → AlignmentEstimator`: Fact Sheet가 판정 입력으로 사용
- `AlignmentEstimator → MR`: alignment_score + reasoning이 추천 생성 입력
- `IS → (간접) → MR`: rec_info_seeking 페이즈에서 IS가 수집한 선호도가 MR 추천에 반영
- `Estimator ✗ IS`: IS에는 estimator 출력이 전달되지 않음

| 에이전트 | 역할 | 주요 파일 |
|----------|------|-----------|
| **Orchestrator** | 중앙 허브 — user intent 분석, action 결정, 사용자 대면 텍스트 생성, 식사 평가 | `models/orchestrator.py` |
| **InformationSeeker** | Orchestrator에 구조화된 질문 템플릿 반환 (meal_info / recommendation_info 두 모드) | `models/information_seeker.py` |
| **MealRecommender** | Orchestrator에 구조화된 추천 템플릿 반환 (alignment 결과를 입력으로 받음) | `models/meal_recommender.py` |
| **User** | 식사 정보를 점진적으로 공개, 페르소나 반영 (사람 또는 AI) | `models/user.py` |
| **MealTracker** | 대화에서 Fact Sheet 추출 → AlignmentEstimator 판정 입력 | `models/meal_tracker.py` |
| **DialogSummarizer** | 대화 흐름을 서술형으로 요약 → IS/User 프롬프트에 주입 | `models/dialog_summarizer.py` |
| **AlignmentEstimator** | Fact Sheet 기반 목표 정렬 판정 (score + reasoning → MR에 전달) | `models/alignment_estimator.py` |
| **UncertaintyEstimator** | 정보 충분성(certainty) 판정 | `models/uncertainty_estimator.py` |
| **Guardrail** | LLM 기반 양방향 안전 필터 — Input Guard + Output Guard | `models/guardrail.py` |
| **Memorizer** | 다중 식사 세션 간 사용자 프로필 유지 | `models/memorizer.py` |

> **참고:** 배치 시뮬레이션(`code/`)에서는 Orchestrator 이전의 구식 흐름을 사용합니다.
> IS가 직접 질문을 생성하고 AlignmentEstimator가 종료를 판단하며, Orchestrator/MealRecommender/Guardrail/Memorizer는 사용되지 않습니다.

---

## 대화 구조 (2단계 설계)

```
Turn 0  Coach: "저녁에 무엇을 드실 예정인가요?"
        User : 음식 이름 전체 공개 (meal_description)

Turn 1+ Coach: 각 음식의 재료·조리법·양을 구체적으로 질문
        User : 상세 정보를 점진적으로 공개 (meal_ingredient)
        MealTracker: 식사 정보를 Fact Sheet로 구조화 (매 N턴)
        DialogSummarizer: 대화 흐름을 서술형 요약 (매 N턴)
        AlignmentEstimator: Fact Sheet 기반 목표 정렬 신호 생성 (score + reasoning)
        UncertaintyEstimator: 정보 충분성(certainty) 신호 생성
        Orchestrator: 모든 신호를 종합하여 다음 행동 결정
               → seek_information / recommend / terminate
```

`meal_description`(음식 이름, Turn 0에 공개)과 `meal_ingredient`(재료/조리법, 점진적 공개)의 분리를 통해 AlignmentEstimator가 단번에 모든 정보를 받는 것을 방지하고, **증거 축적 과정**이 코칭 대화의 자연스러운 흐름 속에 담기도록 합니다.

---

## 대화 Phase (Phase-based Flow)

Orchestrator가 대화 Phase를 관리하며, 각 Phase에서 허용되는 행동이 제한됩니다:

```
info_seeking → assessment → rec_info_seeking → recommending ─┬─ [수락] → motivational_ending → terminated
                                                              └─ [거절] → negotiation → motivational_ending → terminated
```

| Phase | 설명 | 허용 행동 |
|-------|------|----------|
| `info_seeking` | 식사 정보 수집 | seek_meal_info, assess_meal, terminate |
| `assessment` | 식사 평가 | assess_meal, terminate |
| `rec_info_seeking` | 추천을 위한 사용자 선호도 수집 | seek_recommendation_info, recommend, terminate |
| `recommending` | 식사 개선 추천 | recommend, seek_recommendation_info, motivational_close, terminate |
| `negotiation` | 사용자 거절 후 대안 탐색 | seek_recommendation_info, recommend, motivational_close, terminate |
| `motivational_ending` | 동기부여 마무리 | motivational_close, terminate |
| `terminated` | 대화 종료 | — |

---

## 다중 식사 세션 (Multi-meal)

인터랙티브 모드(`code_interactive/`)에서는 한 식사 세션이 종료된 후 **다음 식사 세션을 시작**할 수 있습니다. Memorizer가 이전 세션의 사용자 프로필(선호도·알레르기·식이 제한)과 과거 식사 요약을 새 세션으로 이어받습니다.

```
세션 1 (아침) → 종료 → Memorizer 프로필 유지
    ↓
세션 2 (점심) → 종료 → Memorizer 프로필 유지 + 아침 요약 포함
    ↓
세션 3 (저녁) → Memorizer가 아침·점심 요약을 MealRecommender에 제공
```

---

## 프로젝트 구조

```
micro-coaching-simulator/
├── code/                         # 배치 시뮬레이션 (자동화, 데이터셋 기반)
│   ├── config.py                 # SimulationConfig — 단일 설정 소스 (CLI 없음)
│   ├── run_simulation.py         # 실행 진입점
│   ├── run_simulation.sh         # GPU 선택 셸 스크립트
│   ├── core/
│   │   ├── memory.py             # ConversationBuffer, SharedConversationHistory
│   │   └── simulation.py         # 단일/배치 대화 루프 오케스트레이터
│   ├── models/
│   │   ├── information_seeker.py # InformationSeeker — LLM 기반 질문 생성
│   │   ├── user.py               # UserModel — LLM 기반 시뮬레이션 사용자
│   │   ├── alignment_estimator.py # AlignmentEstimator — 영양 목표 정렬 신호
│   │   ├── uncertainty_estimator.py # UncertaintyEstimator — 확신도 신호
│   │   ├── meal_tracker.py       # MealTrackerModel — 식사 정보 구조화 추출
│   │   ├── dialog_summarizer.py  # DialogSummarizerModel — 대화 흐름 요약
│   │   ├── orchestrator.py       # Orchestrator — 다음 행동 종합 결정
│   │   ├── meal_recommender.py   # MealRecommender — 식사 개선 추천
│   │   ├── guardrail.py          # Guardrail — LLM 기반 양방향 안전 필터
│   │   └── memorizer.py          # Memorizer — 다중 세션 사용자 프로필 관리
│   ├── utils/
│   │   ├── llm_utils.py          # vLLM 기반 모델 로딩·배치 생성
│   │   └── io_utils.py           # 데이터 로딩·JSON 저장
│   └── _*.py                     # 레거시 스크립트 (이전 버전 참조용)
│
├── code_interactive/             # 실시간 웹 UI (실제 사용자 ↔ Coach LLM)
│   ├── app.py                    # FastAPI 서버, HTTP 엔드포인트
│   ├── session_manager.py        # 세션별 상태 관리, LLM 오케스트레이션
│   ├── config_interactive.py     # code/config.py 값을 읽어 서버에 적용
│   ├── utils/
│   │   ├── llm_utils.py          # llama-cpp-python 기반 LLM 유틸리티
│   │   └── llm_chatgpt.py        # LangGraph + ChatGPT API 기반 LLM 유틸리티
│   ├── templates/
│   │   └── index.html            # 4-스크린 SPA
│   ├── static/
│   │   ├── style.css             # UI 스타일
│   │   └── script.js             # 프론트엔드 로직
│   ├── requirements.txt          # 의존성 (FastAPI, llama-cpp-python, langchain-openai 등)
│   └── start.sh                  # 서버 실행 스크립트
│
├── data/
│   ├── df_normal_without_test_string.csv   # 메인 식사 데이터셋
│   ├── df_normal.csv
│   ├── additional/               # AlignmentEstimator 리소스
│   │   ├── goal_def.json         # 목표별 정의 및 달성 기준
│   │   ├── expert_workflow.json  # 전문가 평가 워크플로우
│   │   └── output_format_inst_*.txt  # AlignmentEstimator 출력 포맷 (binary/0-1/0-100)
│   └── main/                     # 목표별 Train/Test 분할 데이터
│       ├── drink_water/
│       ├── half_fruits_vegetables/
│       ├── lean_protein/
│       └── one_fourth_carbs/
│
└── results/                      # 시뮬레이션 결과 출력 디렉토리
```

---

## `code/` — 배치 시뮬레이션

크라우드소싱된 식사 데이터를 입력으로 받아, **InformationSeeker → User → MealTracker → DialogSummarizer → AlignmentEstimator** 의 간소화된 흐름으로 대화를 자동 반복 생성합니다. Orchestrator, MealRecommender, UncertaintyEstimator, Guardrail, Memorizer는 코드에 정의되어 있으나 배치 시뮬레이션에서는 사용되지 않으며, 인터랙티브 모드(`code_interactive/`)에서만 활용됩니다. 논문 실험에 필요한 대규모 합성 코칭 대화 데이터셋을 효율적으로 구축하는 것이 목적입니다.

**핵심 특징:**
- vLLM `batch_generate()`로 N건의 대화를 턴 단위로 병렬 처리 (GPU 가동률 극대화)
- 단일/배치 두 가지 실행 모드 지원
- AlignmentEstimator가 aligned 판정 시 해당 대화만 조기 종료 (나머지 계속 진행)
- MealTracker가 누적한 Fact Sheet를 AlignmentEstimator 입력으로 사용
- DialogSummarizer가 생성한 요약을 InformationSeeker/User 시스템 프롬프트에 주입
- InformationSeeker가 직접 질문을 생성하고 AlignmentEstimator가 종료를 판단하는 간소화된 구조

> 자세한 내용은 **[code/README.md](code/README.md)** 를 참조하세요.

---

## `code_interactive/` — 실시간 인터랙션 UI

실제 사람이 AI 코치와 대화하는 환경을 제공합니다. 브라우저 기반 SPA로 두 가지 모드를 지원하며, 로컬 Gemma GGUF 모델(처리: llama-cpp-python) 또는 ChatGPT API(LangGraph 경유)를 세션별로 선택할 수 있습니다. InformationSeeker/User와 AlignmentEstimator가 독립적으로 LLM 백엔드를 선택할 수 있습니다.

| 모드 | 설명 |
|------|------|
| **Simulating Chat** | AI끼리 완전 자율 대화 — 배치 시뮬레이션을 웹에서 시각적으로 관찰 |
| **Custom Chat** | 실제 사람이 AI 코치와 직접 대화 — 목표·식사 정보를 입력하고 코치 질문에 응답 |

**안전 필터**: Guardrail이 Input Guard(사용자 입력 검증)와 Output Guard(코치 응답 검증)를 LLM 기반으로 수행합니다. 식사 코칭 범위를 벗어나는 입력은 차단되고, Output Guard가 코치 응답을 거부하면 Orchestrator TextGen에 가드 사유를 전달하여 최대 2회 재생성을 시도하며, 재시도 소진 시에만 안전한 대체 메시지로 교체됩니다.

**다중 식사 지원**: 한 세션이 종료되면 "다음 식사 시작" 버튼으로 새 세션을 시작하며, Memorizer가 사용자 프로필(선호도·알레르기·식이 제한)과 과거 식사 요약을 이어받습니다. 세션 시작 시 페르소나(Persona) 설정으로 초기 프로필을 지정할 수 있습니다.

Alignment Tracker 오버레이를 켜면 매 발화 아래에 `🟢 Goal Aligned` / `🔴 Not Aligned` 칩이 표시됩니다. Certainty 칩은 `💡` (확신도 ≥ 0.85) / `🤔` (확신도 < 0.85) 이모지를 사용합니다. Backend Monitoring 패널에서 Meal Tracker, Alignment Tracker, Uncertainty Tracker, Orchestrator의 상세 결과(점수·reasoning)를 턴별로 확인할 수 있습니다.

### 빠른 시작

```bash
conda activate micro-coaching-chatbot
cd micro-coaching-simulator/code_interactive
./start.sh              # 프로덕션, 포트 8000
./start.sh 8080         # 포트 지정
DEV=1 ./start.sh        # 개발 모드 (hot-reload)
```

브라우저에서 `http://127.0.0.1:8000` 을 열면 됩니다.

> 자세한 내용은 **[code_interactive/README.md](code_interactive/README.md)** 를 참조하세요.

---

## 설정 단일 소스

```
code/config.py  (SimulationConfig)
       │
       ├──▶  code/run_simulation.py              (배치 시뮬레이션)
       └──▶  code_interactive/config_interactive.py
                  └──▶  code_interactive/app.py   (웹 서버 — 시작 시 자동 로드)
```

**`max_turns`, `context_window`, `summarize_every`, `meal_track_every`, `stall_exit_turns`, `min_natural_end_turn` 등** 모든 대화 제어 파라미터는 `code/config.py` 한 곳에서 관리됩니다. Interactive 모드의 `config_interactive.py`는 `SimulationConfig` 값을 자동으로 읽어오므로, 설정 변경 후 서버만 재시작하면 됩니다.

---

## 설계 원칙 (Design Principles)

### 코어 설계 (Core Design — D1–D9)

| 원칙 | 설명 | 구현 위치 |
|------|------|----------|
| **D1. Orchestrator 중심** | Orchestrator가 유일한 사용자 대화 인터페이스 — 모든 사용자 대면 텍스트를 TextGen으로 생성 | `orchestrator.py` TextGen |
| **D2. Intent 분석 → 라우팅** | Router가 `user_intent` 필드를 명시적으로 출력한 뒤 action 결정 | `orchestrator.py` Router |
| **D3. IS 자율 질문 생성** | 하드코딩 없이 LLM 기반 구조화 템플릿 생성 (question_type + reasoning) | `information_seeker.py` |
| **D4. Guardrail 양방향** | Input Guard + Output Guard, Output Guard 실패 시 Orchestrator TextGen 재생성 (최대 2회) | `guardrail.py`, `session_manager.py` |
| **D5. Estimator Bundle 조건부** | MealTracker + Alignment + Uncertainty는 IS/MR 호출 시에만 실행 | `session_manager.py` |
| **D6. 추천 전 정보 수집** | IS(`recommendation_info` 모드)로 선호도·알레르기·제약 수집 후 MR 호출 | `information_seeker.py`, `memorizer.py` |
| **D7. Assessment Double-turn** | assessment 피드백(Turn N) + 선호도 질문(Turn N+1) — Coach가 연속 2턴 발화 | `session_manager.py` assess_meal 분기 |
| **D8. Memorizer 다회 세션** | 사용자 프로필(선호도·알레르기·과거 식사)을 세션 간 이어받음 | `memorizer.py`, `continue_session()` |
| **D9. 효율성 + 이모지** | stall-exit로 대화 길이 제한, 표시 이모지: Alignment 🟢/🔴, Certainty 💡/🤔 | `session_manager.py`, `script.js` |

### 대화 설계 (Conversation Design)

| 원칙 | 설명 | 구현 위치 |
|------|------|----------|
| **P1** | Coach(IS)에게 질문 전략 가이드라인(Action Guidelines) 제공 | `config.py` `ACTION_GUIDELINES`, `information_seeker.py` |
| **P2** | 각 에이전트가 자신의 발화 이력을 별도 관리 (Own Buffer) | `memory.py` `ConversationBuffer` |
| **P3** | 슬라이딩 윈도우로 최근 N턴만 LLM 컨텍스트에 포함 | `memory.py` `SharedConversationHistory` |
| **P4** | 윈도우 밖 맥락을 두 종류의 요약으로 보존 | `meal_tracker.py`, `dialog_summarizer.py` |

---

## 지원 영양 목표

| 목표 키 | 설명 |
|---------|------|
| `lean_protein` | 닭고기·생선·콩류 등 저지방 단백질 중심 식사 |
| `half_fruits_vegetables` | 식판의 절반을 과일·채소로 채우기 |
| `one_fourth_carbs` | 식판의 1/4을 복합 탄수화물로 구성 |
| `drink_water` | 주요 음료를 물로 선택 |

---

## AlignmentEstimator 출력 형식

| 형식 | 출력 범위 | 정렬 판정 기준 |
|------|-----------|----------------|
| `binary` (기본) | `{"answer": "0", "reasoning": "..."}` 또는 `{"answer": "1", "reasoning": "..."}` | `1` = aligned |
| `0-1` | `{"answer": "0.0", "reasoning": "..."}` ~ `{"answer": "1.0", "reasoning": "..."}` | ≥ `alignment_threshold` = aligned |
| `0-100` | `{"answer": "0", "reasoning": "..."}` ~ `{"answer": "100", "reasoning": "..."}` | ÷100 후 ≥ `alignment_threshold` = aligned |

모든 형식에서 `reasoning` 필드를 통해 판정 근거를 제공하며, 이전 턴 점수가 존재할 경우 점수 변동 이유를 반드시 포함합니다.

AlignmentEstimator 프롬프트의 scaffold 블록은 `alignment_use_goal_def`와 `alignment_use_workflow` 플래그로 개별 활성화/비활성화할 수 있습니다.

## UncertaintyEstimator 출력 형식

```json
{"reasoning": "...", "certainty_score": 0.XX}
```

- `certainty_score` : 0.0 ~ 1.0 (정보 충분성 기반 확신도)
- `reasoning` : 현재 알려진 정보 vs. 미지 정보 요약 (이전 턴 점수 대비 변동 이유 포함)
- certainty ≥ 0.85 시 대화 종료 조건으로 사용

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
