# Micro-Coaching Simulator — 인터랙티브 웹 UI (`code_interactive/`)

> 실시간으로 AI 코치와 대화 시뮬레이션을 진행하거나, 사용자가 직접 코치 역할로 참여할 수 있는 FastAPI 기반 웹 인터페이스입니다.

---

## 개요

`code/`의 배치 시뮬레이션과 동일한 10-에이전트 아키텍처를 기반으로 하되, **llama-cpp-python** (로컬 GGUF) 또는 **ChatGPT API** (LangGraph 경유) 백엔드를 선택하여 단일 대화를 실시간으로 시뮬레이션합니다. Guardrail이 사용자 입력과 코치 응답을 LLM 기반으로 검증하고, Memorizer가 다중 식사 세션 간 사용자 프로필을 유지합니다. 두 가지 모드를 지원합니다:

| 모드 | 설명 |
|------|------|
| **Simulating Chat** | Coach, User 모두 AI가 담당 — 자동 시뮬레이션 |
| **Custom Chat** | 사용자가 Coach 역할로 직접 참여 — AI User와 대화 |

---

## 아키텍처

Orchestrator가 대화의 유일한 커뮤니케이션 허브입니다. InformationSeeker와 MealRecommender는 Orchestrator에게 구조화된 템플릿을 제공하는 서브 에이전트이며, 사용자와 직접 소통하지 않습니다.

```
Browser (SPA)
   │
   ▼ REST API (JSON)
┌────────────────────────────────────────────┐
│  FastAPI  (app.py)                         │
│  ├─ lifespan: 모델 사전 로딩              │
│  ├─ 세션 관리 (session_manager.py)        │
│  ├─ 정적 파일 (Jinja2 + static/)         │
│  └─ CORS 미들웨어                         │
├────────────────────────────────────────────┤
│  SessionManager                            │
│  ├─ Orchestrator (중앙 허브)               │
│  │   ├─ Router: user_intent 분석 → 다음 행동 결정 │
│  │   ├─ TextGen: LLM 기반 자연어 생성 (서브에이전트 템플릿 → 자연어) │
│  │   └─ Phase 관리                         │
│  ├─ Orchestrator 서브 에이전트:              │
│  │   ├─ InformationSeeker (질문 템플릿)    │
│  │   └─ MealRecommender  (추천 템플릿)    │
│  ├─ 인프라 서비스 (SessionManager 직접 호출): │
│  │   ├─ AlignmentEstimator (정렬 신호)     │
│  │   ├─ UncertaintyEstimator (확신도 신호) │
│  │   ├─ MealTracker (Fact Sheet 추출)     │
│  │   └─ DialogSummarizer (대화 요약)      │
│  ├─ Guardrail (Input/Output Guard)        │
│  ├─ Memorizer (다중 세션 프로필)          │
│  ├─ User / UserModel (사람 또는 AI)       │
│  └─ submit_reply() / sim_step()            │
├────────────────────────────────────────────┤
│  LLM 백엔드 (세션별 선택)                  │
│  ├─ utils/llm_utils.py    (llama-cpp GGUF)│
│  └─ utils/llm_chatgpt.py  (LangGraph+GPT)│
└────────────────────────────────────────────┘
```

### 웹 UI

```
┌─────────────────────────────────────────────────────────┐
│ Micro-Coaching Simulator                                │
├─────────┬───────────────────────────────────────────────┤
│ 좌측    │  채팅 영역                                   │
│ 사이드  │  ┌─────────────────────────────────┐         │
│ 바      │  │ Coach: "What are you having ..." │         │
│         │  │ User: "I'm thinking about ..."   │         │
│ • Goal  │  │ Coach: "How was it prepared?"     │         │
│ • Meal  │  │ User: "It was grilled ..."        │         │
│ • Mode  │  └─────────────────────────────────┘         │
│ • Start │                                               │
│         │  ┌──────────────────────┬────────┐           │
│ Status  │  │ 메시지 입력 (Custom) │ Send   │           │
│ Panel   │  └──────────────────────┴────────┘           │
│         │  [Simulate Next Turn] (Simulating)            │
└─────────┴───────────────────────────────────────────────┘
```

---

## 디렉토리 구조

```
code_interactive/
├── app.py                   # FastAPI 서버, 라우트 정의, 모델 사전 로딩
├── config_interactive.py    # InteractiveConfig — SimulationConfig 래핑
├── session_manager.py       # SessionManager — 세션 상태, 턴 진행, 종료 판단
├── requirements.txt         # Python 의존성
├── start.sh                 # 서버 시작 스크립트 (Conda + uvicorn)
│
├── utils/
│   ├── __init__.py
│   ├── llm_utils.py         # llama-cpp-python 기반 generate_response()
│   │                        # load_model() — GGUF 모델 로딩
│   └── llm_chatgpt.py       # LangGraph + ChatGPT API 기반
│                             # load_model() / generate_response() / batch_generate()
│
├── static/
│   ├── script.js             # SPA 클라이언트 로직 (fetch API)
│   └── style.css             # UI 스타일
│
└── templates/
    └── index.html            # Jinja2 메인 템플릿 (SPA 셸)
```

---

## 실행 방법

### 1. 설치

```bash
cd code_interactive
pip install -r requirements.txt
```

`requirements.txt`에는 다음이 포함됩니다:
```
llama-cpp-python
fastapi
uvicorn
jinja2
python-multipart
pydantic
langchain-openai>=0.3.0     # ChatGPT 백엔드
langgraph>=0.2.0
python-dotenv>=1.0.0
```

### 2. 설정

모든 설정은 `code/config.py`의 `SimulationConfig`에서 가져옵니다 (Single Source of Truth). `config_interactive.py`는 해당 값들을 래핑하고, 인터랙티브 전용 설정을 추가합니다.

주요 설정 항목:
- `goal` — 영양 목표 (lean_protein, half_fruits_vegetables, one_fourth_carbs, drink_water)
- `coach_llm_repo` — 사용할 LLM 모델 (GGUF 파일은 별도 다운로드 필요)
- `context_window` — 슬라이딩 윈도우 크기
- `alignment_min_turn` — AlignmentEstimator 평가 시작 최소 턴
- `alignment_output_format` — AlignmentEstimator 출력 형식 (binary, 0-1, 0-100)
- `llm_provider` — InformationSeeker/User LLM 백엔드 선택 (`"gemma"` | `"chatgpt"`)
- `alignment_llm_provider` — AlignmentEstimator LLM 백엔드 선택 (`"gemma"` | `"chatgpt"`)
- `chatgpt_model` — ChatGPT 모델명 (기본 `"gpt-5.2"`)

### 3. ChatGPT 설정 (선택)

ChatGPT API를 사용하려면 프로젝트 루트에 `.env` 파일을 생성하세요:

```bash
# micro-coaching-simulator/.env
OPENAI_API_KEY=sk-...
```

서버 시작 시 자동으로 ChatGPT 클라이언트가 초기화됩니다. `.env`가 없거나 키가 유효하지 않으면 ChatGPT 옵션이 비활성화되며, Gemma(로컬 GGUF)만 사용할 수 있습니다.

### 4. 서버 시작

```bash
# start.sh 사용
bash start.sh

# 또는 직접 실행
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

---

## API 엔드포인트

### 시스템

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | 메인 웹 UI (Jinja2 렌더링) |
| `GET` | `/api/status` | 서버 상태 (모델 로딩 여부) |
| `GET` | `/api/goals` | 지원 목표 목록 반환 |

### 세션 관리

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/session/start` | 새 세션 생성 (goal, meal_id, mode 지정) |
| `POST` | `/api/session/{id}/turn` | Custom Chat: 사용자 메시지 제출 → AI User 응답 |
| `POST` | `/api/session/{id}/sim-step` | Simulating Chat: AI가 Coach+User 한 턴 자동 진행 |
| `POST` | `/api/session/{id}/continue` | 다음 식사 세션 시작 (Memorizer 프로필 이어받기) |
| `GET` | `/api/session/{id}/history` | 세션 대화 이력 조회 |
| `DELETE` | `/api/session/{id}` | 세션 삭제 |

### 프롬프트 프리뷰

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/alignment-preview` | AlignmentEstimator 프롬프트 프리뷰 (Fact Sheet + 최근 턴 포함) |
| `GET` | `/api/coach-preview` | Coach 프롬프트 프리뷰 |
| `GET` | `/api/user-preview` | User 프롬프트 프리뷰 |

---

## 세션 라이프사이클

```
POST /api/session/start (goal, meal_id, mode)
│
├─ 세션 생성 → meal data 로딩 → SharedConversationHistory 초기화
│
├─ Turn 0: Coach 오프닝 질문 자동 생성
│   └─ "What are you thinking of having for {meal_type}?"
│
├─ Turn 1 ~ N:
│   ├─ [Custom]      POST /turn     → 사용자 코치 메시지 + AI User 응답
│   ├─ [Simulating]  POST /sim-step → AI Coach + AI User 자동 진행
│   │
│   ├─ Guardrail: Input Guard (사용자 입력 검증) + Output Guard (코치 응답 검증, 실패 시 최대 2회 재생성)
│   ├─ MealTracker: Fact Sheet 업데이트 (meal_track_every 간격)
│   ├─ DialogSummarizer: 대화 요약 업데이트 (summarize_every 간격)
│   ├─ AlignmentEstimator: 목표 정렬 신호 (alignment_min_turn 이후, score + reasoning)
│   ├─ UncertaintyEstimator: 확신도(certainty) 신호
│   └─ Orchestrator: 모든 신호를 종합하여 다음 행동 결정
│
└─ 종료 조건:
    ├─ Orchestrator → terminate 판단
    ├─ max_turns 도달
    ├─ stall_exit_turns 연속 비응답
    └─ AI User [END] 태그 감지 (자연스러운 종료 멘트)

종료 후 "다음 식사 시작" 버튼을 통해 `POST /api/session/{id}/continue`로 새 세션을 시작할 수 있으며, Memorizer가 사용자 프로필과 과거 식사 요약을 이어받습니다.
```

---

## vLLM과의 차이점

| 항목 | `code/` (배치) | `code_interactive/` (인터랙티브) |
|------|---------------|--------------------------------|
| **LLM 백엔드** | vLLM (GPU 배치 추론) | llama-cpp-python (GGUF) 또는 ChatGPT API (LangGraph) |
| **처리 방식** | N개 대화 동시 병렬 배치 | 단일 세션 순차 턴 |
| **실행 방식** | `python run_simulation.py` | `uvicorn app:app` |
| **사용자 인터페이스** | 없음 (자동 실행, JSON 결과) | 웹 브라우저 (SPA) |
| **모드** | 시뮬레이션 전용 | Simulating + Custom Chat |
| **요약 방식** | 주로 Full summarisation | Incremental summarisation |
| **설정** | `config.py` 직접 편집 | `config.py`에서 읽어옴 |

---

## 주요 구현 상세

### 모델 사전 로딩 (Lifespan)

`app.py`의 FastAPI lifespan 이벤트에서 InformationSeeker, User, AlignmentEstimator 모델을 서버 시작 시 미리 로딩합니다. MealTracker와 DialogSummarizer는 InformationSeeker 모델을 공유합니다.

### Incremental Summarisation

인터랙티브 모드에서는 매 업데이트 시 전체 대화를 다시 요약하는 대신, 이전 요약 + 새 턴만으로 점진적 업데이트를 수행합니다.

```
[이전 Fact Sheet] + [새로운 턴 2개] → [업데이트된 Fact Sheet]
[이전 Dialog Summary] + [새로운 턴 3개] → [업데이트된 Dialog Summary]
```

### 자연 종료 감지

AI User가 `<END_OF_CONVERSATION>` 토큰을 생성하면, `min_natural_end_turn` 이후부터 대화가 자연 종료됩니다.

### Stall 감지

User가 연속으로 `stall_exit_turns`번 비응답(non-answer)을 하면, 현재 phase에 따라 다음과 같이 처리됩니다:

| 현재 Phase | Stall 시 동작 |
|-------------|---------------|
| `info_seeking` | Orchestrator Router 없이 `assess_meal` 강제 전환 |
| `rec_info_seeking` | Orchestrator Router 없이 `recommend` 강제 전환 |
| `recommending` / `negotiation` | Orchestrator Router 없이 `motivational_close` 강제 전환 (동기부여 마무리 후 종료) |
| 그 외 | 마무리 발화 후 세션 종료 (`terminated_by = "stall_exit"`) |

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
