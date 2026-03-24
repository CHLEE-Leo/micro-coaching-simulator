# Micro-Coaching Simulator

> **"[Paper Title]"** (under review)

영양 코칭 대화를 자동으로 생성·평가하고, 실제 사용자와 AI 코치 간의 실시간 상호작용을 실험하기 위한 연구용 프레임워크입니다.

---

## 이 레포지토리의 목적

식사 목표 정렬(Nutritional Goal Alignment) 연구를 위해 두 가지 핵심 기능을 제공합니다.

1. **대화 데이터 생성** — Coach LLM, User LLM, Judge LLM 세 에이전트를 조합하여 크라우드소싱 식사 데이터셋 위에서 대량의 합성 코칭 대화를 자동 생성합니다.
2. **실시간 인터랙션 실험** — 실제 사람이 AI 코치와 대화하면서, Judge AI가 매 발화마다 목표 정렬 여부를 실시간으로 평가합니다.

두 기능은 동일한 모델 구조와 대화 설계를 공유하되, **목적과 운영 방식이 다릅니다.**

---

## 두 폴더 한눈에 비교

|  | `code/` | `code_interactive/` |
|---|---|---|
| **역할** | 배치 시뮬레이션 (논문 실험 데이터 생성) | 웹 기반 실시간 코칭 UI |
| **사용자** | 없음 — 세 LLM이 자율적으로 대화 | 실제 사람이 코치와 직접 대화 |
| **실행 방식** | Python 스크립트 / 셸 배치 | FastAPI 서버 + 브라우저 UI |
| **Judge** | 대화 종료 조건으로 사용 | 매 발화마다 실시간 정렬 칩 표시 |
| **목적** | 합성 데이터셋 생성, 오프라인 분석 | 사용자 경험 실험, 프로토타입 검증 |
| **진입점** | `code/run_simulation.py` | `code_interactive/app.py` |

---

## `code/` — 배치 시뮬레이션

### 역할과 의도

크라우드소싱된 식사 데이터를 입력으로 받아, **Coach → User → Judge** 삼자 대화를 자동으로 반복 생성합니다.  
논문 실험에 필요한 대규모 합성 코칭 대화 데이터셋을 효율적으로 구축하는 것이 목적입니다.

### 대화 구조 (2단계 설계)

```
Turn 0  Coach: "저녁에 무엇을 드셨나요?"
        User : 음식 이름 전체 공개 (meal_description)

Turn 1+ Coach: 각 음식의 재료·조리법·양을 구체적으로 질문
        User : 상세 정보를 점진적으로 공개 (meal_ingredient)
        Judge: 매 턴마다 목표 정렬 여부 평가
               → 확신 도달 시 대화 종료
```

이 설계는 Judge가 단번에 모든 정보를 받아 판단하는 것을 막고,  
**증거 축적 과정**이 코칭 대화의 자연스러운 흐름 속에 담기도록 합니다.

### 개요

- `config.py` (`SimulationConfig`) — 모든 실험 파라미터의 **단일 소스**  
  _(이 파일을 수정하면 배치 시뮬레이션과 웹 서버 양쪽이 동시에 반영됩니다)_
- `core/memory.py` — 공유 대화 히스토리, 슬라이딩 윈도우, 롤링 요약
- `core/simulation.py` — 단일/배치 대화 루프 실행기
- `models/` — Coach·User·Judge 모델 래퍼
- `utils/` — LLM 로딩·생성·요약, 데이터 입출력

> 자세한 내용은 **[code/README.md](code/README.md)** 를 참조하세요.

---

## `code_interactive/` — 실시간 인터랙션 UI

### 역할과 의도

실제 사람이 AI 코치와 대화하는 환경을 제공합니다.  
브라우저 기반 SPA(Single-Page Application)로 두 가지 모드를 지원합니다.

| 모드 | 설명 |
|------|------|
| **Simulating Chat** | AI끼리 완전 자율 대화 — 웹에서 배치 시뮬레이션을 시각적으로 관찰 |
| **Custom Chat** | 실제 사람이 AI 코치와 직접 대화 — 목표·식사 정보를 입력하고 코치 질문에 응답 |

Judge AI 오버레이를 켜면 매 발화 아래에 `✓ Goal Aligned` / `✗ Not Aligned` 칩이 표시됩니다.

### 빠른 시작

```bash
conda activate micro-coaching-chatbot
cd micro-coaching-simulator/code_interactive
./start.sh              # 프로덕션, 포트 8000
./start.sh 8080         # 포트 지정
DEV=1 ./start.sh        # 개발 모드 (hot-reload)
```

브라우저에서 `http://127.0.0.1:8000` 을 열면 됩니다.

### 개요

- `app.py` — FastAPI 서버, HTTP 엔드포인트 정의
- `session_manager.py` — 세션별 상태 관리, LLM 오케스트레이션
- `config_interactive.py` — `code/config.py` 의 값을 읽어 서버에 적용  
  _(대화 파라미터를 바꾸려면 `code/config.py` 만 수정 후 서버 재시작)_
- `templates/index.html` + `static/` — 4-스크린 SPA UI
- `models/` — 배치용 모델 클래스를 인터랙티브 모드에 맞게 래핑

> 자세한 내용은 **[code_interactive/README.md](code_interactive/README.md)** 를 참조하세요.

---

## 설정 단일 소스

```
code/config.py  (SimulationConfig)
       │
       ├──▶  code/run_simulation.py       (배치 시뮬레이션)
       └──▶  code_interactive/app.py      (웹 서버 — 시작 시 자동 로드)
```

**`max_turns`, `context_window`, `stall_exit_turns`, `min_natural_end_turn` 등**  
모든 대화 제어 파라미터는 `code/config.py` 한 곳에서 관리됩니다.

---

## 지원 영양 목표

| 목표 키 | 설명 |
|---------|------|
| `lean_protein` | 닭고기·생선·콩류 등 저지방 단백질 중심 식사 |
| `half_fruits_vegetables` | 식판의 절반을 과일·채소로 채우기 |
| `one_fourth_carbs` | 식판의 1/4을 복합 탄수화물로 구성 |
| `drink_water` | 주요 음료를 물로 선택 |

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
