"""
app.py
──────
Micro-Coaching Simulator Interactive 모드 FastAPI 서버.
/ FastAPI server for Micro-Coaching Simulator interactive mode.

엔드포인트 / Endpoints:
  GET  /                              → 메인 UI 페이지 / Main UI page
  GET  /api/goals                     → 지원 영양 목표 목록 / Supported nutrition goals
  POST /api/session/start             → 새 세션 시작 / Start new session
  POST /api/session/{id}/turn         → 사용자 응답 제출 / Submit user reply
  GET  /api/session/{id}/history      → 전체 대화 기록 조회 / Get full conversation
  DELETE /api/session/{id}            → 세션 종료 / End session

실행 방법 / How to run:
  python app.py
  또는 / or:  uvicorn app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import sys
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

# ── 경로 설정 / Path setup ────────────────────────────────────────────────────
# code_interactive/ 를 먼저 등록해야 utils.llm_utils 가 로컈 llama-cpp 버전을 찾습니다.
# Register code_interactive/ FIRST so utils.llm_utils resolves to the local
# llama-cpp-python version, not the vLLM version in /code.
_HERE     = Path(__file__).resolve().parent   # code_interactive/
_CODE_DIR = _HERE.parent / "code"            # code/

# ── sys.path 강제 설정 ────────────────────────────────────────────────────────
# uvicorn -m이 실행 전에 code_interactive/를 sys.path에 추가하기 때문에
# `if not in` 가드를 쓰면 insert(0)가 건너뛰어집니다.
# → 무조건 insert(0)해서 code_interactive/를 최상위 우선순위로 설정합니다.
sys.path.insert(0, str(_CODE_DIR))   # code/ 등록
sys.path.insert(0, str(_HERE))       # code_interactive/ → 최종 index 0

# ── utils 패키지 캐시 무효화 ─────────────────────────────────────────────────
# code/utils/__init__.py가 이미 sys.modules['utils']에 캐시되어 있으면
# from utils.llm_utils import ...가 code/ 버전을 씁니다. → 전부 제거.
for _key in list(sys.modules.keys()):
    if _key == "utils" or _key.startswith("utils."):
        sys.modules.pop(_key, None)

from utils.llm_utils import load_model   # code_interactive/utils/llm_utils.py

from config_interactive import InteractiveConfig
from session_manager import SessionManager

import asyncio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("micro-coach-interactive")

# ─────────────────────────────────────────────────────────────────────────────
# 전역 상태 / Global state
# ─────────────────────────────────────────────────────────────────────────────

_config:          Optional[InteractiveConfig] = None
_session_manager: Optional[SessionManager]   = None


# ─────────────────────────────────────────────────────────────────────────────
# 앱 수명주기 (모델 로딩) / App lifespan (model loading)
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 LLM을 한 번 로드합니다. / Load LLM once at startup."""
    global _config, _session_manager

    _config = InteractiveConfig()

    logger.info("=" * 60)
    logger.info("[Startup] Micro-Coaching Simulator — Interactive Mode")
    logger.info(f"  GGUF path    : {_config.gguf_path}")
    logger.info(f"  n_gpu_layers : {_config.n_gpu_layers}  (0=CPU only)")
    logger.info(f"  max_turns    : {_config.max_turns}")
    logger.info(f"  judge_min_turn: {_config.judge_min_turn}")
    logger.info("=" * 60)

    # llama-cpp 모델 로드는 동기 블로킹 함수이므로 스레드풀에서 실행합니다.
    # Llama() is blocking — run in a thread so uvicorn binds the port immediately.

    def _load():
        return load_model(
            gguf_path=_config.gguf_path,
            n_ctx=_config.n_ctx,
            n_gpu_layers=_config.n_gpu_layers,
        )

    logger.info("[Startup] Loading GGUF model in background thread…")
    llm = await asyncio.to_thread(_load)

    _session_manager = SessionManager(llm=llm, config=_config)
    logger.info("[Startup] Model loaded. Server ready.")

    yield  # ── 서버 실행 구간 / server is running ─────────────────────────

    logger.info("[Shutdown] Cleaning up sessions.")
    # 필요 시 리소스 해제 / release resources if needed


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI 앱 / FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Micro-Coaching Simulator",
    description="Interactive nutritional micro-coaching chatbot powered by LLM",
    version="1.0.0",
    lifespan=lifespan,
)

# 정적 파일 및 템플릿 / Static files and templates
app.mount("/static", StaticFiles(directory=str(_HERE / "static")), name="static")
templates = Jinja2Templates(directory=str(_HERE / "templates"))


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic 스키마 / Pydantic schemas
# ─────────────────────────────────────────────────────────────────────────────

class StartSessionRequest(BaseModel):
    mode:             str  = Field("custom",   description="'custom' | 'simulation'")
    judge_enabled:    bool = Field(True,        description="Judge AI 활성화 여부")
    nutrition_goal:   str  = Field(...,         description="영양 목표 / Nutrition goal")
    meal_type:        str  = Field("meal",      description="식사 유형 / Meal type")
    meal_description: str  = Field("",          description="음식 이름 목록 / Food item names")
    meal_ingredient:  str  = Field("",          description="재료/조리법 상세 / Ingredient details")


class TurnRequest(BaseModel):
    user_reply: str = Field(..., description="사용자 응답 / User reply")


# ─────────────────────────────────────────────────────────────────────────────
# 라우트 / Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """메인 UI 페이지를 반환합니다. / Serve the main UI page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/status")
async def get_status():
    """서버/모델 준비 상태를 반환합니다. / Returns server readiness."""
    coach_label = user_label = None
    if _config:
        coach_label = _config.coach_llm_repo.split('/')[-1]
        user_label  = _config.user_llm_repo.split('/')[-1]
    return {
        "ready":       _session_manager is not None,
        "coach_label": coach_label,
        "user_label":  user_label,
    }


@app.get("/api/goals")
async def get_goals():
    """
    지원되는 영양 목표 목록을 반환합니다.
    / Returns the list of supported nutrition goals.
    """
    labels = {
        "lean_protein":           "Lean Protein",
        "half_fruits_vegetables": "Half Fruits & Vegetables",
        "one_fourth_carbs":       "One-Fourth Carbs",
        "drink_water":            "Drink Water",
    }
    goals = [
        {"value": g, "label": labels.get(g, g)}
        for g in (_config.supported_goals if _config else list(labels.keys()))
    ]
    return {"goals": goals}


@app.post("/api/session/start")
async def start_session(req: StartSessionRequest):
    """
    새 세션을 시작하고 Coach의 첫 번째 질문을 반환합니다.
    / Starts a new session and returns the Coach's first question.

    Response
    --------
    {
      "session_id":    str,
      "first_question": str,
      "mode":           str,
      "judge_enabled":  bool,
      "nutrition_goal": str,
      "meal_type":      str
    }
    """
    if _session_manager is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Please wait.")

    if req.nutrition_goal not in (_config.supported_goals if _config else []):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported goal: {req.nutrition_goal}. "
                   f"Choose from: {_config.supported_goals}"
        )

    # simulation 모드는 meal_description 필수 / meal_description required for simulation mode
    if req.mode == "simulation" and not req.meal_description.strip():
        raise HTTPException(status_code=400, detail="meal_description is required for simulation mode.")

    try:
        session = _session_manager.create_session(
            nutrition_goal=req.nutrition_goal,
            meal_description=req.meal_description.strip(),
            meal_ingredient=req.meal_ingredient.strip(),
            meal_type=req.meal_type.strip() or "meal",
            mode=req.mode,
            judge_enabled=req.judge_enabled,
        )
    except Exception as e:
        logger.exception("Failed to create session")
        raise HTTPException(status_code=500, detail=str(e))

    first_q = session.turns[0].coach_utterance if session.turns else ""
    logger.info(
        f"[Session {session.session_id[:8]}] Created — "
        f"mode={req.mode} goal={req.nutrition_goal} judge={req.judge_enabled}"
    )

    return {
        "session_id":     session.session_id,
        "first_question": first_q,
        "mode":           session.mode,
        "judge_enabled":  session.judge_enabled,
        "nutrition_goal": session.nutrition_goal,
        "meal_type":      session.meal_type,
    }


@app.post("/api/session/{session_id}/turn")
async def submit_turn(session_id: str, req: TurnRequest):
    """
    사용자 응답을 처리하고 다음 Coach 질문 및 Judge 결과를 반환합니다.
    / Processes user reply and returns next Coach question + Judge result.

    Response
    --------
    {
      "turn_idx":       int,
      "coach_question": str | null,   // null if session ended
      "judge_aligned":  bool | null,
      "judge_score":    float | null,
      "status":         str,          // "active" | "terminated" | "max_turns"
      "aligned_label":  str           // "aligned" | "not_aligned" | "pending"
    }
    """
    if _session_manager is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    if not req.user_reply.strip():
        raise HTTPException(status_code=400, detail="user_reply must not be empty.")

    try:
        result = await asyncio.to_thread(
            _session_manager.submit_reply, session_id, req.user_reply
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.exception(f"Error processing turn for session {session_id}")
        raise HTTPException(status_code=500, detail=str(e))

    logger.info(
        f"[Session {session_id[:8]}] Turn {result['turn_idx']} — "
        f"status={result['status']} aligned={result['judge_aligned']}"
    )
    return result


@app.post("/api/session/{session_id}/sim-step")
async def simulation_step(session_id: str):
    """
    Simulation 모드 전용: AI User 발화 한 턴을 실행하고 결과를 반환합니다.
    / Simulation mode only: execute one AI User turn and return results.

    Response
    --------
    {
      "turn_idx":       int,
      "user_reply":     str,
      "coach_question": str | null,
      "judge_aligned":  bool | null,
      "judge_score":    float | null,
      "status":         str,
      "aligned_label":  str
    }
    """
    if _session_manager is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    try:
        result = await asyncio.to_thread(
            _session_manager.sim_step, session_id
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.exception(f"Error in sim-step for session {session_id}")
        raise HTTPException(status_code=500, detail=str(e))

    logger.info(
        f"[Session {session_id[:8]}] SimStep turn={result['turn_idx']} "
        f"status={result['status']} aligned={result['judge_aligned']}"
    )
    return result


@app.get("/api/session/{session_id}/history")
async def get_history(session_id: str):
    """
    세션의 전체 대화 기록을 반환합니다.
    / Returns the full conversation history of a session.
    """
    if _session_manager is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    history = _session_manager.get_history(session_id)
    if not history:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    return history


@app.delete("/api/session/{session_id}")
async def end_session(session_id: str):
    """
    세션을 종료하고 메모리에서 제거합니다.
    / Ends the session and removes it from memory.
    """
    if _session_manager is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    _session_manager.abandon_session(session_id)
    _session_manager.remove_session(session_id)
    logger.info(f"[Session {session_id[:8]}] Removed")
    return {"ok": True, "session_id": session_id}


# ─────────────────────────────────────────────────────────────────────────────
# 직접 실행 / Direct execution
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = InteractiveConfig()
    uvicorn.run(
        "app:app",
        host=cfg.host,
        port=cfg.port,
        reload=cfg.reload,
    )
