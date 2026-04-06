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

    # n_gpu_layers=0 (CPU 전용) 일 때 CUDA context 가 모든 GPU 에 할당되는 것을 방지.
    # llama-cpp-python 이 CUDA 빌드일 경우 라이브러리 로딩 시점에 보이는 GPU 마다
    # CUDA context 를 생성하므로, 모델 로드 전에 GPU 를 숨겨야 합니다.
    import os
    if _config.n_gpu_layers == 0 and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        logger.info("[Startup] n_gpu_layers=0 → CUDA_VISIBLE_DEVICES='' (GPU hidden)")

    logger.info("=" * 60)
    logger.info("[Startup] Micro-Coaching Simulator — Interactive Mode")
    logger.info(f"  Default LLM  : {_config.llm_provider}")
    logger.info(f"  GGUF path    : {_config.gguf_path}")
    logger.info(f"  n_gpu_layers : {_config.n_gpu_layers}  (0=CPU only)")
    logger.info(f"  n_threads    : {_config.n_threads}")
    logger.info(f"  max_turns    : {_config.max_turns}")
    logger.info(f"  alignment_min_turn: {_config.alignment_min_turn}")
    logger.info("=" * 60)

    # ── Gemma (llama-cpp) 모델 로드 ──────────────────────────────────────
    # llama-cpp 모델 로드는 동기 블로킹 함수이므로 스레드풀에서 실행합니다.
    # Llama() is blocking — run in a thread so uvicorn binds the port immediately.
    def _load():
        return load_model(
            gguf_path=_config.gguf_path,
            n_ctx=_config.n_ctx,
            n_gpu_layers=_config.n_gpu_layers,
            n_threads=_config.n_threads,
        )

    logger.info("[Startup] Loading GGUF model in background thread…")
    gemma_llm = await asyncio.to_thread(_load)

    # ── ChatGPT 클라이언트 (lazy: 세션 생성 시 초기화) ──────────────────
    chatgpt_client = None
    try:
        from utils.llm_chatgpt import load_model as load_chatgpt_model
        chatgpt_client = load_chatgpt_model(_config.chatgpt_model)
        logger.info(f"[Startup] ChatGPT client ready (model={_config.chatgpt_model})")
    except Exception as exc:
        logger.warning(f"[Startup] ChatGPT not available: {exc}")

    _session_manager = SessionManager(
        llm=gemma_llm,
        chatgpt_client=chatgpt_client,
        config=_config,
    )
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
    alignment_enabled:    bool = Field(True,        description="Alignment Tracker 활성화 여부")
    nutrition_goal:   str  = Field(...,         description="영양 목표 / Nutrition goal")
    meal_type:        str  = Field("meal",      description="식사 유형 / Meal type")
    meal_description: str  = Field("",          description="음식 이름 목록 / Food item names")
    meal_ingredient:  str  = Field("",          description="재료/조리법 상세 / Ingredient details")
    llm_provider:     Optional[str] = Field(None, description="Coach LLM 프로바이더: gemma | chatgpt")
    user_llm_provider: Optional[str] = Field(None, description="User LLM 프로바이더: gemma | chatgpt")
    alignment_llm_provider: Optional[str] = Field(None, description="Alignment Tracker LLM 프로바이더: gemma | chatgpt")
    coach_conversation_mode: Optional[str] = Field(None, description="Coach 대화 모드: open-ended | template-based")
    uncertainty_tracking: Optional[bool] = Field(None, description="Uncertainty Tracking 활성화 여부")
    dialog_summarization: Optional[bool] = Field(None, description="Dialogue Summarization 활성화 여부")
    alignment_use_goal_def:  Optional[bool] = Field(None, description="Alignment Tracker에 goal_definition 포함 여부")
    alignment_use_workflow:  Optional[bool] = Field(None, description="Alignment Tracker에 expert workflow 포함 여부")
    alignment_output_format: Optional[str]  = Field(None, description="Alignment Tracker 출력 포맷: binary | 0-1 | 0-100")
    persona_preferences:  Optional[list] = Field(None, description="사용자 식품 선호도 / User food preferences")
    persona_allergies:    Optional[list] = Field(None, description="사용자 알레르기 / User allergies")
    persona_restrictions: Optional[list] = Field(None, description="사용자 식이 제한 / User dietary restrictions")


class ContinueSessionRequest(BaseModel):
    nutrition_goal:   str = Field(...,    description="새 식사의 영양 목표 / New meal nutrition goal")
    meal_type:        str = Field("meal", description="식사 유형 / Meal type")
    meal_description: str = Field("",     description="음식 이름 / Food item names")
    meal_ingredient:  str = Field("",     description="재료/조리법 / Ingredient details")


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
    chatgpt_available = False
    if _session_manager is not None:
        chatgpt_available = _session_manager._chatgpt_client is not None
    return {
        "ready":             _session_manager is not None,
        "chatgpt_available": chatgpt_available,
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
      "alignment_enabled":  bool,
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
        session = await asyncio.to_thread(
            _session_manager.create_session,
            nutrition_goal=req.nutrition_goal,
            meal_description=req.meal_description.strip(),
            meal_ingredient=req.meal_ingredient.strip(),
            meal_type=req.meal_type.strip() or "meal",
            mode=req.mode,
            alignment_enabled=req.alignment_enabled,
            llm_provider=req.llm_provider,
            user_llm_provider=req.user_llm_provider,
            alignment_llm_provider=req.alignment_llm_provider,
            coach_conversation_mode=req.coach_conversation_mode,
            uncertainty_tracking=req.uncertainty_tracking,
            dialog_summarization=req.dialog_summarization,
            alignment_use_goal_def=req.alignment_use_goal_def,
            alignment_use_workflow=req.alignment_use_workflow,
            alignment_output_format=req.alignment_output_format,
            persona_preferences=req.persona_preferences,
            persona_allergies=req.persona_allergies,
            persona_restrictions=req.persona_restrictions,
        )
    except Exception as e:
        logger.exception("Failed to create session")
        raise HTTPException(status_code=500, detail=str(e))

    first_q = session.turns[0].coach_utterance if session.turns else ""
    logger.info(
        f"[Session {session.session_id[:8]}] Created — "
        f"mode={req.mode} goal={req.nutrition_goal} alignment={req.alignment_enabled}"
    )

    # 프로바이더에 따른 모델 이름 레이블
    def _provider_label(prov: str) -> str:
        if prov == "chatgpt":
            return _config.chatgpt_model
        repo = _config.coach_llm_repo
        return repo.split("/")[-1] if "/" in repo else repo

    return {
        "session_id":     session.session_id,
        "first_question": first_q,
        "mode":           session.mode,
        "alignment_enabled":  session.alignment_enabled,
        "nutrition_goal":  session.nutrition_goal,
        "meal_type":      session.meal_type,
        "llm_provider":   session.llm_provider,
        "user_llm_provider": session.user_llm_provider,
        "alignment_llm_provider": session.alignment_llm_provider,
        "coach_label":     _provider_label(session.llm_provider),
        "user_label":      _provider_label(session.user_llm_provider),
        "coach_conversation_mode": session.coach_conversation_mode,
        "uncertainty_tracking": session.uncertainty_tracking,
        "dialog_summarization": session.dialog_summarization,
    }


@app.post("/api/session/{session_id}/turn")
async def submit_turn(session_id: str, req: TurnRequest):
    """
    사용자 응답을 처리하고 다음 Coach 질문 및 Alignment Tracker 결과를 반환합니다.
    / Processes user reply and returns next Coach question + Alignment Tracker result.

    Response
    --------
    {
      "turn_idx":       int,
      "coach_question": str | null,   // null if session ended
      "alignment_aligned":  bool | null,
      "alignment_score":    float | null,
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
    except KeyError as e:
        if "Session not found" in str(e):
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
        logger.exception(f"Unexpected KeyError in submit_turn for session {session_id}")
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.exception(f"Error processing turn for session {session_id}")
        raise HTTPException(status_code=500, detail=str(e))

    logger.info(
        f"[Session {session_id[:8]}] Turn {result['turn_idx']} — "
        f"status={result['status']} aligned={result['alignment_aligned']}"
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
      "alignment_aligned":  bool | null,
      "alignment_score":    float | null,
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
    except KeyError as e:
        if "Session not found" in str(e):
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
        logger.exception(f"Unexpected KeyError in sim-step for session {session_id}")
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.exception(f"Error in sim-step for session {session_id}")
        raise HTTPException(status_code=500, detail=str(e))

    logger.info(
        f"[Session {session_id[:8]}] SimStep turn={result['turn_idx']} "
        f"status={result['status']} aligned={result['alignment_aligned']}"
    )
    return result


@app.post("/api/session/{session_id}/continue")
async def continue_session(session_id: str, req: ContinueSessionRequest):
    """
    이전 세션의 사용자 프로필(Memorizer)을 이어받아 새 식사 세션을 시작합니다.
    / Starts a new meal session carrying over the Memorizer from a previous session.

    Response
    --------
    {
      "session_id":    str,    // new session ID
      "first_question": str,
      "previous_session_id": str,
      "previous_meals": int   // number of past meals in Memorizer
    }
    """
    if _session_manager is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    try:
        new_session = await asyncio.to_thread(
            _session_manager.continue_session,
            previous_session_id=session_id,
            nutrition_goal=req.nutrition_goal,
            meal_description=req.meal_description.strip(),
            meal_ingredient=req.meal_ingredient.strip(),
            meal_type=req.meal_type.strip() or "meal",
        )
    except KeyError as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=f"Previous session not found: {session_id}")
        logger.exception(f"Unexpected KeyError in continue_session for {session_id}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception(f"Error continuing session {session_id}")
        raise HTTPException(status_code=500, detail=str(e))

    first_q = new_session.turns[0].coach_utterance if new_session.turns else ""
    past_meals = len(new_session.memorizer._profile["past_meals"]) if new_session.memorizer else 0

    logger.info(
        f"[Session {new_session.session_id[:8]}] Continued from {session_id[:8]} — "
        f"past_meals={past_meals} goal={req.nutrition_goal}"
    )

    return {
        "session_id":          new_session.session_id,
        "first_question":      first_q,
        "previous_session_id": session_id,
        "previous_meals":      past_meals,
        "mode":                new_session.mode,
        "nutrition_goal":      new_session.nutrition_goal,
        "meal_type":           new_session.meal_type,
    }


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
# Alignment Tracker 프롬프트 미리보기 / Alignment Tracker prompt preview
# ─────────────────────────────────────────────────────────────────────────────

from models.alignment_estimator import (
    _build_alignment_system_prompt,
    _get_goal_spec,
    _get_workflow_text,
    _load_output_format,
)

from models.information_seeker import _COACH_SYSTEM_BASE, _COACH_ACTION_GUIDELINES_BLOCK
from models.user  import _USER_SYSTEM_BASE
from config import ACTION_GUIDELINES


@app.get("/api/alignment-preview")
async def alignment_preview(
    nutrition_goal: str = "lean_protein",
    goal_def: bool = True,
    workflow: bool = True,
    output_format: str = "binary",
):
    """
    현재 선택된 옵션으로 Alignment Tracker 시스템 프롬프트 + 유저 메시지를 반환합니다.
    / Returns the Alignment Tracker system prompt and user message built from selected options.
    """
    goal_spec = _get_goal_spec(nutrition_goal)
    goal_definition = goal_spec.get("definition", "") if goal_def else ""
    workflow_text = _get_workflow_text(nutrition_goal) if workflow else ""

    valid_formats = {"binary", "0-1", "0-100"}
    if output_format not in valid_formats:
        output_format = "binary"
    output_fmt = _load_output_format(output_format)

    system_prompt = _build_alignment_system_prompt(
        nutrition_goal=nutrition_goal,
        goal_definition=goal_definition,
        workflow_text=workflow_text,
        output_format_inst=output_fmt,
    )

    # User message — context is always a filler since the actual value comes from conversation
    nutrition_goal_display = nutrition_goal.replace("_", " ")
    user_message = (
        "[context]\n(meal description)\n\n"
        f"[question]\nDoes this meal align with the goal of {nutrition_goal_display}?"
    )

    return {
        "system_prompt":     system_prompt,
        "user_message":      user_message,
        "goal_def_text":     goal_definition,
        "workflow_text":     workflow_text,
        "output_format_text": output_fmt,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Coach 프롬프트 미리보기 / Coach prompt preview
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/coach-preview")
async def coach_preview(
    nutrition_goal: str = "lean_protein",
    meal_type: str = "dinner",
    conversation_mode: str = "template-based",
):
    """Coach 시스템 프롬프트를 미리 반환합니다."""
    system_prompt = _COACH_SYSTEM_BASE.format(
        nutrition_goal=nutrition_goal,
        meal_type=meal_type,
    )
    action_guidelines_text = ""
    if conversation_mode == "template-based":
        action_guidelines_text = ACTION_GUIDELINES
        system_prompt += _COACH_ACTION_GUIDELINES_BLOCK.format(
            action_guidelines=ACTION_GUIDELINES
        )

    return {
        "system_prompt": system_prompt,
        "action_guidelines_text": action_guidelines_text,
    }


# ─────────────────────────────────────────────────────────────────────────────
# User 프롬프트 미리보기 / User prompt preview
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/user-preview")
async def user_preview(
    nutrition_goal: str = "lean_protein",
    meal_description: str = "",
    meal_ingredient: str = "",
):
    """User 시스템 프롬프트를 미리 반환합니다."""
    ingredient_block = ""
    if meal_ingredient:
        ingredient_block = "\nIngredient details                 : " + meal_ingredient

    system_prompt = _USER_SYSTEM_BASE.format(
        nutrition_goal=nutrition_goal,
        meal_description=meal_description or "(meal description)",
        meal_ingredient_block=ingredient_block,
        persona_block="",
        persona_style_block="",
    )

    return {
        "system_prompt": system_prompt,
    }


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
