"""
app.py
──────
Micro-Coaching Simulator Interactive 모드 FastAPI 서버.
/ FastAPI server for Micro-Coaching Simulator interactive mode.

LLM 백엔드: ChatGPT (OpenAI API) 전용.
/ LLM backend: ChatGPT (OpenAI API) only.

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

_HERE     = Path(__file__).resolve().parent   # code_interactive/

try:
    from .web_app_config import WebAppConfig
    from .session_manager import SessionManager
except ImportError:  # pragma: no cover - script execution via python app.py
    from web_app_config import WebAppConfig
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

_config:          Optional[WebAppConfig] = None
_session_manager: Optional[SessionManager]   = None


# ─────────────────────────────────────────────────────────────────────────────
# 앱 수명주기 / App lifespan
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 ChatGPT 클라이언트를 생성합니다."""
    global _config, _session_manager

    _config = WebAppConfig()

    logger.info("=" * 60)
    logger.info("[Startup] Micro-Coaching Simulator — Interactive Mode")
    logger.info(f"  LLM          : ChatGPT ({_config.chatgpt_model})")
    logger.info(f"  LLM (light)  : ChatGPT ({_config.chatgpt_light_model})")
    logger.info(f"  max_turns    : {_config.agent.max_turns}")
    logger.info(f"  alignment_min_turn: {_config.agent.alignment_min_turn}")
    logger.info("=" * 60)

    # ── 모듈별 모델 설정 로깅 / Log per-module model assignments
    _unique_models = sorted(set(
        _config.resolve_model_name(m) for m in _config.module_models
    ))
    for mod, alias in sorted(_config.module_models.items()):
        logger.info(f"  module={mod:15s} → {_config.resolve_model_name(mod)} ({alias})")

    # ── ChatGPT 클라이언트 풀: 고유 모델별로 1개씩 생성 ────────────────────
    # Create one ChatGPT client per unique model name
    try:
        from .agents.openai_client import load_model as load_openai_model
    except ImportError:  # pragma: no cover - script execution via python app.py
        from agents.openai_client import load_model as load_openai_model
    _client_pool: dict = {}
    for model_name in _unique_models:
        _client_pool[model_name] = load_openai_model(model_name)
        logger.info(f"[Startup] ChatGPT client ready (model={model_name})")

    _session_manager = SessionManager(
        chatgpt_client_pool=_client_pool,
        config=_config,
    )
    logger.info("[Startup] Server ready.")

    yield  # ── 서버 실행 구간 / server is running ─────────────────────────

    logger.info("[Shutdown] Cleaning up.")


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
    mode:             str  = Field("custom",   description="'custom' | 'deploy'")
    alignment_enabled:    bool = Field(True,        description="Alignment Tracker 활성화 여부")
    nutrition_goal:   str  = Field(...,         description="영양 목표 / Nutrition goal")
    meal_type:        str  = Field("meal",      description="식사 유형 / Meal type")
    meal_description: str  = Field("",          description="음식 이름 목록 / Food item names")
    meal_ingredient:  str  = Field("",          description="재료/조리법 상세 / Ingredient details")
    coach_conversation_mode: Optional[str] = Field(None, description="Coach 대화 모드: open-ended | template-based")
    uncertainty_tracking: Optional[bool] = Field(None, description="Uncertainty Tracking 활성화 여부")
    context_tracking: Optional[bool] = Field(None, description="Context Tracking (LLM 대화 요약) 활성화 여부")
    alignment_use_goal_def:  Optional[bool] = Field(None, description="Alignment Tracker에 goal_definition 포함 여부")
    alignment_use_workflow:  Optional[bool] = Field(None, description="Alignment Tracker에 expert workflow 포함 여부")
    alignment_output_format: Optional[str]  = Field(None, description="Alignment Tracker 출력 포맷: binary | 0-1 | 0-100")
    persona_activity_level:   Optional[str]  = Field(None, description="활동 수준 / Activity level")
    persona_diet_preferences: Optional[list] = Field(None, description="식이 선호도 / Diet preferences")
    persona_allergies:        Optional[list] = Field(None, description="사용자 알레르기 / User allergies")
    persona_health_concerns:  Optional[list] = Field(None, description="건강 관심사 / Health concerns")


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
    return {
        "ready":             _session_manager is not None,
        "chatgpt_available": _session_manager is not None and _session_manager._chatgpt_client is not None,
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

    if req.mode not in ("custom", "deploy"):
        raise HTTPException(status_code=400, detail="mode must be 'custom' or 'deploy'.")

    try:
        session = await asyncio.to_thread(
            _session_manager.create_session,
            nutrition_goal=req.nutrition_goal,
            meal_description=req.meal_description.strip(),
            meal_ingredient=req.meal_ingredient.strip(),
            meal_type=req.meal_type.strip() or "meal",
            mode=req.mode,
            alignment_enabled=req.alignment_enabled,
            coach_conversation_mode=req.coach_conversation_mode,
            uncertainty_tracking=req.uncertainty_tracking,
            context_tracking=req.context_tracking,
            alignment_use_goal_def=req.alignment_use_goal_def,
            alignment_use_workflow=req.alignment_use_workflow,
            alignment_output_format=req.alignment_output_format,
            persona_activity_level=req.persona_activity_level,
            persona_diet_preferences=req.persona_diet_preferences,
            persona_allergies=req.persona_allergies,
            persona_health_concerns=req.persona_health_concerns,
        )
    except Exception as e:
        logger.exception("Failed to create session")
        raise HTTPException(status_code=500, detail=str(e))

    first_q = session.turns[0].coach_utterance if session.turns else ""
    logger.info(
        f"[Session {session.session_id[:8]}] Created — "
        f"mode={req.mode} goal={req.nutrition_goal} alignment={req.alignment_enabled}"
    )

    _model_label = _config.chatgpt_model

    return {
        "session_id":     session.session_id,
        "first_question": first_q,
        "mode":           session.mode,
        "alignment_enabled":  session.alignment_enabled,
        "nutrition_goal":  session.nutrition_goal,
        "meal_type":      session.meal_type,
        "coach_label":     _model_label,
        "user_label":      _model_label
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


@app.post("/api/session/{session_id}/turn/stream")
async def submit_turn_stream(session_id: str, req: TurnRequest):
    """
    SSE 스트리밍 엔드포인트 — 파이프라인 실행 후 메타데이터 + Coach 텍스트를 청크 단위로 전송.
    / SSE streaming endpoint: sends metadata event, then coach text token-by-token.

    Events
    ------
    event: meta   data: { ... monitoring JSON (same as /turn minus coach_question) }
    event: token  data: <text chunk>
    event: done   data: [DONE]
    """
    import json as _json
    from fastapi.responses import StreamingResponse

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
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.exception(f"Error processing streamed turn for session {session_id}")
        raise HTTPException(status_code=500, detail=str(e))

    # Guardrail 차단 시 coach_question을 meta에 유지 (스트리밍하지 않음)
    is_blocked = result.get("guardrail_blocked", False)
    coach_text = "" if is_blocked else (result.pop("coach_question", "") or "")
    assessment_text = None if is_blocked else result.pop("assessment_message", None)
    coach_messages = [] if is_blocked else result.pop("coach_messages", [])

    async def _sse_generator():
        # 1) 메타데이터 이벤트 (모니터링 데이터 + 상태)
        yield f"event: meta\ndata: {_json.dumps(result, ensure_ascii=False)}\n\n"
        # 2) Multi-bubble: coach_messages 를 순차적으로 스트리밍
        if len(coach_messages) > 1:
            # 첫 번째 메시지 = assessment 피드백 (즉시 전송)
            yield f"event: assessment\ndata: {_json.dumps(coach_messages[0], ensure_ascii=False)}\n\n"
            # 나머지 메시지 = 후속 발화 (단어 단위 스트리밍)
            for msg in coach_messages[1:]:
                yield f"event: bubble_start\ndata: \"\"\n\n"
                words = msg.split(" ")
                for i, word in enumerate(words):
                    chunk = word if i == 0 else " " + word
                    yield f"event: token\ndata: {_json.dumps(chunk, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0.03)
        elif coach_text:
            # 단일 말풍선: 기존 방식 유지 (하위 호환)
            words = coach_text.split(" ")
            for i, word in enumerate(words):
                chunk = word if i == 0 else " " + word
                yield f"event: token\ndata: {_json.dumps(chunk, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.03)
        # 3) 종료 신호
        yield "event: done\ndata: [DONE]\n\n"

    logger.info(
        f"[Session {session_id[:8]}] Turn {result['turn_idx']} (stream) — "
        f"status={result['status']} aligned={result['alignment_aligned']}"
    )
    return StreamingResponse(
        _sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/session/{session_id}/continue")
async def continue_session(session_id: str, req: ContinueSessionRequest):
    """
    이전 세션의 사용자 프로필(ContextTracker)을 이어받아 새 식사 세션을 시작합니다.
    / Starts a new meal session carrying over the ContextTracker from a previous session.

    Response
    --------
    {
      "session_id":    str,    // new session ID
      "first_question": str,
      "previous_session_id": str,
      "previous_meals": int   // number of past meals in ContextTracker
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
    past_meals = len(new_session.context_tracker._profile["past_meals"]) if new_session.context_tracker else 0

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

try:
    from .agents.modules.alignment_estimator import (
        _get_goal_spec,
        _get_workflow_text,
        _load_output_format,
    )
    from .agents.prompts.roles.alignment_estimator import build_alignment_system_prompt
    from .agents.prompts.roles.information_seeker import (
        INFORMATION_SEEKER_ACTION_GUIDELINES,
        INFORMATION_SEEKER_STRATEGY_BLOCK,
        INFORMATION_SEEKER_SYSTEM_PROMPT,
    )
except ImportError:  # pragma: no cover - script execution via python app.py
    from agents.modules.alignment_estimator import (
        _get_goal_spec,
        _get_workflow_text,
        _load_output_format,
    )
    from agents.prompts.roles.alignment_estimator import build_alignment_system_prompt
    from agents.prompts.roles.information_seeker import (
        INFORMATION_SEEKER_ACTION_GUIDELINES,
        INFORMATION_SEEKER_STRATEGY_BLOCK,
        INFORMATION_SEEKER_SYSTEM_PROMPT,
    )


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

    system_prompt = build_alignment_system_prompt(
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
    system_prompt = INFORMATION_SEEKER_SYSTEM_PROMPT.format(
        nutrition_goal=nutrition_goal,
        meal_type=meal_type,
    )
    action_guidelines_text = ""
    if conversation_mode == "template-based":
        action_guidelines_text = INFORMATION_SEEKER_ACTION_GUIDELINES
        system_prompt += INFORMATION_SEEKER_STRATEGY_BLOCK.format(
            action_guidelines=INFORMATION_SEEKER_ACTION_GUIDELINES
        )

    return {
        "system_prompt": system_prompt,
        "action_guidelines_text": action_guidelines_text,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 직접 실행 / Direct execution
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = WebAppConfig()
    uvicorn.run(
        "app:app",
        host=cfg.host,
        port=cfg.port,
        reload=cfg.reload,
    )
