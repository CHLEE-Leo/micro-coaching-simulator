"""
web_app_config.py
─────────────────
Web app mode settings.

This file owns local server settings and OpenAI runtime routing. Portable agent
behavior lives in ``agents.agent_config.AgentConfig``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

try:
    from .agents.agent_config import AgentConfig, SUPPORTED_GOALS
except ImportError:  # pragma: no cover - script execution via python app.py
    from agents.agent_config import AgentConfig, SUPPORTED_GOALS


@dataclass
class WebAppConfig:
    """
    Local web app settings.

    Server
    ------
    host / port / reload : FastAPI 서버 설정

    LLM Runtime
    -----------
    chatgpt_model / chatgpt_light_model / module_* : OpenAI client routing and
    Responses API runtime options.

    Agent
    -----
    agent : portable agent behavior settings.  ``AgentConfig`` is the single
    source for dialogue control, prompt ablation flags, token limits, and
    tracker thresholds.
    """

    # ── 서버 ──────────────────────────────────────────────────────────────────
    host:   str  = "0.0.0.0"
    port:   int  = 8000
    reload: bool = False   # 개발 시 True

    # ── 이식 가능한 agent 동작 설정 ─────────────────────────────────────────
    # AgentConfig만이 agent behavior의 단일 출처입니다. WebAppConfig는
    # 서버/runtime 설정을 담고, agent 설정은 이 필드로 포함만 합니다.
    agent: AgentConfig = field(default_factory=AgentConfig)

    # ── LLM 프로바이더 — OpenAI Responses API ─────────────────────────────────
    # Heavy 모델: 사용자 대면 텍스트 생성, 정책 결정 등 고품질 추론 필요 모듈
    chatgpt_model: str = "gpt-5.4"
    # Light 모델: 내부 평가/추적/필터링 등 경량 모듈
    chatgpt_light_model: str = "gpt-5.4-mini"

    # ── 모듈별 LLM 모델 지정 ─────────────────────────────────────────────────
    # "heavy" → chatgpt_model, "light" → chatgpt_light_model 로 치환.
    # 구체적 모델명(예: "gpt-5.4")을 직접 지정할 수도 있음.
    module_models: dict = field(default_factory=lambda: {
        "dialogue_planner":      "heavy",
        "meal_assessor":         "heavy",
        "info_seeker":           "heavy",
        "recommender":           "heavy",
        "response_generator":    "heavy",
        "meal_tracker":          "light",
        "context_tracker":       "light",
        "interaction_tracker":   "heavy",
        "alignment_estimator":   "light",
        "certainty_estimator":   "light",
        "guardrail":             "light",
    })

    # ── 모듈별 reasoning_effort ──────────────────────────────────────────────
    # OpenAI Responses API의 reasoning.effort 파라미터.
    # none / low / medium / high — 모델의 내부 추론 깊이 제어.
    # reasoning이 불필요한 단순 생성 모듈은 "none", 판단 근거가 중요한 모듈은 "medium"~"high".
    module_reasoning_effort: dict = field(default_factory=lambda: {
        "dialogue_planner":      "none",
        "meal_assessor":         "none",
        "info_seeker":           "medium",
        "recommender":           "none",
        "response_generator":    "none",
        "meal_tracker":          "none",
        "context_tracker":       "low",
        "interaction_tracker":   "none",
        "alignment_estimator":   "none",
        "certainty_estimator":   "none",
        "guardrail":             "low",
    })

    # ── 모듈별 reasoning summary 활성화 ──────────────────────────────────────
    # True인 모듈은 Responses API에서 reasoning.summary="detailed"를 요청하고,
    # 반환된 summary를 output JSON의 "reasoning" 키에 post-processing으로 삽입.
    # reasoning_effort="none"인 모듈은 summary가 생성되지 않으므로 False 유지.
    module_reasoning_summary: dict = field(default_factory=lambda: {
        "dialogue_planner":      False,
        "meal_assessor":         False,
        "info_seeker":           False,
        "recommender":           False,
        "response_generator":    False,
        "meal_tracker":          False,
        "context_tracker":       False,
        "interaction_tracker":   False,
        "alignment_estimator":   False,
        "certainty_estimator":   False,
        "guardrail":             False,
    })

    def resolve_model_name(self, module: str) -> str:
        """모듈 이름 → 실제 모델명 반환. 'heavy'/'light' 에일리어스를 치환."""
        raw = self.module_models.get(module, "light")
        if raw == "heavy":
            return self.chatgpt_model
        if raw == "light":
            return self.chatgpt_light_model
        return raw  # 직접 지정된 구체적 모델명

    def resolve_reasoning_effort(self, module: str) -> str:
        """모듈 이름 → reasoning_effort 값 반환."""
        return self.module_reasoning_effort.get(module, "none")

    def resolve_reasoning_summary(self, module: str) -> Optional[str]:
        """모듈 이름 → reasoning summary 설정 반환. True면 "detailed", False면 None."""
        if self.module_reasoning_summary.get(module, False):
            return "detailed"
        return None

    def with_agent_overrides(self, **overrides) -> "WebAppConfig":
        """Return a copy with selected ``AgentConfig`` fields replaced."""
        valid = {
            key: value
            for key, value in overrides.items()
            if value is not None and hasattr(self.agent, key)
        }
        if not valid:
            return self
        return replace(self, agent=replace(self.agent, **valid))

    # ── 지원 목표 (읽기 전용) ─────────────────────────────────────────────────
    supported_goals: list = field(default_factory=lambda: list(SUPPORTED_GOALS))
