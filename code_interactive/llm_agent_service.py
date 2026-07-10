"""Meal-app-style service facade for agent chatbot replies."""

from __future__ import annotations

from typing import Mapping

try:
    from .agents import (
        AgentConfig,
        CoachingTurnRequest,
        CoachingTurnResult,
        ConversationEngine,
        UserProfileContext,
        build_opening_message,
    )
    from .web_app_config import WebAppConfig
    from .agents.openai_client import generate_response, load_model
except ImportError:  # pragma: no cover - script-style imports from code_interactive/
    from agents import (
        AgentConfig,
        CoachingTurnRequest,
        CoachingTurnResult,
        ConversationEngine,
        UserProfileContext,
        build_opening_message,
    )
    from web_app_config import WebAppConfig
    from agents.openai_client import generate_response, load_model


class LLMAgentService:
    """Meal-app-style service entrypoint for one-turn agent replies."""

    def __init__(
        self,
        config: WebAppConfig | AgentConfig | None = None,
        client_pool: Mapping[str, object] | None = None,
    ) -> None:
        if config is None:
            self.config = WebAppConfig()
        elif isinstance(config, AgentConfig):
            self.config = WebAppConfig(agent=config)
        else:
            self.config = config
        self.client_pool = dict(client_pool or {})
        self.engine = ConversationEngine(
            generate_response=self.run_module_inference,
            config=getattr(self.config, "agent", self.config),
        )

    @property
    def agent_config(self) -> AgentConfig:
        return getattr(self.config, "agent", self.config)

    def _client_for_module(self, module: str):
        model_name = self.config.resolve_model_name(module)
        if model_name not in self.client_pool:
            self.client_pool[model_name] = load_model(model_name)
        return self.client_pool[model_name]

    def run_module_inference(
        self,
        *,
        module: str,
        messages,
        mode: str,
        response_schema: dict | None = None,
    ) -> str:
        options = self.agent_config.generation_options(mode)
        return generate_response(
            self._client_for_module(module),
            [dict(message) for message in messages],
            sampling=options["sampling"],
            max_new_tokens=options["max_new_tokens"],
            stop_at_newline=options["stop_at_newline"],
            reasoning_effort=self.config.resolve_reasoning_effort(module),
            reasoning_summary=self.config.resolve_reasoning_summary(module),
            response_schema=response_schema,
        )

    def generate_chat_replies(
        self,
        request: CoachingTurnRequest,
    ) -> CoachingTurnResult:
        return self.engine.generate_chat_replies(request)

    def build_opening_message(
        self,
        profile: UserProfileContext | None = None,
    ) -> str:
        return build_opening_message(profile)
