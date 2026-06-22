"""OpenAI Responses API client used by the agent modules."""

from __future__ import annotations

import os
import re
import time
from typing import Dict, List, Optional

from openai import OpenAI


def _sanitize(text: str) -> str:
    """Remove control characters that can break API input serialization."""
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)


class OpenAIClient:
    """Small wrapper around the OpenAI Responses API."""

    def __init__(
        self,
        model_name: str = "gpt-5.2",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key or api_key.startswith("sk-your-"):
            raise ValueError(
                "OPENAI_API_KEY is not configured. "
                "Add a valid API key to code_interactive/.env or export it "
                "as an environment variable."
            )
        base_url = base_url or os.environ.get("OPENAI_BASE_URL") or None

        self._model_name = model_name
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    @property
    def model_name(self) -> str:
        return self._model_name

    # Shared logic

    def _build_kwargs(
        self,
        messages: List[Dict[str, str]],
        sampling: str = "greedy",
        max_tokens: int = 80,
        reasoning_effort: Optional[str] = None,
        reasoning_summary: Optional[str] = None,
    ) -> dict:
        sanitized = [
            {"role": m.get("role", "user"), "content": _sanitize(m.get("content", ""))}
            for m in messages
        ]

        kwargs: dict = {
            "model": self._model_name,
            "input": sanitized,
            "max_output_tokens": max_tokens,
        }

        if sampling not in ("greedy", "beam"):
            kwargs["temperature"] = 0.7
            kwargs["top_p"] = 0.9

        reasoning: dict = {}
        if reasoning_effort and reasoning_effort != "none":
            reasoning["effort"] = reasoning_effort
        if reasoning_summary:
            reasoning["summary"] = reasoning_summary
        if reasoning:
            kwargs["reasoning"] = reasoning

        return kwargs

    def invoke(
        self,
        messages: List[Dict[str, str]],
        sampling: str = "greedy",
        max_tokens: int = 80,
        reasoning_effort: Optional[str] = None,
        reasoning_summary: Optional[str] = None,
    ) -> str:
        kwargs = self._build_kwargs(
            messages, sampling, max_tokens, reasoning_effort, reasoning_summary,
        )
        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = self._client.responses.create(**kwargs)
                text = response.output_text or ""
                if not text.strip():
                    print(f"[OpenAI] WARNING: empty response "
                          f"(model={self._model_name}, tokens={max_tokens})")
                return text
            except Exception as e:
                err_str = str(e)
                is_retryable = any(k in err_str for k in (
                    "could not parse", "500", "502", "503", "529",
                    "Connection", "Timeout", "rate_limit",
                ))
                if is_retryable and attempt < max_retries - 1:
                    wait = 2 ** attempt
                    print(f"[OpenAI] Retryable error (attempt {attempt + 1}/{max_retries}), "
                          f"retrying in {wait}s: {err_str[:120]}")
                    time.sleep(wait)
                    continue
                print(f"[OpenAI] ERROR in invoke: {e}")
                return f"[API_ERROR: {err_str[:200]}]"

        return "[API_ERROR: max retries exceeded]"


def load_model(
    model_name: str = "gpt-5.2",
    *,
    api_key: str | None = None,
    base_url: str | None = None,
) -> OpenAIClient:
    """Create an OpenAI client for one model name."""
    return OpenAIClient(model_name=model_name, api_key=api_key, base_url=base_url)


def generate_response(
    client: OpenAIClient,
    messages: List[Dict[str, str]],
    sampling: str = "greedy",
    max_new_tokens: int = 80,
    stop_at_newline: bool = True,
    reasoning_effort: Optional[str] = None,
    reasoning_summary: Optional[str] = None,
) -> str:
    raw = client.invoke(
        messages,
        sampling=sampling,
        max_tokens=max_new_tokens,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
    )

    if stop_at_newline:
        lines = [line.strip() for line in raw.split("\n") if line.strip()]
        return lines[0] if lines else ""
    return raw.strip()
