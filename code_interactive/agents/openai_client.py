"""OpenAI Responses API client used by the agent modules."""

from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI


def _sanitize(text: str) -> str:
    """Remove control characters that can break API input serialization."""
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)


class OpenAIClient:
    """Small wrapper around the OpenAI Responses API."""

    def __init__(
        self,
        model_name: str = "gpt-5.4",
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
        response_schema: Optional[dict] = None,
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

        if response_schema:
            kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": str(response_schema.get("name") or "structured_output"),
                    "schema": response_schema["schema"],
                    "strict": bool(response_schema.get("strict", True)),
                }
            }

        return kwargs

    @staticmethod
    def _extract_output_text(response: Any) -> str:
        """Extract text from Responses API objects across SDK variants."""
        direct = getattr(response, "output_text", None)
        if isinstance(direct, str) and direct.strip():
            return direct

        chunks: list[str] = []

        def visit(node: Any) -> None:
            if node is None:
                return
            if isinstance(node, str):
                if node.strip():
                    chunks.append(node)
                return
            if isinstance(node, (list, tuple)):
                for item in node:
                    visit(item)
                return
            if isinstance(node, dict):
                text = node.get("text")
                if isinstance(text, str) and text.strip():
                    chunks.append(text)
                for key in ("output", "content", "message"):
                    if key in node:
                        visit(node.get(key))
                return
            text = getattr(node, "text", None)
            if isinstance(text, str) and text.strip():
                chunks.append(text)
            for attr in ("output", "content", "message"):
                if hasattr(node, attr):
                    visit(getattr(node, attr))

        visit(getattr(response, "output", None))
        if not chunks and hasattr(response, "model_dump"):
            try:
                dumped = response.model_dump(exclude_none=True)
            except TypeError:
                dumped = response.model_dump()
            visit(dumped.get("output"))

        return "\n".join(part.strip() for part in chunks if part.strip())

    @staticmethod
    def _empty_response_details(response: Any) -> str:
        """Return compact diagnostics for empty Responses API outputs."""
        status = getattr(response, "status", None)
        incomplete = getattr(response, "incomplete_details", None)
        output = getattr(response, "output", None)
        output_len = len(output) if isinstance(output, (list, tuple)) else "n/a"
        return (
            f"status={status!r}, incomplete_details={incomplete!r}, "
            f"output_len={output_len}"
        )

    def invoke(
        self,
        messages: List[Dict[str, str]],
        sampling: str = "greedy",
        max_tokens: int = 80,
        reasoning_effort: Optional[str] = None,
        reasoning_summary: Optional[str] = None,
        response_schema: Optional[dict] = None,
    ) -> str:
        kwargs = self._build_kwargs(
            messages,
            sampling,
            max_tokens,
            reasoning_effort,
            reasoning_summary,
            response_schema,
        )
        max_retries = 3 if response_schema else 2
        for attempt in range(max_retries):
            try:
                response = self._client.responses.create(**kwargs)
                text = self._extract_output_text(response)
                if not text.strip():
                    details = self._empty_response_details(response)
                    if response_schema and attempt < max_retries - 1:
                        wait = 2 ** attempt
                        print(
                            "[OpenAI] Empty structured response "
                            f"(attempt {attempt + 1}/{max_retries}), "
                            f"retrying in {wait}s: model={self._model_name}, "
                            f"tokens={max_tokens}, {details}"
                        )
                        time.sleep(wait)
                        continue
                    print(
                        "[OpenAI] WARNING: empty response "
                        f"(model={self._model_name}, tokens={max_tokens}, {details})"
                    )
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
    model_name: str = "gpt-5.4",
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
    response_schema: Optional[dict] = None,
) -> str:
    raw = client.invoke(
        messages,
        sampling=sampling,
        max_tokens=max_new_tokens,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
        response_schema=response_schema,
    )

    if stop_at_newline:
        lines = [line.strip() for line in raw.split("\n") if line.strip()]
        return lines[0] if lines else ""
    return raw.strip()
