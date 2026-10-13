"""OpenAI-compatible Chat Completions over plain HTTP. Optional: needs OPENAI_API_KEY.

Also works against any server that speaks the same API, via ``base_url``.
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

from evalproof.config import GenerationParams
from evalproof.providers.base import ChatResponse, Message, ProviderError, with_retries

DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OpenAIProvider:
    name = "openai"

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        timeout_s: float = 120.0,
        client: httpx.Client | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ProviderError("openai: set OPENAI_API_KEY to use this provider")
        self._headers = {"authorization": f"Bearer {key}", "content-type": "application/json"}
        self._client = client or httpx.Client(timeout=timeout_s)

    def payload(self, messages: list[Message], params: GenerationParams) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [m.model_dump() for m in messages],
            "temperature": params.temperature,
            "max_tokens": params.max_tokens,
        }
        if params.seed is not None:
            body["seed"] = params.seed
        body.update(params.extra)
        return body

    def chat(self, messages: list[Message], params: GenerationParams) -> ChatResponse:
        body = self.payload(messages, params)

        def call() -> httpx.Response:
            resp = self._client.post(
                f"{self.base_url}/chat/completions", json=body, headers=self._headers
            )
            resp.raise_for_status()
            return resp

        start = time.perf_counter()
        resp = with_retries(call)
        latency_ms = (time.perf_counter() - start) * 1000
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise ProviderError(f"openai: no choices in response: {str(data)[:300]}")
        usage = data.get("usage") or {}
        return ChatResponse(
            text=(choices[0].get("message") or {}).get("content") or "",
            model=data.get("model", self.model),
            tokens_in=int(usage.get("prompt_tokens") or 0),
            tokens_out=int(usage.get("completion_tokens") or 0),
            latency_ms=latency_ms,
        )
