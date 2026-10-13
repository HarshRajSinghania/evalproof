"""Anthropic Messages API over plain HTTP. Optional: needs ANTHROPIC_API_KEY."""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

from evalproof.config import GenerationParams
from evalproof.providers.base import (
    ChatResponse,
    Message,
    ProviderError,
    split_system,
    with_retries,
)

DEFAULT_BASE_URL = "https://api.anthropic.com"
API_VERSION = "2023-06-01"


class AnthropicProvider:
    name = "anthropic"

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
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ProviderError("anthropic: set ANTHROPIC_API_KEY to use this provider")
        self._headers = {
            "x-api-key": key,
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
        }
        self._client = client or httpx.Client(timeout=timeout_s)

    def payload(self, messages: list[Message], params: GenerationParams) -> dict[str, Any]:
        system, rest = split_system(messages)
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "messages": [m.model_dump() for m in rest],
        }
        if system:
            body["system"] = system
        body.update(params.extra)
        return body

    def chat(self, messages: list[Message], params: GenerationParams) -> ChatResponse:
        body = self.payload(messages, params)

        def call() -> httpx.Response:
            resp = self._client.post(
                f"{self.base_url}/v1/messages", json=body, headers=self._headers
            )
            resp.raise_for_status()
            return resp

        start = time.perf_counter()
        resp = with_retries(call)
        latency_ms = (time.perf_counter() - start) * 1000
        data = resp.json()
        blocks = data.get("content") or []
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        usage = data.get("usage") or {}
        return ChatResponse(
            text=text,
            model=data.get("model", self.model),
            tokens_in=int(usage.get("input_tokens") or 0),
            tokens_out=int(usage.get("output_tokens") or 0),
            latency_ms=latency_ms,
        )
