"""Provider registry. A target string ``<provider>:<model>`` picks the implementation."""

from __future__ import annotations

import httpx

from evalproof.config import TargetConfig
from evalproof.providers.anthropic import AnthropicProvider
from evalproof.providers.base import (
    ChatResponse,
    Message,
    Provider,
    ProviderError,
    with_retries,
)
from evalproof.providers.ollama import OllamaProvider
from evalproof.providers.openai import OpenAIProvider

__all__ = [
    "AnthropicProvider",
    "ChatResponse",
    "Message",
    "OllamaProvider",
    "OpenAIProvider",
    "Provider",
    "ProviderError",
    "make_provider",
    "parse_target",
    "with_retries",
]

LLM_PROVIDERS = {"ollama", "anthropic", "openai"}


def parse_target(target: str) -> tuple[str, str]:
    """``"ollama:qwen3:8b"`` → ``("ollama", "qwen3:8b")``. Model names may contain colons."""
    if ":" not in target:
        raise ProviderError(f"target {target!r} must look like '<provider>:<model>'")
    provider, model = target.split(":", 1)
    if not model:
        raise ProviderError(f"target {target!r} has no model")
    return provider, model


def make_provider(cfg: TargetConfig, client: httpx.Client | None = None) -> Provider:
    provider, model = parse_target(cfg.target)
    kwargs = {"base_url": cfg.base_url, "timeout_s": cfg.timeout_s, "client": client}
    if provider == "ollama":
        return OllamaProvider(model, **kwargs)  # type: ignore[arg-type]
    if provider == "anthropic":
        return AnthropicProvider(model, **kwargs)  # type: ignore[arg-type]
    if provider == "openai":
        return OpenAIProvider(model, **kwargs)  # type: ignore[arg-type]
    raise ProviderError(f"unknown provider {provider!r}; expected one of {sorted(LLM_PROVIDERS)}")
