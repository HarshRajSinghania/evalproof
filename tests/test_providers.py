from __future__ import annotations

import json

import httpx
import pytest

from evalproof.config import GenerationParams, TargetConfig
from evalproof.providers import (
    AnthropicProvider,
    Message,
    OllamaProvider,
    OpenAIProvider,
    ProviderError,
    make_provider,
    parse_target,
    with_retries,
)
from tests.conftest import mock_client

MSGS = [Message(role="system", content="be brief"), Message(role="user", content="hi")]


def test_parse_target() -> None:
    assert parse_target("ollama:qwen3:8b") == ("ollama", "qwen3:8b")
    for bad in ("ollama", "ollama:"):
        with pytest.raises(ProviderError):
            parse_target(bad)


def test_ollama_request_and_response() -> None:
    seen: dict[str, object] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen.update(json.loads(req.content))
        assert req.url.path == "/api/chat"
        return httpx.Response(
            200,
            json={
                "model": "qwen3:8b",
                "message": {"role": "assistant", "content": "hello"},
                "prompt_eval_count": 12,
                "eval_count": 3,
            },
        )

    p = OllamaProvider("qwen3:8b", client=mock_client(handler))
    resp = p.chat(MSGS, GenerationParams(seed=7, max_tokens=64, think=False))
    assert resp.text == "hello" and resp.tokens_in == 12 and resp.tokens_out == 3
    assert seen["stream"] is False and seen["think"] is False
    assert seen["options"] == {"temperature": 0.0, "num_predict": 64, "seed": 7}


def test_ollama_bad_shape() -> None:
    p = OllamaProvider("m", client=mock_client(lambda r: httpx.Response(200, json={"x": 1})))
    with pytest.raises(ProviderError):
        p.chat(MSGS, GenerationParams())


def test_anthropic_pulls_system_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    seen: dict[str, object] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen.update(json.loads(req.content))
        assert req.headers["x-api-key"] == "k"
        assert req.headers["anthropic-version"] == "2023-06-01"
        return httpx.Response(
            200,
            json={
                "model": "m",
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 5, "output_tokens": 2},
            },
        )

    resp = AnthropicProvider("m", client=mock_client(handler)).chat(MSGS, GenerationParams())
    assert resp.text == "ok" and resp.tokens_in == 5
    assert seen["system"] == "be brief"
    assert seen["messages"] == [{"role": "user", "content": "hi"}]


def test_openai_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "k")

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        assert body["seed"] == 0 and req.headers["authorization"] == "Bearer k"
        return httpx.Response(
            200,
            json={
                "model": "m",
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 1},
            },
        )

    resp = OpenAIProvider("m", client=mock_client(handler)).chat(MSGS, GenerationParams())
    assert resp.text == "ok" and resp.tokens_out == 1


def test_openai_no_choices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    p = OpenAIProvider("m", client=mock_client(lambda r: httpx.Response(200, json={})))
    with pytest.raises(ProviderError):
        p.chat(MSGS, GenerationParams())


def test_missing_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ProviderError):
        AnthropicProvider("m")
    with pytest.raises(ProviderError):
        OpenAIProvider("m")


def test_retries_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, json={"message": {"content": "ok"}})

    p = OllamaProvider("m", client=mock_client(handler))
    sleeps: list[float] = []
    import evalproof.providers.ollama as mod

    orig = mod.with_retries
    mod.with_retries = lambda fn: orig(fn, sleep=sleeps.append)  # type: ignore[assignment]
    try:
        assert p.chat(MSGS, GenerationParams()).text == "ok"
    finally:
        mod.with_retries = orig
    assert calls["n"] == 3 and len(sleeps) == 2 and sleeps[1] > sleeps[0] * 0.5


def test_non_retryable_status_raises() -> None:
    p = OllamaProvider("m", client=mock_client(lambda r: httpx.Response(400, text="bad model")))
    with pytest.raises(ProviderError, match="400"):
        p.chat(MSGS, GenerationParams())


def test_transport_errors_give_up() -> None:
    def boom() -> int:
        raise httpx.ConnectError("down")

    with pytest.raises(ProviderError, match="giving up"):
        with_retries(boom, attempts=3, sleep=lambda s: None)


def test_make_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    assert make_provider(TargetConfig(target="ollama:m")).name == "ollama"
    assert make_provider(TargetConfig(target="anthropic:m")).name == "anthropic"
    assert make_provider(TargetConfig(target="openai:m")).name == "openai"
    with pytest.raises(ProviderError):
        make_provider(TargetConfig(target="nope:m"))
