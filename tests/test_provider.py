from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from longcoqa.config import ProviderSettings
from longcoqa.providers import ProviderClient
from longcoqa.providers.cache import ResponseCache


def _settings(protocol: str = "openai") -> ProviderSettings:
    return ProviderSettings(
        name="test",
        protocol=protocol,
        base_url="https://provider.example/v1",
        api_keys=("secret-key",),
        model="test-model",
        concurrency=1,
        requests_per_minute=1_000_000,
        timeout_seconds=5.0,
        max_retries=0,
    )


def test_openai_generation_and_cache(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer secret-key"
        body = json.loads(request.content)
        assert body["model"] == "test-model"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            },
        )

    with ResponseCache(tmp_path / "cache.sqlite3") as cache:
        with ProviderClient(
            _settings(),
            cache=cache,
            transport=httpx.MockTransport(handler),
        ) as client:
            first = client.generate([{"role": "user", "content": "ping"}])
            second = client.generate([{"role": "user", "content": "ping"}])

    assert first.text == "ok"
    assert first.cached is False
    assert second.text == "ok"
    assert second.cached is True
    assert len(requests) == 1


def test_anthropic_message_shape() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/messages"
        body = json.loads(request.content)
        assert body["system"] == "be concise"
        assert body["messages"] == [{"role": "user", "content": "ping"}]
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "pong"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 2, "output_tokens": 1},
            },
        )

    with ProviderClient(
        _settings("anthropic"),
        transport=httpx.MockTransport(handler),
    ) as client:
        result = client.generate(
            [
                {"role": "system", "content": "be concise"},
                {"role": "user", "content": "ping"},
            ]
        )

    assert result.text == "pong"


def test_non_retryable_http_error_is_raised_once() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, json={"error": "unauthorized"})

    settings = _settings()
    settings = ProviderSettings(**{**settings.__dict__, "max_retries": 3})
    with ProviderClient(
        settings,
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(httpx.HTTPStatusError):
            client.list_models()

    assert calls == 1
