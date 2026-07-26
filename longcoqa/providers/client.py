from __future__ import annotations

import random
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlparse

import httpx

from longcoqa.config import ProviderSettings
from longcoqa.providers.cache import ResponseCache

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def _endpoint(base_url: str, path: str) -> str:
    if not base_url:
        raise ValueError("Provider base URL is not configured")
    base = base_url.rstrip("/")
    path = "/" + path.lstrip("/")
    if urlparse(base).path.rstrip("/").endswith("/v1") and path.startswith("/v1/"):
        path = path[3:]
    return base + path


class RequestRateLimiter:
    def __init__(self, requests_per_minute: int) -> None:
        self.interval = 60.0 / max(1, requests_per_minute)
        self._next_allowed = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next_allowed - now)
            self._next_allowed = max(now, self._next_allowed) + self.interval
        if delay:
            time.sleep(delay)


@dataclass(frozen=True)
class GenerationResult:
    provider: str
    model: str
    text: str
    usage: dict[str, Any]
    finish_reason: str | None
    request_hash: str
    cached: bool
    latency_seconds: float


class ProviderClient:
    def __init__(
        self,
        settings: ProviderSettings,
        *,
        cache: ResponseCache | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not settings.api_keys:
            raise ValueError(f"No API key configured for provider {settings.name}")
        self.settings = settings
        self.cache = cache
        self.rate_limiter = RequestRateLimiter(settings.requests_per_minute)
        self._client = httpx.Client(
            timeout=settings.timeout_seconds,
            transport=transport,
            follow_redirects=True,
        )
        self._key_cursor = 0
        self._key_lock = threading.Lock()

    def _headers(self, api_key: str) -> dict[str, str]:
        if self.settings.protocol == "openai":
            return {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        if self.settings.protocol == "anthropic":
            return {
                "Authorization": f"Bearer {api_key}",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            }
        raise ValueError(f"Unsupported protocol {self.settings.protocol!r}")

    def _next_key(self) -> str:
        with self._key_lock:
            key = self.settings.api_keys[self._key_cursor % len(self.settings.api_keys)]
            self._key_cursor += 1
        return key

    def _request_with_retry(
        self,
        method: str,
        url: str,
        *,
        json_body: dict[str, Any] | None = None,
    ) -> httpx.Response:
        last_error: BaseException | None = None
        for attempt in range(self.settings.max_retries + 1):
            self.rate_limiter.wait()
            try:
                response = self._client.request(
                    method,
                    url,
                    headers=self._headers(self._next_key()),
                    json=json_body,
                )
                if response.status_code not in RETRYABLE_STATUS:
                    response.raise_for_status()
                    return response
                last_error = httpx.HTTPStatusError(
                    f"retryable status {response.status_code}",
                    request=response.request,
                    response=response,
                )
                retry_after = response.headers.get("retry-after")
                if retry_after:
                    try:
                        delay = min(float(retry_after), 60.0)
                    except ValueError:
                        delay = 0.0
                else:
                    delay = 0.0
            except (httpx.TimeoutException, httpx.NetworkError) as error:
                last_error = error
                delay = 0.0

            if attempt >= self.settings.max_retries:
                break
            exponential = min(30.0, 0.5 * (2**attempt))
            time.sleep(max(delay, exponential + random.random() * 0.25))

        assert last_error is not None
        raise last_error

    def list_models(self) -> list[dict[str, Any]]:
        path = "/v1/models" if self.settings.protocol == "anthropic" else "/models"
        response = self._request_with_retry("GET", _endpoint(self.settings.base_url, path))
        payload = response.json()
        models = payload.get("data", payload.get("models", []))
        safe_models = []
        for model in models if isinstance(models, list) else []:
            if not isinstance(model, dict):
                continue
            safe_models.append(
                {
                    key: model[key]
                    for key in ("id", "display_name", "created_at", "max_input_tokens", "max_tokens")
                    if key in model
                }
            )
        return safe_models

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        max_tokens: int = 128,
        temperature: float = 0.0,
        extra_body: dict[str, Any] | None = None,
    ) -> GenerationResult:
        model = model or self.settings.model
        canonical_request = {
            "provider": self.settings.name,
            "protocol": self.settings.protocol,
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "extra_body": extra_body or {},
        }
        request_hash = ResponseCache.request_hash(canonical_request)
        if self.cache and (cached_payload := self.cache.get(request_hash)):
            cached_payload["cached"] = True
            cached_payload["latency_seconds"] = 0.0
            return GenerationResult(**cached_payload)

        if self.settings.protocol == "openai":
            body = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                **(extra_body or {}),
            }
            url = _endpoint(self.settings.base_url, "/chat/completions")
        else:
            system_messages = [item["content"] for item in messages if item["role"] == "system"]
            body = {
                "model": model,
                "messages": [item for item in messages if item["role"] != "system"],
                "max_tokens": max_tokens,
                "temperature": temperature,
                **(extra_body or {}),
            }
            if system_messages:
                body["system"] = "\n\n".join(system_messages)
            url = _endpoint(self.settings.base_url, "/v1/messages")

        started = time.monotonic()
        response = self._request_with_retry("POST", url, json_body=body)
        latency = time.monotonic() - started
        payload = response.json()
        if self.settings.protocol == "openai":
            choice = payload["choices"][0]
            text = choice["message"].get("content") or ""
            finish_reason = choice.get("finish_reason")
        else:
            blocks = payload.get("content", [])
            text = "".join(str(block.get("text", "")) for block in blocks if block.get("type") == "text")
            finish_reason = payload.get("stop_reason")
        result = GenerationResult(
            provider=self.settings.name,
            model=model,
            text=text,
            usage=payload.get("usage", {}),
            finish_reason=finish_reason,
            request_hash=request_hash,
            cached=False,
            latency_seconds=latency,
        )
        if self.cache:
            self.cache.put(request_hash, self.settings.name, model, asdict(result))
        return result

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> ProviderClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
