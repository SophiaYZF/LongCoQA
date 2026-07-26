from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from longcoqa.paths import DEFAULT_ENV_FILES, resolve_project_path
from longcoqa.utils.io import read_yaml


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            continue
        values[key] = _unquote(value)
    return values


def load_environment(paths: tuple[Path, ...] = DEFAULT_ENV_FILES) -> dict[str, str]:
    merged: dict[str, str] = {}
    for path in reversed(paths):
        merged.update(read_env_file(path))
    merged.update(os.environ)
    return merged


def _parse_key_pool(value: str) -> tuple[str, ...]:
    value = value.strip()
    if not value:
        return ()
    if value.startswith("["):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return tuple(str(item).strip() for item in parsed if str(item).strip())
        except json.JSONDecodeError:
            pass
    candidates = re.split(r"[,;\s]+", value)
    return tuple(dict.fromkeys(candidate for candidate in candidates if candidate))


@dataclass(frozen=True)
class ProviderSettings:
    name: str
    protocol: str
    base_url: str
    api_keys: tuple[str, ...]
    model: str
    concurrency: int
    requests_per_minute: int
    timeout_seconds: float
    max_retries: int

    def safe_summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "protocol": self.protocol,
            "base_url_configured": bool(self.base_url),
            "api_key_count": len(self.api_keys),
            "model": self.model,
            "concurrency": self.concurrency,
            "requests_per_minute": self.requests_per_minute,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
        }


def load_provider_settings(
    provider_name: str,
    config_path: str | Path = "configs/providers.yaml",
    environment: dict[str, str] | None = None,
) -> ProviderSettings:
    config = read_yaml(resolve_project_path(config_path))
    providers = config.get("providers")
    if not isinstance(providers, dict) or provider_name not in providers:
        raise KeyError(f"Unknown provider {provider_name!r}")
    raw = providers[provider_name]
    environment = environment or load_environment()

    base_url = environment.get(str(raw["base_url_env"]), "").strip().rstrip("/")
    key_names = raw.get("api_key_env", [])
    if isinstance(key_names, str):
        key_names = [key_names]
    keys: list[str] = []
    for key_name in key_names:
        keys.extend(_parse_key_pool(environment.get(str(key_name), "")))
    keys = list(dict.fromkeys(keys))

    model = environment.get(str(raw.get("model_env", "")), "").strip()
    model = _unquote(model) if model else str(raw["default_model"])
    return ProviderSettings(
        name=provider_name,
        protocol=str(raw["protocol"]),
        base_url=base_url,
        api_keys=tuple(keys),
        model=model,
        concurrency=int(raw["concurrency"]),
        requests_per_minute=int(raw["requests_per_minute"]),
        timeout_seconds=float(raw["timeout_seconds"]),
        max_retries=int(raw["max_retries"]),
    )
