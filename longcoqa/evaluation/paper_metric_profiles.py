from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any

from longcoqa.utils.hashing import stable_json_hash
from longcoqa.utils.io import read_yaml


class MetricCompatibility(StrEnum):
    EXACT = "exact"
    PAPER_NAMED_LOCKED = "paper_named_locked"
    APPROXIMATE = "approximate"
    MODERN_EXTENSION = "modern_extension"


@dataclass(frozen=True)
class MetricProfile:
    profile_id: str
    metrics: tuple[str, ...]
    compatibility: MetricCompatibility
    implementation: str
    normalization: str
    tokenization: str
    options: Mapping[str, Any] = field(default_factory=dict)
    model_id: str | None = None
    model_revision: str | None = None

    def __post_init__(self) -> None:
        for name in ("profile_id", "implementation", "normalization", "tokenization"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty")
        if not self.metrics or any(not str(metric).strip() for metric in self.metrics):
            raise ValueError("metrics must contain non-empty metric names")
        if len(set(self.metrics)) != len(self.metrics):
            raise ValueError("metrics must not contain duplicates")
        if self.model_id and not str(self.model_revision or "").strip():
            raise ValueError("A learned metric model requires a locked model_revision")
        if self.model_revision and not str(self.model_id or "").strip():
            raise ValueError("model_revision requires model_id")
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "metrics": list(self.metrics),
            "compatibility": self.compatibility.value,
            "implementation": self.implementation,
            "normalization": self.normalization,
            "tokenization": self.tokenization,
            "options": dict(self.options),
            "model_id": self.model_id,
            "model_revision": self.model_revision,
        }

    @property
    def profile_hash(self) -> str:
        return stable_json_hash(self.to_dict())


def metric_profiles_from_mapping(raw: Mapping[str, Any]) -> dict[str, MetricProfile]:
    values = raw.get("profiles")
    if not isinstance(values, list) or not values:
        raise ValueError("Metric config must contain a non-empty profiles list")
    profiles: dict[str, MetricProfile] = {}
    for index, value in enumerate(values):
        if not isinstance(value, Mapping):
            raise ValueError(f"profiles[{index}] must be a mapping")
        profile = MetricProfile(
            profile_id=str(value.get("profile_id", "")),
            metrics=tuple(str(metric) for metric in value.get("metrics", [])),
            compatibility=MetricCompatibility(str(value.get("compatibility", ""))),
            implementation=str(value.get("implementation", "")),
            normalization=str(value.get("normalization", "")),
            tokenization=str(value.get("tokenization", "")),
            options=dict(value.get("options", {})),
            model_id=(str(value["model_id"]) if value.get("model_id") else None),
            model_revision=(
                str(value["model_revision"]) if value.get("model_revision") else None
            ),
        )
        if profile.profile_id in profiles:
            raise ValueError(f"Duplicate metric profile ID: {profile.profile_id}")
        profiles[profile.profile_id] = profile
    return profiles


def load_metric_profiles(path: Path) -> dict[str, MetricProfile]:
    return metric_profiles_from_mapping(read_yaml(path))
