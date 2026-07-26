from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from longcoqa.config import ProviderSettings
from longcoqa.inference.api_runner import (
    run_api_inference,
    select_dialogue_samples,
)
from longcoqa.inference.free_running import (
    run_free_running_inference,
    validate_free_running_chain,
)
from longcoqa.providers import ProviderClient
from longcoqa.utils.hashing import stable_json_hash
from longcoqa.utils.io import atomic_write_json, read_jsonl

FORMAL_PROTOCOLS = (
    "z1_teacher_forced",
    "z2_free_running",
    "g3_aligned_raw",
)


def preflight_key_pool(
    settings: ProviderSettings,
    *,
    client_factory: Callable[..., Any] = ProviderClient,
) -> tuple[ProviderSettings, dict[str, Any]]:
    working_keys: list[str] = []
    working_indexes: list[int] = []
    for index, api_key in enumerate(settings.api_keys, start=1):
        candidate = replace(
            settings,
            api_keys=(api_key,),
            concurrency=1,
            max_retries=0,
        )
        try:
            with client_factory(candidate) as client:
                response = client.generate(
                    [{"role": "user", "content": "只回复：LONGCOQA_CLAUDE_FORMAL_OK"}],
                    max_tokens=32,
                    temperature=0.0,
                )
            if str(response.text).strip() != "LONGCOQA_CLAUDE_FORMAL_OK":
                continue
        except Exception:
            continue
        working_keys.append(api_key)
        working_indexes.append(index)
    if not working_keys:
        raise ValueError("No configured provider key passed the Claude Opus 4.6 preflight")
    return replace(settings, api_keys=tuple(working_keys)), {
        "configured_keys": len(settings.api_keys),
        "working_key_indexes": working_indexes,
        "working_keys": len(working_keys),
    }


def validate_formal_protocol(
    rows: list[dict[str, Any]],
    *,
    expected_ids: list[object],
    protocol: str,
    model: str,
) -> dict[str, Any]:
    if protocol not in FORMAL_PROTOCOLS:
        raise ValueError(f"Unsupported formal protocol: {protocol}")
    expected = [str(sample_id) for sample_id in expected_ids]
    actual = [str(row.get("sample_id", "")) for row in rows]
    if actual != expected:
        raise ValueError(f"{protocol}: sample order or coverage mismatch")
    if any(row.get("status") != "ok" for row in rows):
        raise ValueError(f"{protocol}: non-success rows")
    if any(row.get("protocol") != protocol for row in rows):
        raise ValueError(f"{protocol}: protocol mismatch")
    if any(row.get("model_requested", row.get("model")) != model for row in rows):
        raise ValueError(f"{protocol}: model mismatch")
    chain = validate_free_running_chain(rows) if protocol == "z2_free_running" else None
    return {
        "status": "ok",
        "protocol": protocol,
        "model": model,
        "successful_samples": len(rows),
        "failed_samples": 0,
        "sample_ids_hash": stable_json_hash(actual),
        "z2_chain": chain,
    }


def run_formal_protocol(
    dataset: Any,
    settings: ProviderSettings,
    *,
    protocol: str,
    alignment_path: Path,
    cache_path: Path,
    output_root: Path,
    seed: int = 42,
    dialogues_per_type: int | None = None,
    temperature: float = 0.0,
    max_tokens_factoid: int = 256,
    max_tokens_non_factoid: int = 768,
) -> dict[str, Any]:
    if protocol not in FORMAL_PROTOCOLS:
        raise ValueError(f"Unsupported formal protocol: {protocol}")
    all_samples = list(dataset.iter_samples())
    selected = (
        select_dialogue_samples(
            all_samples,
            dialogues_per_type=dialogues_per_type,
            seed=seed,
        )
        if dialogues_per_type is not None
        else all_samples
    )
    label = f"dialogues{dialogues_per_type}_per_type" if dialogues_per_type else "full"
    output_dir = output_root / protocol / label
    common = {
        "cache_path": cache_path,
        "output_dir": output_dir,
        "seed": seed,
        "dialogues_per_type": dialogues_per_type,
        "temperature": temperature,
        "max_tokens_factoid": max_tokens_factoid,
        "max_tokens_non_factoid": max_tokens_non_factoid,
    }
    if protocol == "z2_free_running":
        summary = run_free_running_inference(dataset, settings, **common)
    else:
        summary = run_api_inference(
            dataset,
            settings,
            protocol=protocol,
            alignment_path=alignment_path,
            **common,
        )

    rows = read_jsonl(output_dir / "results.jsonl")
    validation = validate_formal_protocol(
        rows,
        expected_ids=[sample["sample_id"] for sample in selected],
        protocol=protocol,
        model=settings.model,
    )
    updated_summary = {**summary, "formal_validation": validation}
    atomic_write_json(output_dir / "summary.json", updated_summary)
    (output_dir / ".complete").touch()
    return updated_summary


def finalize_formal_suite(
    dataset: Any,
    *,
    output_root: Path,
    model: str,
) -> dict[str, Any]:
    expected_ids = [str(row["sample_id"]) for row in dataset.iter_samples()]
    validations: dict[str, Any] = {}
    reports: dict[str, Any] = {}
    for protocol in FORMAL_PROTOCOLS:
        protocol_root = output_root / protocol / "full"
        if not (protocol_root / ".complete").is_file():
            raise ValueError(f"{protocol}: missing completion marker")
        results_path = protocol_root / "results.jsonl"
        summary_path = protocol_root / "summary.json"
        if not results_path.is_file() or not summary_path.is_file():
            raise ValueError(f"{protocol}: missing formal artifacts")
        rows = read_jsonl(results_path)
        validations[protocol] = validate_formal_protocol(
            rows,
            expected_ids=expected_ids,
            protocol=protocol,
            model=model,
        )
        reports[protocol] = json.loads(summary_path.read_text(encoding="utf-8"))

    summary = {
        "schema_version": "longcoqa-claude-opus-46-formal-suite-v1",
        "status": "ok",
        "model": model,
        "protocols": list(FORMAL_PROTOCOLS),
        "samples_per_protocol": len(expected_ids),
        "reports": reports,
        "validation": validations,
    }
    atomic_write_json(output_root / "suite_full.json", summary)
    (output_root / ".complete").touch()
    return summary
