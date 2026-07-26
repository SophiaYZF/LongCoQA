from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

from longcoqa.config import ProviderSettings
from longcoqa.data import LongCoQADataset
from longcoqa.evaluation.text_metrics import score_answer
from longcoqa.inference.prompts import build_messages, index_samples
from longcoqa.providers import ProviderClient
from longcoqa.providers.cache import ResponseCache
from longcoqa.utils.hashing import stable_json_hash
from longcoqa.utils.io import (
    atomic_write_json,
    atomic_write_jsonl,
    read_jsonl,
)


def validate_api_protocol_results(
    results: list[dict[str, Any]],
    *,
    protocol: str,
    expected_samples: int,
) -> dict[str, Any]:
    if len(results) != expected_samples:
        raise ValueError(
            f"{protocol}: expected {expected_samples} rows, found {len(results)}"
        )
    sample_ids = [str(item.get("sample_id", "")) for item in results]
    unique_sample_ids = len(set(sample_ids))
    if unique_sample_ids != len(sample_ids):
        raise ValueError(
            f"{protocol}: duplicate sample IDs: rows={len(sample_ids)} "
            f"unique={unique_sample_ids}"
        )
    failed_samples = sum(item.get("status") != "ok" for item in results)
    if failed_samples:
        raise ValueError(f"{protocol}: {failed_samples} non-success rows")
    mismatched = sum(item.get("protocol") != protocol for item in results)
    if mismatched:
        raise ValueError(f"{protocol}: {mismatched} protocol mismatch rows")
    return {
        "protocol": protocol,
        "expected_samples": expected_samples,
        "result_rows": len(results),
        "unique_sample_ids": unique_sample_ids,
        "successful_samples": len(results),
        "failed_samples": 0,
        "status": "ok",
    }


def _select_smoke_samples(
    samples: list[dict[str, Any]],
    *,
    limit: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    if limit is None or limit >= len(samples):
        return samples
    random_state = random.Random(seed)
    by_type = {
        question_type: [item for item in samples if item["question_type"] == question_type]
        for question_type in ("factoid", "non_factoid")
    }
    factoid_count = limit // 2
    selected = random_state.sample(by_type["factoid"], factoid_count)
    selected.extend(random_state.sample(by_type["non_factoid"], limit - factoid_count))
    return sorted(
        selected,
        key=lambda item: (
            str(item["question_type"]),
            int(item["dialogue_id"]),
            int(item["turn"]),
        ),
    )


def select_dialogue_samples(
    samples: list[dict[str, Any]],
    *,
    dialogues_per_type: int,
    seed: int,
) -> list[dict[str, Any]]:
    if dialogues_per_type < 1:
        raise ValueError("dialogues_per_type must be positive")
    grouped: dict[str, dict[int, list[dict[str, Any]]]] = {
        "factoid": {},
        "non_factoid": {},
    }
    for sample in samples:
        question_type = str(sample["question_type"])
        if question_type not in grouped:
            raise ValueError(f"Unknown question type: {question_type}")
        dialogue_id = int(sample["dialogue_id"])
        grouped[question_type].setdefault(dialogue_id, []).append(sample)

    selected: list[dict[str, Any]] = []
    for question_type in ("factoid", "non_factoid"):
        available = sorted(grouped[question_type])
        if len(available) < dialogues_per_type:
            raise ValueError(
                f"{question_type}: requested {dialogues_per_type} dialogues, "
                f"but only {len(available)} are available"
            )
        chosen = set(
            random.Random(f"{seed}:{question_type}").sample(
                available,
                dialogues_per_type,
            )
        )
        for dialogue_id in sorted(chosen):
            selected.extend(
                sorted(
                    grouped[question_type][dialogue_id],
                    key=lambda item: int(item["turn"]),
                )
            )
    return selected


def _load_evidence_maps(
    dataset: LongCoQADataset,
    alignment_path: Path,
) -> tuple[dict[int, str], dict[int, str]]:
    released = {int(row.id): str(row.value) for row in dataset.evidence.itertuples(index=False)}
    aligned = {int(item["evidence_id"]): str(item["source_text"]) for item in read_jsonl(alignment_path)}
    return released, aligned


def run_api_inference(
    dataset: LongCoQADataset,
    settings: ProviderSettings,
    *,
    protocol: str,
    alignment_path: Path,
    cache_path: Path,
    output_dir: Path,
    seed: int = 42,
    limit: int | None = None,
    dialogues_per_type: int | None = None,
    temperature: float = 0.0,
    max_tokens_factoid: int = 256,
    max_tokens_non_factoid: int = 768,
) -> dict[str, Any]:
    if limit is not None and dialogues_per_type is not None:
        raise ValueError("limit and dialogues_per_type are mutually exclusive")
    all_samples = list(dataset.iter_samples())
    selected = (
        select_dialogue_samples(
            all_samples,
            dialogues_per_type=dialogues_per_type,
            seed=seed,
        )
        if dialogues_per_type is not None
        else _select_smoke_samples(all_samples, limit=limit, seed=seed)
    )
    dialogues = index_samples(all_samples)
    released, aligned = _load_evidence_maps(dataset, alignment_path)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def checkpoint_complete(sample: dict[str, Any]) -> bool:
        path = checkpoint_dir / f"{sample['sample_id'].replace(':', '__')}.json"
        if not path.is_file():
            return False
        try:
            return json.loads(path.read_text(encoding="utf-8")).get("status") == "ok"
        except (json.JSONDecodeError, OSError):
            return False

    pending = [sample for sample in selected if not checkpoint_complete(sample)]
    errors: list[dict[str, str]] = []

    with ResponseCache(cache_path) as cache, ProviderClient(settings, cache=cache) as client:

        def generate(sample: dict[str, Any]) -> dict[str, Any]:
            messages = build_messages(
                sample,
                protocol=protocol,
                dialogues=dialogues,
                released_evidence=released,
                aligned_source=aligned,
            )
            max_tokens = (
                max_tokens_factoid if sample["question_type"] == "factoid" else max_tokens_non_factoid
            )
            response = client.generate(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            metrics = score_answer(response.text, str(sample["answer"]))
            return {
                "status": "ok",
                "sample_id": sample["sample_id"],
                "question_type": sample["question_type"],
                "dialogue_id": sample["dialogue_id"],
                "turn": sample["turn"],
                "protocol": protocol,
                "provider": settings.name,
                "model": response.model,
                "question": sample["question"],
                "reference": sample["answer"],
                "evidence_ids": sample["evidence_ids"],
                "messages": messages,
                "prediction": response.text,
                "metrics": metrics,
                "response": asdict(response),
            }

        with ThreadPoolExecutor(max_workers=settings.concurrency) as executor:
            futures = {executor.submit(generate, sample): sample for sample in pending}
            for future in as_completed(futures):
                sample = futures[future]
                checkpoint_path = checkpoint_dir / (f"{sample['sample_id'].replace(':', '__')}.json")
                try:
                    result = future.result()
                except Exception as error:
                    result = {
                        "status": "error",
                        "sample_id": sample["sample_id"],
                        "protocol": protocol,
                        "provider": settings.name,
                        "model": settings.model,
                        "error_type": type(error).__name__,
                        "error": str(error)[:500],
                    }
                    errors.append(
                        {
                            "sample_id": str(sample["sample_id"]),
                            "error_type": type(error).__name__,
                        }
                    )
                atomic_write_json(checkpoint_path, result)

    results = [
        json.loads(
            (checkpoint_dir / f"{sample['sample_id'].replace(':', '__')}.json").read_text(encoding="utf-8")
        )
        for sample in selected
    ]
    atomic_write_jsonl(output_dir / "results.jsonl", results)
    successful = [item for item in results if item["status"] == "ok"]
    metric_names = ("exact_match", "character_f1", "rouge_l_f1")
    usage = {
        key: sum(int(item["response"]["usage"].get(key, 0)) for item in successful)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    summary = {
        "schema_version": "longcoqa-api-inference-v1",
        "provider": settings.safe_summary(),
        "protocol": protocol,
        "seed": seed,
        "requested_samples": len(selected),
        "successful_samples": len(successful),
        "failed_samples": len(results) - len(successful),
        "errors": errors,
        "metrics": {
            metric: (
                sum(float(item["metrics"][metric]) for item in successful) / len(successful)
                if successful
                else None
            )
            for metric in metric_names
        },
        "usage": usage,
        "sample_ids_hash": stable_json_hash([item["sample_id"] for item in selected]),
        "results_hash": stable_json_hash(
            [
                {
                    "sample_id": item["sample_id"],
                    "status": item["status"],
                    "prediction": item.get("prediction"),
                }
                for item in results
            ]
        ),
    }
    atomic_write_json(output_dir / "summary.json", summary)
    return summary


def run_api_suite(
    dataset: LongCoQADataset,
    settings: ProviderSettings,
    *,
    protocols: tuple[str, ...],
    alignment_path: Path,
    cache_path: Path,
    output_root: Path,
    seed: int = 42,
    limit: int | None = None,
    dialogues_per_type: int | None = None,
    temperature: float = 0.0,
    max_tokens_factoid: int = 256,
    max_tokens_non_factoid: int = 768,
) -> dict[str, Any]:
    if limit is not None and dialogues_per_type is not None:
        raise ValueError("limit and dialogues_per_type are mutually exclusive")
    if dialogues_per_type is not None:
        selected = select_dialogue_samples(
            list(dataset.iter_samples()),
            dialogues_per_type=dialogues_per_type,
            seed=seed,
        )
        expected_samples = len(selected)
        sample_label = f"dialogues{dialogues_per_type}_per_type"
    elif limit is not None:
        expected_samples = limit
        sample_label = f"smoke{limit}"
    else:
        expected_samples = len(list(dataset.iter_samples()))
        sample_label = "full"
    reports: dict[str, Any] = {}
    validations: dict[str, Any] = {}
    for protocol in protocols:
        output_dir = output_root / protocol / sample_label
        reports[protocol] = run_api_inference(
            dataset,
            settings,
            protocol=protocol,
            alignment_path=alignment_path,
            cache_path=cache_path,
            output_dir=output_dir,
            seed=seed,
            limit=limit,
            dialogues_per_type=dialogues_per_type,
            temperature=temperature,
            max_tokens_factoid=max_tokens_factoid,
            max_tokens_non_factoid=max_tokens_non_factoid,
        )
        validations[protocol] = validate_api_protocol_results(
            read_jsonl(output_dir / "results.jsonl"),
            protocol=protocol,
            expected_samples=expected_samples,
        )
    summary = {
        "schema_version": "longcoqa-api-suite-v1",
        "status": "ok",
        "provider": settings.safe_summary(),
        "protocols": list(protocols),
        "seed": seed,
        "samples_per_protocol": expected_samples,
        "dialogues_per_type": dialogues_per_type,
        "reports": reports,
        "validation": validations,
    }
    atomic_write_json(output_root / f"suite_{sample_label}.json", summary)
    return summary
