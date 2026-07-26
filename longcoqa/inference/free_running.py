from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from typing import Any

from longcoqa.config import ProviderSettings
from longcoqa.evaluation.text_metrics import score_answer
from longcoqa.inference.api_runner import select_dialogue_samples
from longcoqa.inference.prompts import SYSTEM_PROMPT
from longcoqa.providers import ProviderClient
from longcoqa.providers.cache import ResponseCache
from longcoqa.utils.hashing import stable_json_hash
from longcoqa.utils.io import atomic_write_json, atomic_write_jsonl

PROMPT_CONTRACT = "longcoqa-z2-free-running-v1"
Z2_PROTOCOL = "z2_free_running"

PriorPrediction = tuple[str, str]
DialogueKey = tuple[str, int]


def group_dialogue_samples(
    samples: Iterable[dict[str, Any]],
) -> dict[DialogueKey, list[dict[str, Any]]]:
    grouped: dict[DialogueKey, list[dict[str, Any]]] = defaultdict(list)
    for row in samples:
        key = (str(row["question_type"]), int(row["dialogue_id"]))
        grouped[key].append(row)
    return {
        key: sorted(rows, key=lambda row: int(row["turn"]))
        for key, rows in sorted(grouped.items())
    }


def history_hash(prior_predictions: Sequence[PriorPrediction]) -> str:
    return stable_json_hash(
        [
            {"question": str(question), "prediction": str(prediction)}
            for question, prediction in prior_predictions
        ]
    )


def build_free_running_messages(
    sample: dict[str, Any],
    prior_predictions: Sequence[PriorPrediction],
) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for question, prediction in prior_predictions:
        messages.extend(
            [
                {"role": "user", "content": str(question)},
                {"role": "assistant", "content": str(prediction)},
            ]
        )
    messages.append({"role": "user", "content": str(sample["question"])})
    return messages


def validate_free_running_chain(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    grouped = group_dialogue_samples(rows)
    validated_rows = 0
    for dialogue_rows in grouped.values():
        prior: list[PriorPrediction] = []
        previous_turn: int | None = None
        for row in dialogue_rows:
            if row.get("status") != "ok":
                raise ValueError(f"{row.get('sample_id')}: non-success Z2 row")
            if row.get("protocol") != Z2_PROTOCOL:
                raise ValueError(f"{row.get('sample_id')}: protocol mismatch")
            turn = int(row["turn"])
            if previous_turn is not None and turn <= previous_turn:
                raise ValueError(f"{row.get('sample_id')}: non-increasing turn order")
            expected_hash = history_hash(prior)
            if row.get("history_hash") != expected_hash:
                raise ValueError(f"{row.get('sample_id')}: history hash mismatch")
            expected_messages = build_free_running_messages(row, prior)
            if row.get("messages") != expected_messages:
                raise ValueError(f"{row.get('sample_id')}: assistant history mismatch")
            prior.append((str(row["question"]), str(row["prediction"])))
            previous_turn = turn
            validated_rows += 1
    return {
        "dialogues": len(grouped),
        "rows": validated_rows,
        "gold_history_messages": 0,
    }


def _checkpoint_path(output_dir: Any, sample_id: str) -> Any:
    return output_dir / "checkpoints" / f"{sample_id.replace(':', '__')}.json"


def _compatible_checkpoint(
    path: Any,
    *,
    sample: dict[str, Any],
    model: str,
    expected_history_hash: str,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        row.get("status") != "ok"
        or row.get("sample_id") != sample["sample_id"]
        or row.get("protocol") != Z2_PROTOCOL
        or row.get("model_requested") != model
        or row.get("prompt_contract") != PROMPT_CONTRACT
        or row.get("history_hash") != expected_history_hash
    ):
        return None
    return row


def run_free_running_inference(
    dataset: Any,
    settings: ProviderSettings,
    *,
    cache_path: Any,
    output_dir: Any,
    seed: int = 42,
    dialogues_per_type: int | None = None,
    temperature: float = 0.0,
    max_tokens_factoid: int = 256,
    max_tokens_non_factoid: int = 768,
    client_factory: Any = ProviderClient,
) -> dict[str, Any]:
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
    grouped = group_dialogue_samples(selected)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    def run_dialogue(
        dialogue_rows: list[dict[str, Any]],
        client: Any,
    ) -> list[dict[str, Any]]:
        prior: list[PriorPrediction] = []
        produced: list[dict[str, Any]] = []
        for sample in dialogue_rows:
            expected_history_hash = history_hash(prior)
            checkpoint_path = _checkpoint_path(output_dir, str(sample["sample_id"]))
            checkpoint = _compatible_checkpoint(
                checkpoint_path,
                sample=sample,
                model=settings.model,
                expected_history_hash=expected_history_hash,
            )
            if checkpoint is not None:
                produced.append(checkpoint)
                prior.append((str(sample["question"]), str(checkpoint["prediction"])))
                continue

            messages = build_free_running_messages(sample, prior)
            max_tokens = (
                max_tokens_factoid
                if sample["question_type"] == "factoid"
                else max_tokens_non_factoid
            )
            try:
                response = client.generate(
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except Exception as error:
                failed = {
                    "status": "error",
                    "sample_id": sample["sample_id"],
                    "question_type": sample["question_type"],
                    "dialogue_id": sample["dialogue_id"],
                    "turn": sample["turn"],
                    "protocol": Z2_PROTOCOL,
                    "model_requested": settings.model,
                    "prompt_contract": PROMPT_CONTRACT,
                    "history_hash": expected_history_hash,
                    "error_type": type(error).__name__,
                    "error": str(error)[:500],
                }
                atomic_write_json(checkpoint_path, failed)
                produced.append(failed)
                break

            metrics = score_answer(response.text, str(sample["answer"]))
            result = {
                "status": "ok",
                "sample_id": sample["sample_id"],
                "question_type": sample["question_type"],
                "dialogue_id": sample["dialogue_id"],
                "turn": sample["turn"],
                "protocol": Z2_PROTOCOL,
                "provider": settings.name,
                "model": response.model,
                "model_requested": settings.model,
                "prompt_contract": PROMPT_CONTRACT,
                "question": sample["question"],
                "reference": sample["answer"],
                "evidence_ids": sample["evidence_ids"],
                "history_hash": expected_history_hash,
                "messages": messages,
                "prediction": response.text,
                "metrics": metrics,
                "response": asdict(response),
            }
            atomic_write_json(checkpoint_path, result)
            produced.append(result)
            prior.append((str(sample["question"]), str(response.text)))
        return produced

    produced_by_id: dict[str, dict[str, Any]] = {}
    with ResponseCache(cache_path) as cache, client_factory(settings, cache=cache) as client:
        with ThreadPoolExecutor(max_workers=settings.concurrency) as executor:
            futures = {
                executor.submit(run_dialogue, rows, client): key
                for key, rows in grouped.items()
            }
            for future in as_completed(futures):
                for row in future.result():
                    produced_by_id[str(row["sample_id"])] = row

    results: list[dict[str, Any]] = []
    for sample in selected:
        sample_id = str(sample["sample_id"])
        row = produced_by_id.get(sample_id)
        if row is None:
            row = {
                "status": "not_attempted",
                "sample_id": sample_id,
                "question_type": sample["question_type"],
                "dialogue_id": sample["dialogue_id"],
                "turn": sample["turn"],
                "protocol": Z2_PROTOCOL,
                "model_requested": settings.model,
                "error_type": "BlockedByPriorTurn",
            }
        results.append(row)
    atomic_write_jsonl(output_dir / "results.jsonl", results)

    successful = [row for row in results if row["status"] == "ok"]
    metric_names = ("exact_match", "character_f1", "rouge_l_f1")
    usage_names = ("prompt_tokens", "completion_tokens", "total_tokens")
    latencies = [float(row["response"]["latency_seconds"]) for row in successful]
    summary = {
        "schema_version": "longcoqa-z2-free-running-inference-v1",
        "provider": settings.safe_summary(),
        "protocol": Z2_PROTOCOL,
        "prompt_contract": PROMPT_CONTRACT,
        "seed": seed,
        "dialogues_per_type": dialogues_per_type,
        "requested_samples": len(selected),
        "successful_samples": len(successful),
        "failed_samples": len(results) - len(successful),
        "metrics": {
            metric: (
                sum(float(row["metrics"][metric]) for row in successful) / len(successful)
                if successful
                else None
            )
            for metric in metric_names
        },
        "usage": {
            name: sum(int(row["response"]["usage"].get(name, 0)) for row in successful)
            for name in usage_names
        },
        "cache_hits": sum(bool(row["response"].get("cached")) for row in successful),
        "latency_seconds": {
            "mean": sum(latencies) / len(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
        "sample_ids_hash": stable_json_hash([row["sample_id"] for row in results]),
        "results_hash": stable_json_hash(
            [
                {
                    "sample_id": row["sample_id"],
                    "status": row["status"],
                    "prediction": row.get("prediction"),
                }
                for row in results
            ]
        ),
    }
    atomic_write_json(output_dir / "summary.json", summary)
    return summary
