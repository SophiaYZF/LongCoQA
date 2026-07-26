from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from longcoqa.evaluation.text_metrics import score_answer
from longcoqa.utils.hashing import sha256_file, stable_json_hash
from longcoqa.utils.io import atomic_write_json, atomic_write_jsonl, read_jsonl


def summarize_rag_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    metric_names = ("exact_match", "character_f1", "rouge_l_f1")
    return {
        name: sum(float(row["metrics"][name]) for row in rows) / len(rows)
        for name in metric_names
    }


def checkpoint_is_current(path: Path, contexts_checksum: str) -> bool:
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return (
        isinstance(value, dict)
        and value.get("status") == "ok"
        and value.get("contexts_checksum") == contexts_checksum
    )


def run_local_rag(
    *,
    model: dict[str, Any],
    contexts_path: Path,
    output_dir: Path,
    limit: int | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict[str, Any]:
    from vllm import LLM, SamplingParams

    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("Invalid RAG shard")
    rows = read_jsonl(contexts_path)
    if limit is not None:
        rows = rows[:limit]
    rows = [row for index, row in enumerate(rows) if index % shard_count == shard_index]
    contexts_checksum = sha256_file(contexts_path)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    pending = [
        row
        for row in rows
        if not checkpoint_is_current(
            checkpoint_dir / f"{str(row['sample_id']).replace(':', '__')}.json",
            contexts_checksum,
        )
    ]
    if pending:
        engine = LLM(
            model=str(model["path"]),
            dtype=str(model["dtype"]),
            tensor_parallel_size=int(model["tensor_parallel_size"]),
            gpu_memory_utilization=float(model["gpu_memory_utilization"]),
            max_model_len=int(model["max_model_len"]),
            max_num_seqs=int(model["max_num_seqs"]),
            trust_remote_code=bool(model.get("trust_remote_code", False)),
        )
        sampling = [
            SamplingParams(
                temperature=0,
                max_tokens=96 if row["question_type"] == "factoid" else 768,
            )
            for row in pending
        ]
        outputs = engine.chat(
            [row["messages"] for row in pending],
            sampling_params=sampling,
            chat_template_kwargs={"enable_thinking": False},
            use_tqdm=True,
        )
        for row, output in zip(pending, outputs, strict=True):
            prediction = output.outputs[0].text
            value = {
                **row,
                "status": "ok",
                "model": model["name"],
                "contexts_checksum": contexts_checksum,
                "prediction": prediction,
                "metrics": score_answer(prediction, str(row["reference"])),
                "usage": {
                    "prompt_tokens": len(output.prompt_token_ids or []),
                    "completion_tokens": len(output.outputs[0].token_ids),
                },
            }
            atomic_write_json(
                checkpoint_dir / f"{str(row['sample_id']).replace(':', '__')}.json",
                value,
            )
        del engine
    results = [
        json.loads(
            (
                checkpoint_dir / f"{str(row['sample_id']).replace(':', '__')}.json"
            ).read_text(encoding="utf-8")
        )
        for row in rows
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    atomic_write_jsonl(results_path, results)
    summary = {
        "schema_version": "longcoqa-rag-inference-v1",
        "model": {key: model[key] for key in ("name", "path", "revision")},
        "examples": len(results),
        "shard": {"index": shard_index, "count": shard_count},
        "metrics": summarize_rag_metrics(results),
        "contexts_checksum": contexts_checksum,
        "sample_ids_hash": stable_json_hash([row["sample_id"] for row in results]),
        "artifact_checksum": sha256_file(results_path),
    }
    atomic_write_json(output_dir / "summary.json", summary)
    return summary


def merge_rag_shards(
    *,
    shard_root: Path,
    output_dir: Path,
    shard_count: int = 8,
    expected_samples: int = 3019,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    shard_checksums: dict[str, str] = {}
    shard_summaries: dict[int, dict[str, Any] | None] = {}
    for shard_index in range(shard_count):
        shard_dir = shard_root / f"shard_{shard_index:02d}"
        path = shard_dir / "results.jsonl"
        if not path.is_file():
            raise ValueError(f"Missing RAG shard: {path}")
        rows = read_jsonl(path)
        results.extend(rows)
        shard_checksums[str(shard_index)] = sha256_file(path)
        summary_path = shard_dir / "summary.json"
        shard_summaries[shard_index] = (
            json.loads(summary_path.read_text(encoding="utf-8"))
            if summary_path.is_file()
            else None
        )
    sample_ids = [str(row["sample_id"]) for row in results]
    if len(results) != expected_samples:
        raise ValueError(
            f"Expected {expected_samples} RAG results, found {len(results)}"
        )
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Duplicate sample IDs across RAG shards")
    non_ok = [str(row["sample_id"]) for row in results if row.get("status") != "ok"]
    if non_ok:
        raise ValueError(f"RAG shards contain non-ok rows: {non_ok[:5]}")
    context_checksums: set[str] = set()
    for shard_index, summary in shard_summaries.items():
        if summary is None:
            raise ValueError(f"Missing RAG shard summary for shard {shard_index}")
        shard = summary.get("shard", {})
        if int(shard.get("index", -1)) != shard_index or int(
            shard.get("count", -1)
        ) != shard_count:
            raise ValueError(f"Invalid shard metadata for shard {shard_index}")
        checksum = str(summary.get("contexts_checksum", ""))
        if not checksum:
            raise ValueError(f"Missing context checksum for shard {shard_index}")
        context_checksums.add(checksum)
    if len(context_checksums) != 1:
        raise ValueError("RAG shards have inconsistent context checksum values")
    contexts_checksum = next(iter(context_checksums))
    results.sort(
        key=lambda row: (
            str(row["question_type"]),
            int(row["dialogue_id"]),
            int(row["turn"]),
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "results.jsonl"
    atomic_write_jsonl(path, results)
    summary = {
        "schema_version": "longcoqa-rag-inference-merged-v1",
        "examples": len(results),
        "shards": shard_count,
        "contexts_checksum": contexts_checksum,
        "metrics": summarize_rag_metrics(results),
        "shard_checksums": shard_checksums,
        "sample_ids_hash": stable_json_hash([row["sample_id"] for row in results]),
        "artifact_checksum": sha256_file(path),
    }
    atomic_write_json(output_dir / "summary.json", summary)
    return summary
