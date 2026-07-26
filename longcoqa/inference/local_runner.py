from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from longcoqa.data import LongCoQADataset
from longcoqa.evaluation.text_metrics import score_answer
from longcoqa.inference.api_runner import _load_evidence_maps, _select_smoke_samples
from longcoqa.inference.prompts import build_messages, index_samples
from longcoqa.utils.hashing import stable_json_hash
from longcoqa.utils.io import atomic_write_json, atomic_write_jsonl

REASONING_CLOSE_RE = re.compile(r"</(?:think|analysis)>", re.IGNORECASE)


def normalize_final_answer(text: str) -> str:
    matches = list(REASONING_CLOSE_RE.finditer(text))
    if not matches:
        return text
    return text[matches[-1].end() :].strip()


def _checkpoint_path(checkpoint_dir: Path, sample_id: str) -> Path:
    return checkpoint_dir / f"{sample_id.replace(':', '__')}.json"


def _checkpoint_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "ok"
    except (json.JSONDecodeError, OSError):
        return False


def summarize_local_results(
    results: list[dict[str, Any]],
    *,
    model: dict[str, Any],
    protocol: str,
    seed: int,
) -> dict[str, Any]:
    successful = [item for item in results if item["status"] == "ok"]
    metric_names = ("exact_match", "character_f1", "rouge_l_f1")
    return {
        "schema_version": "longcoqa-local-inference-v1",
        "model": model,
        "protocol": protocol,
        "seed": seed,
        "requested_samples": len(results),
        "successful_samples": len(successful),
        "failed_samples": len(results) - len(successful),
        "metrics": {
            metric: (
                sum(float(item["metrics"][metric]) for item in successful) / len(successful)
                if successful
                else None
            )
            for metric in metric_names
        },
        "usage": {
            "prompt_tokens": sum(int(item["usage"]["prompt_tokens"]) for item in successful),
            "completion_tokens": sum(int(item["usage"]["completion_tokens"]) for item in successful),
        },
        "sample_ids_hash": stable_json_hash([item["sample_id"] for item in results]),
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


def _create_engine(model: dict[str, Any]) -> Any:
    from vllm import LLM

    return LLM(
        model=str(model["path"]),
        tensor_parallel_size=int(model["tensor_parallel_size"]),
        dtype=str(model.get("dtype", "bfloat16")),
        gpu_memory_utilization=float(model["gpu_memory_utilization"]),
        max_model_len=int(model["max_model_len"]),
        max_num_seqs=int(model["max_num_seqs"]),
        trust_remote_code=bool(model.get("trust_remote_code", False)),
        enforce_eager=bool(model.get("enforce_eager", False)),
    )


def run_local_inference(
    dataset: LongCoQADataset,
    model: dict[str, Any],
    *,
    protocol: str,
    alignment_path: Path,
    output_dir: Path,
    seed: int = 42,
    limit: int | None = None,
    temperature: float = 0.0,
    max_tokens_factoid: int = 256,
    max_tokens_non_factoid: int = 768,
    engine: Any | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict[str, Any]:
    # CUDA visibility must be fixed before vLLM/torch is imported.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(model["cuda_visible_devices"])
    from vllm import SamplingParams

    all_samples = list(dataset.iter_samples())
    selected = _select_smoke_samples(all_samples, limit=limit, seed=seed)
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError(
            f"Invalid shard_index={shard_index}, shard_count={shard_count}"
        )
    selected = [
        sample
        for index, sample in enumerate(selected)
        if index % shard_count == shard_index
    ]
    dialogues = index_samples(all_samples)
    released, aligned = _load_evidence_maps(dataset, alignment_path)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    pending = [
        sample
        for sample in selected
        if not _checkpoint_complete(_checkpoint_path(checkpoint_dir, str(sample["sample_id"])))
    ]

    if pending:
        owns_engine = engine is None
        if engine is None:
            engine = _create_engine(model)
        messages = [
            build_messages(
                sample,
                protocol=protocol,
                dialogues=dialogues,
                released_evidence=released,
                aligned_source=aligned,
            )
            for sample in pending
        ]
        sampling_params = [
            SamplingParams(
                temperature=temperature,
                max_tokens=(
                    max_tokens_factoid
                    if sample["question_type"] == "factoid"
                    else max_tokens_non_factoid
                ),
            )
            for sample in pending
        ]
        outputs = engine.chat(
            messages,
            sampling_params=sampling_params,
            chat_template_kwargs={"enable_thinking": False},
            use_tqdm=True,
        )
        for sample, sample_messages, request_output in zip(
            pending, messages, outputs, strict=True
        ):
            candidate = request_output.outputs[0]
            prediction = normalize_final_answer(candidate.text)
            result = {
                "status": "ok",
                "sample_id": sample["sample_id"],
                "question_type": sample["question_type"],
                "dialogue_id": sample["dialogue_id"],
                "turn": sample["turn"],
                "protocol": protocol,
                "provider": "local_vllm",
                "model": model["name"],
                "question": sample["question"],
                "reference": sample["answer"],
                "evidence_ids": sample["evidence_ids"],
                "messages": sample_messages,
                "prediction": prediction,
                "metrics": score_answer(prediction, str(sample["answer"])),
                "usage": {
                    "prompt_tokens": len(request_output.prompt_token_ids or []),
                    "completion_tokens": len(candidate.token_ids),
                },
                "finish_reason": candidate.finish_reason,
            }
            atomic_write_json(
                _checkpoint_path(checkpoint_dir, str(sample["sample_id"])),
                result,
            )
        if owns_engine:
            del engine

    results = [
        json.loads(
            _checkpoint_path(checkpoint_dir, str(sample["sample_id"])).read_text(
                encoding="utf-8"
            )
        )
        for sample in selected
    ]
    atomic_write_jsonl(output_dir / "results.jsonl", results)
    summary = summarize_local_results(
        results,
        model={
            key: model[key]
            for key in (
                "name",
                "path",
                "revision",
                "dtype",
                "tensor_parallel_size",
                "gpu_memory_utilization",
                "max_model_len",
                "max_num_seqs",
            )
        },
        protocol=protocol,
        seed=seed,
    )
    summary["shard"] = {"index": shard_index, "count": shard_count}
    atomic_write_json(output_dir / "summary.json", summary)
    return summary


def run_local_suite(
    dataset: LongCoQADataset,
    model: dict[str, Any],
    *,
    protocols: tuple[str, ...],
    alignment_path: Path,
    output_root: Path,
    seed: int = 42,
    limit: int | None = None,
    temperature: float = 0.0,
    max_tokens_factoid: int = 256,
    max_tokens_non_factoid: int = 768,
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict[str, Any]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(model["cuda_visible_devices"])
    engine = _create_engine(model)
    reports: dict[str, Any] = {}
    sample_label = f"smoke{limit}" if limit is not None else "full"
    try:
        for protocol in protocols:
            reports[protocol] = run_local_inference(
                dataset,
                model,
                protocol=protocol,
                alignment_path=alignment_path,
                output_dir=output_root / protocol / sample_label,
                seed=seed,
                limit=limit,
                temperature=temperature,
                max_tokens_factoid=max_tokens_factoid,
                max_tokens_non_factoid=max_tokens_non_factoid,
                engine=engine,
                shard_index=shard_index,
                shard_count=shard_count,
            )
    finally:
        del engine
    suite_summary = {
        "schema_version": "longcoqa-local-suite-v1",
        "model": model["name"],
        "protocols": list(protocols),
        "seed": seed,
        "samples_per_protocol": limit,
        "shard": {"index": shard_index, "count": shard_count},
        "reports": reports,
    }
    atomic_write_json(output_root / f"suite_{sample_label}.json", suite_summary)
    return suite_summary
