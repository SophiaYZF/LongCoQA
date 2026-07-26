from __future__ import annotations

import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from longcoqa.data import LongCoQADataset
from longcoqa.evaluation.text_metrics import score_answer
from longcoqa.utils.hashing import sha256_file, stable_json_hash
from longcoqa.utils.io import atomic_write_json, atomic_write_jsonl, read_jsonl

ABSTENTION_PATTERNS = (
    "证据不足",
    "信息不足",
    "无法回答",
    "不能回答",
    "无法确定",
    "不能确定",
)


def training_record_to_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": str(record["system"])}]
    for turn in record.get("history", []):
        if not isinstance(turn, list) or len(turn) != 2:
            raise ValueError(f"Invalid history turn for {record.get('metadata')}: {turn!r}")
        messages.extend(
            [
                {"role": "user", "content": str(turn[0])},
                {"role": "assistant", "content": str(turn[1])},
            ]
        )
    instruction = str(record["instruction"])
    query = str(record.get("input", ""))
    if query:
        instruction = f"{instruction}\n\n{query}"
    messages.append({"role": "user", "content": instruction})
    return messages


def strip_citation_footer(text: str) -> str:
    return re.split(r"\n*\s*引用\s*[：:]", text, maxsplit=1)[0].strip()


def parse_citation_ids(text: str) -> list[str]:
    match = re.search(r"引用\s*[：:]\s*(.*)$", text, flags=re.DOTALL)
    if match is None:
        return []
    values = re.findall(r"(?:EVIDENCE\s+)?((?:NEG-)?\d+)", match.group(1), flags=re.IGNORECASE)
    return list(dict.fromkeys(values))


def is_abstention(text: str) -> bool:
    answer = strip_citation_footer(text)
    return any(pattern in answer for pattern in ABSTENTION_PATTERNS)


def _select_records(
    records: list[dict[str, Any]], *, limit: int | None, seed: int
) -> list[dict[str, Any]]:
    if limit is None or limit >= len(records):
        return records
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        metadata = record["metadata"]
        key = f"{metadata['question_type']}:{metadata.get('raft_mode', 'standard')}"
        strata[key].append(record)
    rng = random.Random(seed)
    for rows in strata.values():
        rng.shuffle(rows)
    selected: list[dict[str, Any]] = []
    keys = sorted(strata)
    while len(selected) < limit:
        progressed = False
        for key in keys:
            if strata[key]:
                selected.append(strata[key].pop())
                progressed = True
                if len(selected) == limit:
                    break
        if not progressed:
            break
    return selected


def load_p1_records(
    path: Path,
    *,
    eval_variant: str,
    limit: int | None = None,
    seed: int = 42,
    expected_split: str = "test",
) -> list[dict[str, Any]]:
    records = read_jsonl(path)
    sample_ids: list[str] = []
    for record in records:
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"Missing metadata in {path}")
        if metadata.get("split") != expected_split:
            raise ValueError(
                f"Expected split={expected_split!r} in evaluation input: {metadata}"
            )
        if metadata.get("variant") != eval_variant:
            raise ValueError(
                f"Expected variant={eval_variant!r}, found {metadata.get('variant')!r}"
            )
        sample_ids.append(str(metadata["sample_id"]))
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"Duplicate sample IDs in {path}")
    return _select_records(records, limit=limit, seed=seed)


def _checkpoint_path(
    output_dir: Path, model_variant: str, eval_variant: str, sample_id: str
) -> Path:
    safe_id = sample_id.replace(":", "__")
    return output_dir / model_variant / eval_variant / "checkpoints" / f"{safe_id}.json"


def _checkpoint_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "ok"
    except (json.JSONDecodeError, OSError):
        return False


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize_p1_results(
    results: list[dict[str, Any]],
    *,
    model: dict[str, Any],
    model_variant: str,
    eval_variant: str,
    seed: int,
) -> dict[str, Any]:
    successful = [item for item in results if item.get("status") == "ok"]
    answerable = [item for item in successful if item.get("answerable", True)]
    metric_names = ("exact_match", "character_f1", "rouge_l_f1")

    cited = 0
    gold = 0
    correct_citations = 0
    for item in successful:
        cited_ids = set(map(str, item.get("cited_evidence_ids", [])))
        gold_ids = set(map(str, item.get("gold_evidence_ids", [])))
        cited += len(cited_ids)
        gold += len(gold_ids)
        correct_citations += len(cited_ids & gold_ids)

    if eval_variant == "raft":
        true_positive = sum(item["abstained"] and not item["answerable"] for item in successful)
        false_positive = sum(item["abstained"] and item["answerable"] for item in successful)
        false_negative = sum(not item["abstained"] and not item["answerable"] for item in successful)
    else:
        true_positive = false_positive = false_negative = 0
    abstention_precision = _safe_ratio(true_positive, true_positive + false_positive)
    abstention_recall = _safe_ratio(true_positive, true_positive + false_negative)
    abstention_f1 = (
        2 * abstention_precision * abstention_recall / (abstention_precision + abstention_recall)
        if abstention_precision is not None
        and abstention_recall is not None
        and abstention_precision + abstention_recall
        else None
    )

    def metric_average(rows: list[dict[str, Any]], metric: str) -> float | None:
        values = [float(row["metrics"][metric]) for row in rows if row.get("metrics")]
        return sum(values) / len(values) if values else None

    by_question_type: dict[str, Any] = {}
    for question_type in ("factoid", "non_factoid"):
        rows = [item for item in answerable if item["question_type"] == question_type]
        by_question_type[question_type] = {
            "answerable_samples": len(rows),
            "metrics": {metric: metric_average(rows, metric) for metric in metric_names},
        }

    by_raft_mode: dict[str, Any] = {}
    for mode in sorted({str(item.get("raft_mode")) for item in successful if item.get("raft_mode")}):
        rows = [item for item in successful if item.get("raft_mode") == mode]
        by_raft_mode[mode] = {
            "samples": len(rows),
            "abstention_rate": _safe_ratio(sum(item["abstained"] for item in rows), len(rows)),
        }

    return {
        "schema_version": "longcoqa-p1-inference-v1",
        "model": model,
        "model_variant": model_variant,
        "eval_variant": eval_variant,
        "seed": seed,
        "requested_samples": len(results),
        "successful_samples": len(successful),
        "failed_samples": len(results) - len(successful),
        "answerable_samples": len(answerable),
        "metrics": {metric: metric_average(answerable, metric) for metric in metric_names},
        "by_question_type": by_question_type,
        "by_raft_mode": by_raft_mode,
        "citation": {
            "precision": _safe_ratio(correct_citations, cited),
            "recall": _safe_ratio(correct_citations, gold),
            "cited": cited,
            "gold": gold,
            "correct": correct_citations,
        },
        "abstention": {
            "precision": abstention_precision,
            "recall": abstention_recall,
            "f1": abstention_f1,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
        },
        "usage": {
            "prompt_tokens": sum(int(item["usage"]["prompt_tokens"]) for item in successful),
            "completion_tokens": sum(
                int(item["usage"]["completion_tokens"]) for item in successful
            ),
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


def _create_engine(model: dict[str, Any], *, max_lora_rank: int) -> Any:
    from vllm import LLM

    kwargs: dict[str, Any] = {
        "model": str(model["path"]),
        "tensor_parallel_size": int(model["tensor_parallel_size"]),
        "dtype": str(model.get("dtype", "bfloat16")),
        "gpu_memory_utilization": float(model["gpu_memory_utilization"]),
        "max_model_len": int(model["max_model_len"]),
        "max_num_seqs": int(model["max_num_seqs"]),
        "trust_remote_code": bool(model.get("trust_remote_code", False)),
        "enforce_eager": bool(model.get("enforce_eager", False)),
        "enable_lora": True,
        "max_lora_rank": max_lora_rank,
        "max_loras": 1,
        "max_cpu_loras": 4,
    }
    if model.get("max_num_batched_tokens") is not None:
        kwargs["max_num_batched_tokens"] = int(model["max_num_batched_tokens"])
    if model.get("enable_chunked_prefill") is not None:
        kwargs["enable_chunked_prefill"] = bool(model["enable_chunked_prefill"])
    return LLM(**kwargs)


def _adapter_request(model_variant: str, adapter_path: Path | None, adapter_id: int) -> Any:
    if adapter_path is None:
        return None
    from vllm.lora.request import LoRARequest

    return LoRARequest(model_variant, adapter_id, str(adapter_path))


def _model_metadata(
    model: dict[str, Any], *, model_variant: str, adapter_path: Path | None
) -> dict[str, Any]:
    metadata = {
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
    }
    metadata["variant"] = model_variant
    metadata["adapter_path"] = str(adapter_path) if adapter_path else None
    if adapter_path:
        metadata["adapter_config_sha256"] = sha256_file(adapter_path / "adapter_config.json")
        metadata["adapter_model_sha256"] = sha256_file(
            adapter_path / "adapter_model.safetensors"
        )
    return metadata


def run_p1_suite(
    *,
    model: dict[str, Any],
    model_variants: dict[str, Path | None],
    eval_sets: dict[str, Path],
    output_root: Path,
    seed: int = 42,
    limit: int | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
    temperature: float = 0.0,
    max_tokens_factoid: int = 256,
    max_tokens_non_factoid: int = 768,
    max_lora_rank: int = 64,
    expected_split: str = "test",
) -> dict[str, Any]:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError(f"Invalid shard_index={shard_index}, shard_count={shard_count}")
    # DLC/debug launchers bind each lane to one physical GPU. Do not replace
    # that external binding with the model config's single-GPU default.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(model["cuda_visible_devices"]))

    dataset_samples = {
        str(item["sample_id"]): item for item in LongCoQADataset().iter_samples()
    }
    records_by_eval = {
        variant: load_p1_records(
            path,
            eval_variant=variant,
            limit=limit,
            seed=seed,
            expected_split=expected_split,
        )
        for variant, path in eval_sets.items()
    }
    for adapter_path in model_variants.values():
        if adapter_path is None:
            continue
        for filename in ("adapter_config.json", "adapter_model.safetensors"):
            if not (adapter_path / filename).is_file():
                raise FileNotFoundError(adapter_path / filename)

    from vllm import SamplingParams

    engine = _create_engine(model, max_lora_rank=max_lora_rank)
    reports: dict[str, Any] = {}
    try:
        for adapter_id, (model_variant, adapter_path) in enumerate(
            model_variants.items(), start=1
        ):
            lora_request = _adapter_request(model_variant, adapter_path, adapter_id)
            model_metadata = _model_metadata(
                model, model_variant=model_variant, adapter_path=adapter_path
            )
            reports[model_variant] = {}
            for eval_variant, all_records in records_by_eval.items():
                selected = [
                    record
                    for index, record in enumerate(all_records)
                    if index % shard_count == shard_index
                ]
                run_dir = output_root / model_variant / eval_variant
                pending = [
                    record
                    for record in selected
                    if not _checkpoint_complete(
                        _checkpoint_path(
                            output_root,
                            model_variant,
                            eval_variant,
                            str(record["metadata"]["sample_id"]),
                        )
                    )
                ]
                if pending:
                    messages = [training_record_to_messages(record) for record in pending]
                    sampling_params = [
                        SamplingParams(
                            temperature=temperature,
                            max_tokens=(
                                max_tokens_factoid
                                if record["metadata"]["question_type"] == "factoid"
                                else max_tokens_non_factoid
                            ),
                        )
                        for record in pending
                    ]
                    outputs = engine.chat(
                        messages,
                        sampling_params=sampling_params,
                        lora_request=lora_request,
                        chat_template_kwargs={"enable_thinking": False},
                        use_tqdm=True,
                    )
                    for record, sample_messages, request_output in zip(
                        pending, messages, outputs, strict=True
                    ):
                        metadata = record["metadata"]
                        sample_id = str(metadata["sample_id"])
                        candidate = request_output.outputs[0]
                        prediction = candidate.text
                        raft_mode = metadata.get("raft_mode")
                        answerable = raft_mode != "missing_gold_abstention"
                        reference = str(dataset_samples[sample_id]["answer"])
                        answer_text = strip_citation_footer(prediction)
                        gold_evidence_ids = (
                            list(
                                map(
                                    str,
                                    metadata.get(
                                        "gold_evidence_ids",
                                        dataset_samples[sample_id]["evidence_ids"],
                                    ),
                                )
                            )
                            if eval_variant != "conversation"
                            else []
                        )
                        result = {
                            "status": "ok",
                            "sample_id": sample_id,
                            "question_type": str(metadata["question_type"]),
                            "dialogue_id": int(metadata["dialogue_id"]),
                            "turn": int(metadata["turn"]),
                            "model_variant": model_variant,
                            "eval_variant": eval_variant,
                            "raft_mode": raft_mode,
                            "answerable": answerable,
                            "provider": "local_vllm",
                            "model": str(model["name"]),
                            "reference": reference,
                            "target_output": str(record["output"]),
                            "messages": sample_messages,
                            "prediction": prediction,
                            "answer_text": answer_text,
                            "abstained": is_abstention(prediction),
                            "gold_evidence_ids": gold_evidence_ids,
                            "cited_evidence_ids": parse_citation_ids(prediction),
                            "metrics": score_answer(answer_text, reference) if answerable else None,
                            "usage": {
                                "prompt_tokens": len(request_output.prompt_token_ids or []),
                                "completion_tokens": len(candidate.token_ids),
                            },
                            "finish_reason": candidate.finish_reason,
                        }
                        atomic_write_json(
                            _checkpoint_path(
                                output_root, model_variant, eval_variant, sample_id
                            ),
                            result,
                        )

                results = [
                    json.loads(
                        _checkpoint_path(
                            output_root,
                            model_variant,
                            eval_variant,
                            str(record["metadata"]["sample_id"]),
                        ).read_text(encoding="utf-8")
                    )
                    for record in selected
                ]
                atomic_write_jsonl(run_dir / "results.jsonl", results)
                report = summarize_p1_results(
                    results,
                    model=model_metadata,
                    model_variant=model_variant,
                    eval_variant=eval_variant,
                    seed=seed,
                )
                report["shard"] = {"index": shard_index, "count": shard_count}
                report["p1_test_file"] = str(eval_sets[eval_variant])
                atomic_write_json(run_dir / "summary.json", report)
                reports[model_variant][eval_variant] = report
    finally:
        del engine

    suite = {
        "schema_version": "longcoqa-p1-shard-suite-v1",
        "seed": seed,
        "limit": limit,
        "shard": {"index": shard_index, "count": shard_count},
        "model_variants": list(model_variants),
        "eval_variants": list(eval_sets),
        "reports": reports,
    }
    atomic_write_json(output_root / "suite.json", suite)
    return suite


def merge_p1_shards(
    *,
    shard_root: Path,
    output_root: Path,
    model_variants: tuple[str, ...],
    eval_sets: dict[str, Path],
    shard_count: int,
    seed: int = 42,
    limit: int | None = None,
    expected_split: str = "test",
) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for model_variant in model_variants:
        reports[model_variant] = {}
        for eval_variant, eval_path in eval_sets.items():
            expected = len(
                load_p1_records(
                    eval_path,
                    eval_variant=eval_variant,
                    limit=limit,
                    seed=seed,
                    expected_split=expected_split,
                )
            )
            results: list[dict[str, Any]] = []
            shard_summaries: list[dict[str, Any]] = []
            for shard_index in range(shard_count):
                run_dir = (
                    shard_root
                    / f"shard_{shard_index:02d}"
                    / model_variant
                    / eval_variant
                )
                summary_path = run_dir / "summary.json"
                results_path = run_dir / "results.jsonl"
                if not summary_path.is_file() or not results_path.is_file():
                    raise FileNotFoundError(f"Incomplete P1 shard output: {run_dir}")
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if summary.get("failed_samples") != 0:
                    raise ValueError(f"P1 shard has failed samples: {summary_path}")
                shard_summaries.append(summary)
                results.extend(read_jsonl(results_path))

            sample_ids = [str(item["sample_id"]) for item in results]
            if len(results) != expected:
                raise ValueError(
                    f"{model_variant}/{eval_variant}: expected {expected}, found {len(results)}"
                )
            if len(set(sample_ids)) != len(sample_ids):
                raise ValueError(f"{model_variant}/{eval_variant}: duplicate sample IDs")
            if any(item.get("status") != "ok" for item in results):
                raise ValueError(f"{model_variant}/{eval_variant}: failed merged row")
            results.sort(
                key=lambda item: (
                    str(item["question_type"]),
                    int(item["dialogue_id"]),
                    int(item["turn"]),
                )
            )
            target = output_root / model_variant / eval_variant / "full"
            atomic_write_jsonl(target / "results.jsonl", results)
            report = summarize_p1_results(
                results,
                model=dict(shard_summaries[0]["model"]),
                model_variant=model_variant,
                eval_variant=eval_variant,
                seed=seed,
            )
            report["merged_shards"] = shard_count
            report["expected_samples"] = expected
            report["p1_test_file"] = str(eval_path)
            atomic_write_json(target / "summary.json", report)
            reports[model_variant][eval_variant] = report

    suite = {
        "schema_version": "longcoqa-p1-merged-suite-v1",
        "shard_root": str(shard_root),
        "shard_count": shard_count,
        "limit": limit,
        "model_variants": list(model_variants),
        "eval_variants": list(eval_sets),
        "reports": reports,
    }
    atomic_write_json(output_root / "suite_full.json", suite)
    return suite
