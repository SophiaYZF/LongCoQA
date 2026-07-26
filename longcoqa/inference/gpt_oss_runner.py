from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from longcoqa.data import LongCoQADataset
from longcoqa.evaluation.text_metrics import score_answer
from longcoqa.inference.api_runner import _load_evidence_maps, _select_smoke_samples
from longcoqa.inference.local_runner import (
    _checkpoint_path,
    summarize_local_results,
)
from longcoqa.inference.prompts import build_messages, index_samples
from longcoqa.utils.io import atomic_write_json, atomic_write_jsonl

HarmonyParser = Callable[[Sequence[int]], tuple[str | None, str | None, bool]]
HARMONY_CONTROL_RE = re.compile(
    r"<\|(?:start|end|channel|message|return|call)[^|]*\|>"
)
FORMAT_REPAIR_SUFFIX = "[格式修复] 请仅输出简短的中文最终答案。"


def contains_harmony_control(text: str) -> bool:
    return bool(HARMONY_CONTROL_RE.search(text))


def _default_harmony_parser(token_ids: Sequence[int]) -> tuple[str | None, str | None, bool]:
    from vllm.entrypoints.openai.parser.harmony_utils import parse_chat_output

    return parse_chat_output(token_ids)


def parse_harmony_candidate(
    token_ids: Sequence[int],
    *,
    parser: HarmonyParser = _default_harmony_parser,
) -> tuple[str, str | None]:
    """Return visible final text and optional hidden analysis from Harmony tokens."""
    reasoning, final, is_tool_call = parser(token_ids)
    if is_tool_call:
        raise ValueError("unexpected Harmony tool call")
    if final is None or not final.strip():
        raise ValueError("missing final Harmony channel")
    final = final.strip()
    if contains_harmony_control(final):
        raise ValueError("Harmony control token leaked into final")
    normalized_reasoning = reasoning.strip() if reasoning and reasoning.strip() else None
    return final, normalized_reasoning


def build_gpt_oss_result(
    sample: dict[str, Any],
    sample_messages: list[dict[str, str]],
    request_output: Any,
    *,
    model: dict[str, Any],
    protocol: str,
    generation_max_tokens: int,
    parser: HarmonyParser = _default_harmony_parser,
) -> dict[str, Any]:
    """Build one auditable record while scoring only the Harmony final channel."""
    candidate = request_output.outputs[0]
    result = {
        "sample_id": sample["sample_id"],
        "question_type": sample["question_type"],
        "dialogue_id": sample["dialogue_id"],
        "turn": sample["turn"],
        "protocol": protocol,
        "provider": "local_vllm_harmony",
        "model": model["name"],
        "question": sample["question"],
        "reference": sample["answer"],
        "evidence_ids": sample["evidence_ids"],
        "messages": sample_messages,
        "response_format": model["response_format"],
        "reasoning_effort": model["reasoning_effort"],
        "generation_max_tokens": generation_max_tokens,
        "usage": {
            "prompt_tokens": len(request_output.prompt_token_ids or []),
            "completion_tokens": len(candidate.token_ids),
        },
        "finish_reason": candidate.finish_reason,
    }
    try:
        prediction, reasoning = parse_harmony_candidate(
            list(candidate.token_ids), parser=parser
        )
    except (ValueError, RuntimeError) as exc:
        result.update(
            {
                "status": "failed",
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        return result
    result.update(
        {
            "status": "ok",
            "prediction": prediction,
            "reasoning": reasoning,
            "metrics": score_answer(prediction, str(sample["answer"])),
        }
    )
    return result


def _run_gpt_oss_requests(
    engine: Any,
    samples: list[dict[str, Any]],
    messages: list[list[dict[str, str]]],
    sampling_params: list[Any],
    *,
    model: dict[str, Any],
    protocol: str,
    generation_budget: Callable[[dict[str, Any]], int],
    parser: HarmonyParser = _default_harmony_parser,
    max_parse_retries: int = 2,
) -> list[dict[str, Any]]:
    """Generate a batch and isolate-retry only malformed Harmony responses."""
    outputs = engine.chat(
        messages,
        sampling_params=sampling_params,
        chat_template_kwargs={
            "reasoning_effort": str(model["reasoning_effort"])
        },
        use_tqdm=True,
    )
    results = [
        build_gpt_oss_result(
            sample,
            sample_messages,
            request_output,
            model=model,
            protocol=protocol,
            generation_max_tokens=generation_budget(sample),
            parser=parser,
        )
        for sample, sample_messages, request_output in zip(
            samples, messages, outputs, strict=True
        )
    ]
    for result in results:
        result["generation_attempt"] = 1

    for attempt in range(2, max_parse_retries + 2):
        failed_indices = [
            index for index, result in enumerate(results) if result["status"] != "ok"
        ]
        if not failed_indices:
            break
        for index in failed_indices:
            retry_messages = messages[index]
            if attempt >= 3:
                retry_messages = [dict(message) for message in messages[index]]
                retry_messages[-1]["content"] = (
                    f"{retry_messages[-1]['content']}\n\n{FORMAT_REPAIR_SUFFIX}"
                )
            retry_output = engine.chat(
                [retry_messages],
                sampling_params=[sampling_params[index]],
                chat_template_kwargs={
                    "reasoning_effort": str(model["reasoning_effort"])
                },
                use_tqdm=False,
            )[0]
            result = build_gpt_oss_result(
                samples[index],
                retry_messages,
                retry_output,
                model=model,
                protocol=protocol,
                generation_max_tokens=generation_budget(samples[index]),
                parser=parser,
            )
            result["generation_attempt"] = attempt
            if attempt >= 3:
                result["recovery_strategy"] = "format_repair_suffix"
            results[index] = result
    return results


def _gpt_oss_checkpoint_current(path: Path, generation_max_tokens: int) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        isinstance(payload, dict)
        and payload.get("status") == "ok"
        and payload.get("generation_max_tokens") == generation_max_tokens
    )


def _validate_model(model: dict[str, Any]) -> None:
    if model.get("response_format") != "harmony":
        raise ValueError("gpt-oss runner requires response_format=harmony")
    if model.get("reasoning_effort") != "low":
        raise ValueError("gpt-oss P0 requires reasoning_effort=low")
    if int(model.get("tensor_parallel_size", 0)) != 1:
        raise ValueError("gpt-oss P0 lanes require tensor_parallel_size=1")


def _create_engine(model: dict[str, Any]) -> Any:
    from vllm import LLM

    _validate_model(model)
    return LLM(
        model=str(model["path"]),
        tensor_parallel_size=int(model["tensor_parallel_size"]),
        dtype=str(model.get("dtype", "auto")),
        gpu_memory_utilization=float(model["gpu_memory_utilization"]),
        max_model_len=int(model["max_model_len"]),
        max_num_seqs=int(model["max_num_seqs"]),
        trust_remote_code=bool(model.get("trust_remote_code", False)),
        enforce_eager=bool(model.get("enforce_eager", False)),
    )


def run_gpt_oss_inference(
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
    parser: HarmonyParser = _default_harmony_parser,
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict[str, Any]:
    """Run one protocol with explicit Harmony parsing and resumable checkpoints."""
    _validate_model(model)
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
    def generation_budget(sample: dict[str, Any]) -> int:
        return (
            max_tokens_factoid
            if sample["question_type"] == "factoid"
            else max_tokens_non_factoid
        )

    pending = [
        sample
        for sample in selected
        if not _gpt_oss_checkpoint_current(
            _checkpoint_path(checkpoint_dir, str(sample["sample_id"])),
            generation_budget(sample),
        )
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
                max_tokens=generation_budget(sample),
                seed=seed,
                skip_special_tokens=False,
            )
            for sample in pending
        ]
        results = _run_gpt_oss_requests(
            engine,
            pending,
            messages,
            sampling_params,
            model=model,
            protocol=protocol,
            generation_budget=generation_budget,
            parser=parser,
        )
        for sample, result in zip(pending, results, strict=True):
            atomic_write_json(
                _checkpoint_path(checkpoint_dir, str(sample["sample_id"])), result
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
                "response_format",
                "reasoning_effort",
            )
        },
        protocol=protocol,
        seed=seed,
    )
    summary["schema_version"] = "longcoqa-gpt-oss-inference-v1"
    summary["shard"] = {"index": shard_index, "count": shard_count}
    atomic_write_json(output_dir / "summary.json", summary)
    return summary


def run_gpt_oss_suite(
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
    parser: HarmonyParser = _default_harmony_parser,
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict[str, Any]:
    """Reuse one one-GPU gpt-oss engine across the requested protocols."""
    _validate_model(model)
    if not protocols:
        raise ValueError("at least one protocol is required")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(model["cuda_visible_devices"])
    engine = _create_engine(model)
    reports: dict[str, Any] = {}
    sample_label = f"smoke{limit}" if limit is not None else "full"
    try:
        for protocol in protocols:
            reports[protocol] = run_gpt_oss_inference(
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
                parser=parser,
                shard_index=shard_index,
                shard_count=shard_count,
            )
    finally:
        del engine
    suite_summary = {
        "schema_version": "longcoqa-gpt-oss-suite-v1",
        "model": model["name"],
        "response_format": model["response_format"],
        "reasoning_effort": model["reasoning_effort"],
        "protocols": list(protocols),
        "seed": seed,
        "samples_per_protocol": limit,
        "shard": {"index": shard_index, "count": shard_count},
        "reports": reports,
    }
    atomic_write_json(output_root / f"suite_{sample_label}.json", suite_summary)
    return suite_summary
