from __future__ import annotations

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

from longcoqa.config import ProviderSettings
from longcoqa.evaluation.llm_judge import (
    REFUSAL_SCORE_FIELDS,
    SCORE_FIELDS,
    build_judge_messages,
    build_refusal_judge_messages,
    parse_judge_response,
    parse_refusal_judge_response,
)
from longcoqa.providers import ProviderClient
from longcoqa.providers.cache import ResponseCache
from longcoqa.utils.hashing import stable_json_hash
from longcoqa.utils.io import atomic_write_json, atomic_write_jsonl, read_jsonl


def summarize_judgments(results: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [item for item in results if item["status"] == "ok"]
    answer_results = [
        item
        for item in successful
        if item.get("judgment_type", "answer_quality") == "answer_quality"
    ]
    refusal_results = [
        item
        for item in successful
        if item.get("judgment_type", "answer_quality") == "refusal_quality"
    ]
    by_type: dict[str, dict[str, Any]] = {}
    for question_type in ("factoid", "non_factoid"):
        subset = [
            item
            for item in answer_results
            if item["question_type"] == question_type
        ]
        by_type[question_type] = {
            "samples": len(subset),
            "scores": {
                field: (
                    sum(float(item["judgment"][field]) for item in subset) / len(subset)
                    if subset
                    else None
                )
                for field in SCORE_FIELDS
            },
            "correct_or_mostly_correct_rate": (
                sum(
                    item["judgment"]["verdict"] in {"correct", "mostly_correct"}
                    for item in subset
                )
                / len(subset)
                if subset
                else None
            ),
            "factual_conflict_rate": (
                sum(item["judgment"]["has_factual_conflict"] for item in subset)
                / len(subset)
                if subset
                else None
            ),
            "unsupported_claim_rate": (
                sum(item["judgment"]["has_unsupported_claim"] for item in subset)
                / len(subset)
                if subset
                else None
            ),
        }
    answer_correctness = (
        sum(float(item["judgment"]["correctness"]) for item in answer_results)
        / len(answer_results)
        if answer_results
        else None
    )
    answer_summary = {
        "successful_samples": len(answer_results),
        "judge_100": (
            25.0 * (answer_correctness - 1.0)
            if answer_correctness is not None
            else None
        ),
        "by_question_type": by_type,
    }
    refusal_summary = {
        "successful_samples": len(refusal_results),
        "scores": {
            field: (
                sum(float(item["judgment"][field]) for item in refusal_results)
                / len(refusal_results)
                if refusal_results
                else None
            )
            for field in REFUSAL_SCORE_FIELDS
        },
        "appropriate_refusal_rate": (
            sum(
                item["judgment"]["verdict"] == "appropriate_refusal"
                for item in refusal_results
            )
            / len(refusal_results)
            if refusal_results
            else None
        ),
        "unsupported_answer_rate": (
            sum(item["judgment"]["has_unsupported_answer"] for item in refusal_results)
            / len(refusal_results)
            if refusal_results
            else None
        ),
        "irrelevant_citation_rate": (
            sum(
                item["judgment"]["cites_irrelevant_evidence"]
                for item in refusal_results
            )
            / len(refusal_results)
            if refusal_results
            else None
        ),
    }
    return {
        "successful_samples": len(successful),
        "failed_samples": len(results) - len(successful),
        "by_question_type": by_type,
        "answer_quality": answer_summary,
        "refusal_quality": refusal_summary,
    }


def select_judge_inputs(
    inputs: list[dict[str, Any]], limit: int | None
) -> list[dict[str, Any]]:
    if limit is None or limit >= len(inputs):
        return inputs
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in inputs:
        turn = int(item.get("turn", 1))
        turn_bucket = "turn_1" if turn == 1 else "turn_2_4" if turn <= 4 else "turn_5_plus"
        strata[(str(item["question_type"]), turn_bucket)].append(item)
    for rows in strata.values():
        rows.sort(key=lambda item: stable_json_hash(str(item["sample_id"])))
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


def run_llm_judge(
    input_path: Path,
    settings: ProviderSettings,
    *,
    cache_path: Path,
    output_dir: Path,
    limit: int | None = None,
    max_tokens: int = 512,
) -> dict[str, Any]:
    return run_llm_judge_inputs(
        [item for item in read_jsonl(input_path) if item.get("status") == "ok"],
        settings,
        input_label=str(input_path),
        cache_path=cache_path,
        output_dir=output_dir,
        limit=limit,
        max_tokens=max_tokens,
    )


def run_llm_judge_inputs(
    raw_inputs: list[dict[str, Any]],
    settings: ProviderSettings,
    *,
    input_label: str,
    cache_path: Path,
    output_dir: Path,
    limit: int | None = None,
    max_tokens: int = 512,
) -> dict[str, Any]:
    inputs = select_judge_inputs(raw_inputs, limit)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def checkpoint_path(item: dict[str, Any]) -> Path:
        run = item.get(
            "protocol",
            item.get("retrieval_method", item.get("eval_variant", "unknown")),
        )
        variant = item.get("model_variant", item["model"])
        judgment_type = item.get("judgment_type", "answer_quality")
        identifier = (
            f"{variant}__{run}__{judgment_type}__{item['sample_id']}"
        ).replace(":", "__")
        return checkpoint_dir / f"{identifier}.json"

    def complete(item: dict[str, Any]) -> bool:
        path = checkpoint_path(item)
        if not path.is_file():
            return False
        try:
            return json.loads(path.read_text(encoding="utf-8")).get("status") == "ok"
        except (json.JSONDecodeError, OSError):
            return False

    pending = [item for item in inputs if not complete(item)]
    with ResponseCache(cache_path) as cache, ProviderClient(settings, cache=cache) as client:

        def judge(item: dict[str, Any]) -> dict[str, Any]:
            judgment_type = item.get("judgment_type", "answer_quality")
            if judgment_type == "answer_quality":
                messages = build_judge_messages(item)
                parser = parse_judge_response
            elif judgment_type == "refusal_quality":
                messages = build_refusal_judge_messages(item)
                parser = parse_refusal_judge_response
            else:
                raise ValueError(f"Unknown judgment_type: {judgment_type}")
            response = client.generate(
                messages,
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return {
                "status": "ok",
                "sample_id": item["sample_id"],
                "question_type": item["question_type"],
                "judgment_type": judgment_type,
                "answer_model": item["model"],
                "answer_protocol": item.get(
                    "protocol",
                    item.get("retrieval_method", item.get("eval_variant", "unknown")),
                ),
                "judge_provider": settings.name,
                "judge_model": response.model,
                "judgment": parser(response.text),
                "judge_response": asdict(response),
                "answer_hash": stable_json_hash(
                    {
                        "question": item["question"],
                        "reference": (
                            item["reference"]
                            if judgment_type == "answer_quality"
                            else None
                        ),
                        "prediction": item["prediction"],
                        "judgment_type": judgment_type,
                    }
                ),
            }

        with ThreadPoolExecutor(max_workers=settings.concurrency) as executor:
            futures = {executor.submit(judge, item): item for item in pending}
            for future in as_completed(futures):
                item = futures[future]
                try:
                    result = future.result()
                except Exception as error:
                    result = {
                        "status": "error",
                        "sample_id": item["sample_id"],
                        "question_type": item["question_type"],
                        "judgment_type": item.get(
                            "judgment_type", "answer_quality"
                        ),
                        "answer_model": item["model"],
                        "judge_model": settings.model,
                        "error_type": type(error).__name__,
                        "error": str(error)[:500],
                    }
                atomic_write_json(checkpoint_path(item), result)

    results = [
        json.loads(checkpoint_path(item).read_text(encoding="utf-8"))
        for item in inputs
    ]
    atomic_write_jsonl(output_dir / "judgments.jsonl", results)
    summary = {
        "schema_version": "longcoqa-llm-judge-v1",
        "input": input_label,
        "input_hash": stable_json_hash(
            [
                {
                    "sample_id": item["sample_id"],
                    "model": item["model"],
                    "prediction": item["prediction"],
                    "judgment_type": item.get(
                        "judgment_type", "answer_quality"
                    ),
                }
                for item in inputs
            ]
        ),
        "judge": settings.safe_summary(),
        **summarize_judgments(results),
    }
    atomic_write_json(output_dir / "summary.json", summary)
    return summary
