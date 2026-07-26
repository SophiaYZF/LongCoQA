from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from longcoqa.config import ProviderSettings
from longcoqa.data import LongCoQADataset
from longcoqa.evaluation.judge_runner import run_llm_judge_inputs, select_judge_inputs
from longcoqa.utils.hashing import stable_json_hash
from longcoqa.utils.io import atomic_write_json, read_jsonl

MODEL_VARIANTS = ("base", "conversation", "grounded", "raft")
EVAL_VARIANTS = ("conversation", "grounded", "raft")
EXPECTED_SAMPLES = {"conversation": 602, "grounded": 539, "raft": 539}
EXPECTED_RUBRICS = {
    "conversation": {"answer_quality": 602, "refusal_quality": 0},
    "grounded": {"answer_quality": 539, "refusal_quality": 0},
    "raft": {"answer_quality": 484, "refusal_quality": 55},
}


def expected_matrix_totals(
    model_count: int, eval_variants: tuple[str, ...]
) -> dict[str, int]:
    if model_count < 1:
        raise ValueError("model_count must be positive")
    unknown = set(eval_variants) - set(EXPECTED_RUBRICS)
    if unknown:
        raise ValueError(f"Unknown P1 evaluation variants: {sorted(unknown)}")
    per_model_answer = sum(
        EXPECTED_RUBRICS[evaluation]["answer_quality"]
        for evaluation in eval_variants
    )
    per_model_refusal = sum(
        EXPECTED_RUBRICS[evaluation]["refusal_quality"]
        for evaluation in eval_variants
    )
    return {
        "successful_samples": model_count
        * (per_model_answer + per_model_refusal),
        "answer_quality_samples": model_count * per_model_answer,
        "refusal_quality_samples": model_count * per_model_refusal,
    }


def _dataset_question_index() -> dict[str, str]:
    return {
        str(sample["sample_id"]): str(sample["question"])
        for sample in LongCoQADataset().iter_samples()
    }


def normalize_p1_inputs(
    rows: list[dict[str, Any]],
    *,
    question_index: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Convert P1 inference rows into the common judge input schema."""
    questions = question_index if question_index is not None else _dataset_question_index()
    normalized: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id not in questions:
            raise ValueError(f"Unknown P1 sample ID: {sample_id}")
        item = dict(row)
        item["question"] = questions[sample_id]
        item["prediction"] = str(row.get("answer_text") or row["prediction"])
        item["judgment_type"] = (
            "answer_quality" if bool(row.get("answerable", True)) else "refusal_quality"
        )
        normalized.append(item)
    return normalized


def select_p1_judge_inputs(
    inputs: list[dict[str, Any]], limit: int | None
) -> list[dict[str, Any]]:
    """Select a deterministic preflight, covering both rubrics when possible."""
    if limit is None or limit >= len(inputs):
        return inputs
    if limit < 2:
        return select_judge_inputs(inputs, limit)

    by_rubric: dict[str, list[dict[str, Any]]] = {
        "answer_quality": [],
        "refusal_quality": [],
    }
    for item in inputs:
        rubric = str(item.get("judgment_type", "answer_quality"))
        if rubric in by_rubric:
            by_rubric[rubric].append(item)
    if not all(by_rubric.values()):
        return select_judge_inputs(inputs, limit)

    selected: list[dict[str, Any]] = []
    for rubric in ("answer_quality", "refusal_quality"):
        selected.append(
            min(
                by_rubric[rubric],
                key=lambda item: stable_json_hash(str(item["sample_id"])),
            )
        )
    selected_ids = {id(item) for item in selected}
    remaining = [item for item in inputs if id(item) not in selected_ids]
    selected.extend(select_judge_inputs(remaining, limit - len(selected)))
    return selected


def validate_p1_inputs(
    rows: list[dict[str, Any]], *, expected_samples: int
) -> None:
    if len(rows) != expected_samples:
        raise ValueError(f"Expected {expected_samples} P1 inputs, found {len(rows)}")
    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Duplicate sample IDs in P1 judge inputs")
    failures = [row for row in rows if row.get("status") != "ok"]
    if failures:
        raise ValueError(f"P1 judge inputs contain {len(failures)} failed rows")


def validate_p1_judgments(
    results: list[dict[str, Any]],
    *,
    expected_samples: int,
    expected_judge_model: str,
    expected_answer_quality: int,
    expected_refusal_quality: int,
) -> dict[str, int]:
    if len(results) != expected_samples:
        raise ValueError(f"Expected {expected_samples} judgments, found {len(results)}")
    failures = [row for row in results if row.get("status") != "ok"]
    if failures:
        raise ValueError(f"P1 judge output contains {len(failures)} failed rows")
    sample_ids = [str(row["sample_id"]) for row in results]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Duplicate sample IDs in P1 judgments")
    wrong_models = {
        str(row.get("judge_model"))
        for row in results
        if row.get("judge_model") != expected_judge_model
    }
    if wrong_models:
        raise ValueError(f"Unexpected judge model(s): {sorted(wrong_models)}")
    counts = Counter(str(row.get("judgment_type")) for row in results)
    actual = {
        "answer_quality": counts["answer_quality"],
        "refusal_quality": counts["refusal_quality"],
    }
    expected = {
        "answer_quality": expected_answer_quality,
        "refusal_quality": expected_refusal_quality,
    }
    if actual != expected:
        raise ValueError(f"Unexpected judgment rubric counts: {actual}; expected {expected}")
    return actual


def run_p1_judge_matrix(
    settings: ProviderSettings,
    *,
    input_root: Path,
    output_root: Path,
    cache_path: Path,
    expected_samples: dict[str, int] | None = None,
    model_variants: tuple[str, ...] = MODEL_VARIANTS,
    eval_variants: tuple[str, ...] = EVAL_VARIANTS,
    model_variant: str | None = None,
    eval_variant: str | None = None,
    limit: int | None = None,
    max_tokens: int = 512,
) -> dict[str, Any]:
    """Judge one requested P1 cell or the complete 4x3 matrix."""
    if (model_variant is None) != (eval_variant is None):
        raise ValueError("model_variant and eval_variant must be supplied together")
    expected_by_eval = expected_samples or EXPECTED_SAMPLES
    cells = (
        [(model_variant, eval_variant)]
        if model_variant is not None and eval_variant is not None
        else [(model, evaluation) for model in model_variants for evaluation in eval_variants]
    )
    questions = _dataset_question_index()
    cell_summaries: list[dict[str, Any]] = []

    for model, evaluation in cells:
        if model not in model_variants or evaluation not in eval_variants:
            raise ValueError(f"Unknown P1 cell: {model}/{evaluation}")
        input_path = input_root / model / evaluation / "full" / "results.jsonl"
        rows = read_jsonl(input_path)
        expected_count = int(expected_by_eval[evaluation])
        validate_p1_inputs(rows, expected_samples=expected_count)
        normalized = normalize_p1_inputs(rows, question_index=questions)
        selected = select_p1_judge_inputs(normalized, limit)
        cell_output = output_root / model / evaluation / "full"
        summary = run_llm_judge_inputs(
            selected,
            settings,
            input_label=str(input_path),
            cache_path=cache_path,
            output_dir=cell_output,
            limit=None,
            max_tokens=max_tokens,
        )
        judgments = read_jsonl(cell_output / "judgments.jsonl")
        if limit is None:
            rubric_counts = EXPECTED_RUBRICS[evaluation]
            validate_p1_judgments(
                judgments,
                expected_samples=expected_count,
                expected_judge_model=settings.model,
                expected_answer_quality=rubric_counts["answer_quality"],
                expected_refusal_quality=rubric_counts["refusal_quality"],
            )
        cell_summaries.append(
            {
                "model_variant": model,
                "eval_variant": evaluation,
                "output": str(cell_output),
                **summary,
            }
        )

    matrix_summary = {
        "schema_version": "longcoqa-p1-judge-matrix-v1",
        "judge": settings.safe_summary(),
        "full_matrix": model_variant is None and eval_variant is None and limit is None,
        "cells": cell_summaries,
        "successful_samples": sum(
            int(cell["successful_samples"]) for cell in cell_summaries
        ),
        "failed_samples": sum(int(cell["failed_samples"]) for cell in cell_summaries),
        "answer_quality_samples": sum(
            int(cell["answer_quality"]["successful_samples"])
            for cell in cell_summaries
        ),
        "refusal_quality_samples": sum(
            int(cell["refusal_quality"]["successful_samples"])
            for cell in cell_summaries
        ),
    }
    atomic_write_json(output_root / "matrix_summary.json", matrix_summary)
    if matrix_summary["full_matrix"]:
        expected_totals = expected_matrix_totals(
            len(model_variants), eval_variants
        )
        actual_totals = {
            key: int(matrix_summary[key]) for key in expected_totals
        }
        if actual_totals != expected_totals or matrix_summary["failed_samples"] != 0:
            raise ValueError(f"Invalid full P1 judge matrix totals: {matrix_summary}")
        atomic_write_json(output_root / ".complete", matrix_summary)
    return matrix_summary
