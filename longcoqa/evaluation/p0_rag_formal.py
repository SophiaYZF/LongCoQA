from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from longcoqa.evaluation.llm_judge import (
    REFUSAL_SCORE_FIELDS,
    SCORE_FIELDS,
    build_judge_messages,
    build_refusal_judge_messages,
    extract_evidence_text,
    parse_judge_response,
    parse_refusal_judge_response,
)
from longcoqa.evaluation.rag_quality import (
    aggregate_rag_quality,
    citation_scores,
    make_rag_quality_evaluator,
    parse_citations,
)
from longcoqa.paths import resolve_project_path
from longcoqa.utils.hashing import stable_json_hash
from longcoqa.utils.io import atomic_write_json, atomic_write_jsonl, read_jsonl, read_yaml

TASKS = {"answer_quality", "rag_quality"}
EVIDENCE_MODES = {"none", "released_gold", "retrieved"}
CITATION_MODES = {"none", "not_requested", "retrieved_positions"}


@dataclass(frozen=True)
class P0RagCell:
    cell_id: str
    input_path: Path
    tasks: tuple[str, ...]
    evidence_mode: str
    citation_mode: str


def load_p0_rag_config(path: Path) -> tuple[dict[str, Any], tuple[P0RagCell, ...]]:
    raw = read_yaml(path)
    section = raw.get("p0_rag_formal")
    if not isinstance(section, dict):
        raise ValueError("Config is missing p0_rag_formal mapping")
    raw_cells = section.get("cells")
    if not isinstance(raw_cells, list) or not raw_cells:
        raise ValueError("p0_rag_formal.cells must be a non-empty list")
    cells: list[P0RagCell] = []
    seen: set[str] = set()
    for index, value in enumerate(raw_cells):
        if not isinstance(value, dict):
            raise ValueError(f"cells[{index}] must be a mapping")
        cell_id = str(value.get("cell_id", "")).strip()
        if not cell_id:
            raise ValueError(f"cells[{index}] has an empty cell_id")
        if cell_id in seen:
            raise ValueError(f"Duplicate cell_id: {cell_id}")
        seen.add(cell_id)
        tasks = tuple(str(task) for task in value.get("tasks", ()))
        unknown_tasks = sorted(set(tasks) - TASKS)
        if not tasks or unknown_tasks:
            raise ValueError(f"Invalid tasks for {cell_id}: {unknown_tasks or tasks}")
        evidence_mode = str(value.get("evidence_mode", ""))
        citation_mode = str(value.get("citation_mode", ""))
        if evidence_mode not in EVIDENCE_MODES:
            raise ValueError(f"Invalid evidence_mode for {cell_id}: {evidence_mode}")
        if citation_mode not in CITATION_MODES:
            raise ValueError(f"Invalid citation_mode for {cell_id}: {citation_mode}")
        if "rag_quality" in tasks and evidence_mode == "none":
            raise ValueError(f"rag_quality requires evidence_mode for {cell_id}")
        if citation_mode == "retrieved_positions" and evidence_mode != "retrieved":
            raise ValueError(f"retrieved citation_mode requires retrieved evidence for {cell_id}")
        if evidence_mode == "none" and citation_mode != "none":
            raise ValueError(f"evidence_mode none requires citation_mode none for {cell_id}")
        cells.append(
            P0RagCell(
                cell_id=cell_id,
                input_path=resolve_project_path(str(value.get("input_path", ""))),
                tasks=tasks,
                evidence_mode=evidence_mode,
                citation_mode=citation_mode,
            )
        )
    return section, tuple(cells)


def load_p0_rag_manifest(path: Path | str) -> tuple[P0RagCell, ...]:
    return load_p0_rag_config(Path(path))[1]


def load_evidence_parent_map(
    alignment_path: Path, parents_path: Path
) -> dict[str, tuple[str, ...]]:
    paragraph_to_parents: dict[str, set[str]] = {}
    for parent in read_jsonl(parents_path):
        parent_id = str(parent["parent_id"])
        for paragraph_id in parent.get("paragraph_ids", ()):
            paragraph_to_parents.setdefault(str(paragraph_id), set()).add(parent_id)
    result: dict[str, tuple[str, ...]] = {}
    for alignment in read_jsonl(alignment_path):
        parent_ids: set[str] = set()
        for paragraph_id in alignment.get("paragraph_ids", ()):
            parent_ids.update(paragraph_to_parents.get(str(paragraph_id), ()))
        result[str(alignment["evidence_id"])] = tuple(sorted(parent_ids))
    return result


def citation_position_scores(
    *,
    prediction: str,
    evidence_ids: Sequence[Any],
    retrieved_parent_ids: Sequence[Any],
    evidence_parent_map: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    gold_parents: set[str] = set()
    unmapped_evidence_ids: list[str] = []
    for evidence_id in evidence_ids:
        key = str(evidence_id)
        parents = evidence_parent_map.get(key, ())
        if not parents:
            unmapped_evidence_ids.append(key)
        gold_parents.update(str(parent) for parent in parents)
    retrieved = tuple(str(parent) for parent in retrieved_parent_ids)
    gold_positions = tuple(
        str(index)
        for index, parent_id in enumerate(retrieved, start=1)
        if parent_id in gold_parents
    )
    cited_positions = parse_citations(prediction)
    out_of_range = tuple(
        value
        for value in cited_positions
        if not value.isdigit() or not 1 <= int(value) <= len(retrieved)
    )
    score = citation_scores(
        cited_positions, gold_positions, answerable=bool(evidence_ids)
    )
    return {
        "status": score.status,
        "precision": score.precision,
        "recall": score.recall,
        "f1": score.f1,
        "cited_positions": list(score.cited_ids),
        "gold_positions": list(score.gold_ids),
        "correct_positions": list(score.correct_ids),
        "unknown_positions": list(score.unknown_ids),
        "out_of_range_positions": list(out_of_range),
        "unmapped_evidence_ids": unmapped_evidence_ids,
        "gold_parent_ids": sorted(gold_parents),
    }


def normalize_formal_row(
    row: Mapping[str, Any],
    cell: P0RagCell,
    *,
    evidence_parent_map: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    required = ("sample_id", "question_type", "question", "reference", "model")
    missing = [field for field in required if field not in row]
    if missing:
        raise ValueError(f"{cell.cell_id} row is missing: {', '.join(missing)}")
    prediction = row.get("answer_text") or row.get("prediction")
    if prediction is None:
        raise ValueError(f"{cell.cell_id} row {row['sample_id']} has no prediction")
    normalized = {
        "sample_id": str(row["sample_id"]),
        "dialogue_id": row.get("dialogue_id"),
        "turn": row.get("turn"),
        "question_type": str(row["question_type"]),
        "question": str(row["question"]),
        "reference": str(row["reference"]),
        "prediction": str(prediction),
        "model": str(row["model"]),
        "protocol": cell.cell_id,
        "answerable": (
            True if cell.evidence_mode == "none" else bool(row.get("evidence_ids", ()))
        ),
    }
    if cell.evidence_mode != "none":
        evidence_text = extract_evidence_text(dict(row)) or ""
        if not evidence_text and (
            cell.evidence_mode == "retrieved" or normalized["answerable"]
        ):
            raise ValueError(f"{cell.cell_id} row {row['sample_id']} has no extractable evidence")
        normalized["evidence_text"] = evidence_text
    if cell.citation_mode == "not_requested":
        normalized["citation_status"] = "inapplicable"
        normalized["citation_reason"] = "not_requested"
    elif cell.citation_mode == "retrieved_positions":
        normalized["citation"] = citation_position_scores(
            prediction=str(prediction),
            evidence_ids=row.get("evidence_ids", ()),
            retrieved_parent_ids=row.get("retrieved_parent_ids", ()),
            evidence_parent_map=evidence_parent_map,
        )
    return normalized


def make_answer_quality_evaluator(
    client: Any,
    *,
    max_tokens: int = 512,
    extra_body: Mapping[str, Any] | None = None,
):
    generation_options = deepcopy(dict(extra_body)) if extra_body is not None else None

    def evaluate(row: dict[str, Any]) -> dict[str, Any]:
        answerable = bool(row.get("answerable", True))
        generation_kwargs: dict[str, Any] = {
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        if generation_options is not None:
            generation_kwargs["extra_body"] = deepcopy(generation_options)
        response = client.generate(
            build_judge_messages(row) if answerable else build_refusal_judge_messages(row),
            **generation_kwargs,
        )
        parsed = (
            parse_judge_response(response.text)
            if answerable
            else parse_refusal_judge_response(response.text)
        )
        parsed["judgment_type"] = "answer_quality" if answerable else "refusal_quality"
        return parsed

    return evaluate


def make_formal_rag_quality_evaluator(
    client: Any, *, citation_mode: str, max_tokens: int = 1024
):
    if citation_mode not in {"not_requested", "retrieved_positions"}:
        raise ValueError(f"Invalid formal RAG citation mode: {citation_mode}")
    semantic_evaluator = make_rag_quality_evaluator(client, max_tokens=max_tokens)

    def evaluate(row: dict[str, Any]) -> dict[str, Any]:
        result = semantic_evaluator(row)
        if citation_mode == "not_requested":
            result["citation_status"] = "inapplicable"
            result["citation_reason"] = "not_requested"
        else:
            citation = row.get("citation")
            if not isinstance(citation, Mapping):
                raise ValueError("RAG row is missing deterministic citation mapping")
            result["citation_status"] = str(citation.get("status", "error"))
            result["citation_deterministic"] = dict(citation)
        return result

    return evaluate


def validate_p0_rag_inputs(
    cells: Sequence[P0RagCell],
    *,
    expected_samples: int,
    evidence_parent_map: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    answer_calls = 0
    rag_calls = 0
    answer_rubric_calls = 0
    refusal_rubric_calls = 0
    for cell in cells:
        if not cell.input_path.is_file():
            raise ValueError(f"Missing input for {cell.cell_id}: {cell.input_path}")
        raw_rows = read_jsonl(cell.input_path)
        failed = [row for row in raw_rows if row.get("status") != "ok"]
        if failed:
            raise ValueError(f"{cell.cell_id} contains {len(failed)} non-ok rows")
        if len(raw_rows) != expected_samples:
            raise ValueError(
                f"{cell.cell_id} expected {expected_samples} rows, found {len(raw_rows)}"
            )
        sample_ids = [str(row.get("sample_id", "")) for row in raw_rows]
        if any(not value for value in sample_ids) or len(sample_ids) != len(set(sample_ids)):
            raise ValueError(f"{cell.cell_id} has missing or duplicate sample IDs")
        normalized = [
            normalize_formal_row(row, cell, evidence_parent_map=evidence_parent_map)
            for row in raw_rows
        ]
        answer_calls += len(normalized) if "answer_quality" in cell.tasks else 0
        rag_calls += len(normalized) if "rag_quality" in cell.tasks else 0
        cell_answerable = sum(bool(row["answerable"]) for row in normalized)
        cell_refusal = len(normalized) - cell_answerable
        if "answer_quality" in cell.tasks:
            answer_rubric_calls += cell_answerable
            refusal_rubric_calls += cell_refusal
        reports.append(
            {
                "cell_id": cell.cell_id,
                "rows": len(normalized),
                "tasks": list(cell.tasks),
                "answer_rubric_rows": cell_answerable,
                "refusal_rubric_rows": cell_refusal,
                "status": "valid",
            }
        )
    return {
        "schema_version": "longcoqa-p0-rag-formal-validation-v1",
        "status": "valid",
        "cells": reports,
        "valid_cells": len(reports),
        "answer_quality_calls": answer_calls,
        "answer_rubric_calls": answer_rubric_calls,
        "refusal_rubric_calls": refusal_rubric_calls,
        "rag_quality_calls": rag_calls,
        "total_calls": answer_calls + rag_calls,
    }


def _checkpoint_path(output_dir: Path, sample_id: str) -> Path:
    return output_dir / "checkpoints" / f"{stable_json_hash(sample_id)}.json"


def _compatible_checkpoint(
    path: Path, *, input_hash: str, evaluator_hash: str
) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        value.get("status") == "ok"
        and value.get("input_hash") == input_hash
        and value.get("evaluator_hash") == evaluator_hash
    ):
        return value
    return None


def run_formal_task(
    rows: Sequence[dict[str, Any]],
    *,
    cell_id: str,
    task: str,
    output_dir: Path,
    evaluator: Callable[[dict[str, Any]], dict[str, Any]],
    evaluator_identity: Mapping[str, Any],
    expected_samples: int,
    max_workers: int = 1,
    aggregate_evaluations: Callable[
        [Sequence[Mapping[str, Any]]], Mapping[str, Any]
    ]
    | None = None,
) -> dict[str, Any]:
    if aggregate_evaluations is None and task not in TASKS:
        raise ValueError(f"Unknown formal task without aggregator: {task}")
    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(rows) != expected_samples or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Formal task input count/sample IDs do not meet the strict contract")
    evaluator_identity = dict(evaluator_identity)
    evaluator_hash = stable_json_hash(evaluator_identity)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / ".complete").unlink(missing_ok=True)
    records: dict[str, dict[str, Any]] = {}
    pending: list[tuple[dict[str, Any], str, Path]] = []
    reused = 0
    for row in rows:
        sample_id = str(row["sample_id"])
        input_hash = stable_json_hash(row)
        checkpoint_path = _checkpoint_path(output_dir, sample_id)
        checkpoint = _compatible_checkpoint(
            checkpoint_path, input_hash=input_hash, evaluator_hash=evaluator_hash
        )
        if checkpoint is not None:
            records[sample_id] = checkpoint
            reused += 1
        else:
            pending.append((row, input_hash, checkpoint_path))

    def evaluate(entry: tuple[dict[str, Any], str, Path]) -> dict[str, Any]:
        row, input_hash, checkpoint_path = entry
        started = time.monotonic()
        try:
            evaluation = evaluator(row)
            record = {
                "schema_version": "longcoqa-p0-rag-formal-checkpoint-v1",
                "status": "ok",
                "cell_id": cell_id,
                "task": task,
                "sample_id": str(row["sample_id"]),
                "question_type": row.get("question_type"),
                "answer_model": row.get("model"),
                "input_hash": input_hash,
                "evaluator_hash": evaluator_hash,
                "evaluator_identity": evaluator_identity,
                "elapsed_seconds": time.monotonic() - started,
                "evaluation": evaluation,
            }
        except Exception as error:
            record = {
                "schema_version": "longcoqa-p0-rag-formal-checkpoint-v1",
                "status": "error",
                "cell_id": cell_id,
                "task": task,
                "sample_id": str(row["sample_id"]),
                "question_type": row.get("question_type"),
                "answer_model": row.get("model"),
                "input_hash": input_hash,
                "evaluator_hash": evaluator_hash,
                "evaluator_identity": evaluator_identity,
                "elapsed_seconds": time.monotonic() - started,
                "error_type": type(error).__name__,
                "error": str(error)[:1000],
            }
        atomic_write_json(checkpoint_path, record)
        return record

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        futures = {executor.submit(evaluate, entry): entry for entry in pending}
        for future in as_completed(futures):
            record = future.result()
            records[str(record["sample_id"])] = record
    ordered = [records[sample_id] for sample_id in sample_ids]
    atomic_write_jsonl(output_dir / "results.jsonl", ordered)
    successful = sum(row.get("status") == "ok" for row in ordered)
    successful_evaluations = [
        row["evaluation"] for row in ordered if row.get("status") == "ok"
    ]
    if aggregate_evaluations is not None:
        aggregate = dict(aggregate_evaluations(successful_evaluations))
    elif task == "answer_quality":
        answer_evaluations = [
            value
            for value in successful_evaluations
            if value.get("judgment_type", "answer_quality") == "answer_quality"
        ]
        refusal_evaluations = [
            value
            for value in successful_evaluations
            if value.get("judgment_type") == "refusal_quality"
        ]
        score_values = {
            field: [
                float(value[field])
                for value in answer_evaluations
                if value.get(field) is not None
            ]
            for field in SCORE_FIELDS
        }
        refusal_score_values = {
            field: [
                float(value[field])
                for value in refusal_evaluations
                if value.get(field) is not None
            ]
            for field in REFUSAL_SCORE_FIELDS
        }
        aggregate = {
            "answer_quality": {
                "samples": len(answer_evaluations),
                "scores": {
                    field: sum(values) / len(values) if values else None
                    for field, values in score_values.items()
                },
                "correct_or_mostly_correct_rate": (
                    sum(
                        value.get("verdict") in {"correct", "mostly_correct"}
                        for value in answer_evaluations
                    )
                    / len(answer_evaluations)
                    if answer_evaluations
                    else None
                ),
            },
            "refusal_quality": {
                "samples": len(refusal_evaluations),
                "scores": {
                    field: sum(values) / len(values) if values else None
                    for field, values in refusal_score_values.items()
                },
                "appropriate_refusal_rate": (
                    sum(
                        value.get("verdict") == "appropriate_refusal"
                        for value in refusal_evaluations
                    )
                    / len(refusal_evaluations)
                    if refusal_evaluations
                    else None
                ),
            },
        }
    else:
        aggregate = aggregate_rag_quality(successful_evaluations)
    summary = {
        "schema_version": "longcoqa-p0-rag-formal-summary-v1",
        "cell_id": cell_id,
        "task": task,
        "requested_samples": len(ordered),
        "successful_samples": successful,
        "failed_samples": len(ordered) - successful,
        "reused_samples": reused,
        "evaluator_hash": evaluator_hash,
        "evaluator_identity": evaluator_identity,
        "aggregate": aggregate,
    }
    atomic_write_json(output_dir / "summary.json", summary)
    if successful == expected_samples:
        atomic_write_json(output_dir / ".complete", summary)
    return summary


def validate_formal_task_output(
    output_dir: Path,
    *,
    expected_sample_ids: Sequence[str],
    evaluator_identity: Mapping[str, Any],
) -> dict[str, Any]:
    results_path = output_dir / "results.jsonl"
    if not results_path.is_file() or not (output_dir / ".complete").is_file():
        raise ValueError(f"Formal task is incomplete: {output_dir}")
    rows = read_jsonl(results_path)
    actual_ids = [str(row.get("sample_id", "")) for row in rows]
    if actual_ids != list(expected_sample_ids) or len(actual_ids) != len(set(actual_ids)):
        raise ValueError("Formal output sample IDs do not exactly match the input")
    evaluator_hash = stable_json_hash(dict(evaluator_identity))
    invalid = [
        row
        for row in rows
        if row.get("status") != "ok" or row.get("evaluator_hash") != evaluator_hash
    ]
    if invalid:
        raise ValueError(f"Formal output contains {len(invalid)} failed/incompatible rows")
    return {"status": "complete", "samples": len(rows), "evaluator_hash": evaluator_hash}


def write_matrix_summary(
    cells: Sequence[P0RagCell], *, output_root: Path, expected_samples: int
) -> dict[str, Any]:
    task_cells: list[dict[str, Any]] = []
    complete_count = 0
    for cell in cells:
        for task in cell.tasks:
            directory = output_root / task / cell.cell_id / "full"
            summary_path = directory / "summary.json"
            complete_path = directory / ".complete"
            status = "incomplete"
            summary: dict[str, Any] = {}
            if summary_path.is_file() and complete_path.is_file():
                try:
                    loaded = json.loads(summary_path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        summary = loaded
                    if (
                        summary.get("cell_id") == cell.cell_id
                        and summary.get("task") == task
                        and summary.get("requested_samples") == expected_samples
                        and summary.get("successful_samples") == expected_samples
                        and summary.get("failed_samples") == 0
                    ):
                        status = "complete"
                        complete_count += 1
                except (OSError, json.JSONDecodeError):
                    pass
            task_cells.append(
                {
                    "cell_id": cell.cell_id,
                    "task": task,
                    "status": status,
                    "output_dir": str(directory),
                }
            )
    matrix_status = "complete" if complete_count == len(task_cells) else "incomplete"
    report = {
        "schema_version": "longcoqa-p0-rag-formal-matrix-v1",
        "status": matrix_status,
        "expected_task_cells": len(task_cells),
        "completed_task_cells": complete_count,
        "expected_samples_per_cell": expected_samples,
        "task_cells": task_cells,
    }
    atomic_write_json(output_root / "matrix_summary.json", report)
    matrix_complete = output_root / ".complete"
    if matrix_status == "complete":
        atomic_write_json(matrix_complete, report)
    else:
        matrix_complete.unlink(missing_ok=True)
    return report
