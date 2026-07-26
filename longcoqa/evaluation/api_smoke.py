from __future__ import annotations

import json
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from longcoqa.utils.hashing import stable_json_hash
from longcoqa.utils.io import atomic_write_json, atomic_write_jsonl

ApiEvaluator = Callable[[dict[str, Any]], Mapping[str, Any]]


def turn_bucket(turn: int) -> str:
    if turn < 1:
        raise ValueError("turn must be positive")
    if turn == 1:
        return "turn_1"
    if turn <= 4:
        return "turn_2_4"
    return "turn_5_plus"


def select_stratified_smoke12(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    strata: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") != "ok":
            continue
        question_type = str(row["question_type"])
        if question_type not in {"factoid", "non_factoid"}:
            raise ValueError(f"Unknown question type: {question_type}")
        strata[(question_type, turn_bucket(int(row["turn"])))].append(row)

    selected: list[dict[str, Any]] = []
    for question_type in ("factoid", "non_factoid"):
        for bucket in ("turn_1", "turn_2_4", "turn_5_plus"):
            key = (question_type, bucket)
            candidates = strata.get(key, [])
            label = f"{question_type}:{bucket}"
            if not candidates:
                raise ValueError(f"API smoke is missing required stratum {label}")
            if len(candidates) < 2:
                raise ValueError(f"API smoke stratum {label} requires at least 2 rows")
            ordered = sorted(
                candidates,
                key=lambda row: (
                    stable_json_hash(str(row["sample_id"])),
                    str(row["sample_id"]),
                ),
            )
            selected.extend(dict(row) for row in ordered[:2])
    return selected


def validate_api_smoke_request(
    *,
    smoke: bool,
    limit: int,
    cells: Sequence[str],
    output_root: Path,
) -> None:
    if not smoke:
        raise ValueError("External API evaluation requires explicit --smoke")
    if limit > 12:
        raise ValueError("External API smoke limit cannot exceed 12")
    if limit != 12:
        raise ValueError("External API smoke uses exactly 12 stratified rows")
    if len(cells) != 1:
        raise ValueError("External API smoke must target exactly one cell")
    if "runtime" not in output_root.parts or "paper_metrics_api_smoke" not in output_root.parts:
        raise ValueError("External API output must use the runtime smoke root")


def _safe_cell(cell: str) -> str:
    return cell.replace("/", "__").replace(":", "__")


def _safe_sample(sample_id: str) -> str:
    return sample_id.replace(":", "__").replace("/", "__")


def _load_checkpoint(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def run_api_smoke(
    rows: Sequence[Mapping[str, Any]],
    *,
    cell: str,
    output_root: Path,
    evaluator: ApiEvaluator,
    smoke: bool,
    limit: int,
    evaluator_identity: Mapping[str, Any],
) -> dict[str, Any]:
    validate_api_smoke_request(
        smoke=smoke,
        limit=limit,
        cells=(cell,),
        output_root=output_root,
    )
    selected = select_stratified_smoke12(rows)
    identity = dict(evaluator_identity)
    evaluator_hash = stable_json_hash(identity)
    cell_root = output_root / _safe_cell(cell)
    results: list[dict[str, Any]] = []

    for row in selected:
        sample_id = str(row["sample_id"])
        input_hash = stable_json_hash(dict(row))
        checkpoint_path = cell_root / "checkpoints" / f"{_safe_sample(sample_id)}.json"
        checkpoint = _load_checkpoint(checkpoint_path)
        if (
            checkpoint
            and checkpoint.get("status") == "ok"
            and checkpoint.get("input_hash") == input_hash
            and checkpoint.get("evaluator_hash") == evaluator_hash
        ):
            results.append(checkpoint)
            continue

        base = {
            "schema_version": "longcoqa-api-smoke-result-v1",
            "cell": cell,
            "sample_id": sample_id,
            "question_type": str(row["question_type"]),
            "dialogue_id": int(row["dialogue_id"]),
            "turn": int(row["turn"]),
            "turn_bucket": turn_bucket(int(row["turn"])),
            "input_hash": input_hash,
            "evaluator": identity,
            "evaluator_hash": evaluator_hash,
        }
        started = time.perf_counter()
        try:
            evaluation = dict(evaluator(dict(row)))
            result = {
                **base,
                "status": "ok",
                "evaluation": evaluation,
                "elapsed_seconds": time.perf_counter() - started,
            }
        except Exception as error:
            result = {
                **base,
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error)[:1000],
                "elapsed_seconds": time.perf_counter() - started,
            }
        atomic_write_json(checkpoint_path, result)
        results.append(result)

    atomic_write_jsonl(cell_root / "results.jsonl", results)
    summary = {
        "schema_version": "longcoqa-api-smoke-summary-v1",
        "smoke": True,
        "cell": cell,
        "requested_samples": len(results),
        "successful_samples": sum(row["status"] == "ok" for row in results),
        "failed_samples": sum(row["status"] != "ok" for row in results),
        "sample_ids": [row["sample_id"] for row in results],
        "sample_ids_hash": stable_json_hash([row["sample_id"] for row in results]),
        "evaluator": identity,
        "evaluator_hash": evaluator_hash,
        "output": str(cell_root / "results.jsonl"),
    }
    atomic_write_json(cell_root / "summary.json", summary)
    return summary
