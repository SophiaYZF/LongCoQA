from __future__ import annotations

from pathlib import Path
from typing import Any

from longcoqa.rag.context import normalize_ranking_row
from longcoqa.utils.io import read_jsonl


def validate_ranking_artifact(
    path: Path,
    *,
    ranking_level: str,
    expected_samples: int,
) -> dict[str, Any]:
    rows = read_jsonl(path)
    if len(rows) != expected_samples:
        raise ValueError(
            f"Ranking artifact expected {expected_samples} rows, found {len(rows)}"
        )
    normalized = [normalize_ranking_row(row, ranking_level) for row in rows]
    sample_ids = [str(row["sample_id"]) for row in normalized]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Duplicate ranking sample IDs")
    return {
        "examples": len(rows),
        "unique_sample_ids": len(set(sample_ids)),
        "ranking_level": ranking_level,
    }


def validate_generation_artifact(
    path: Path,
    *,
    expected_samples: int,
) -> dict[str, Any]:
    rows = read_jsonl(path)
    if len(rows) != expected_samples:
        raise ValueError(
            f"Generation artifact expected {expected_samples} rows, found {len(rows)}"
        )
    sample_ids: list[str] = []
    for index, row in enumerate(rows, start=1):
        sample_id = row.get("sample_id")
        if sample_id is None or not str(sample_id):
            raise ValueError(f"Generation row {index} is missing sample_id")
        if row.get("status") != "ok":
            raise ValueError(f"Generation row {sample_id} has non-ok status")
        sample_ids.append(str(sample_id))
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Duplicate generation sample IDs")
    return {
        "examples": len(rows),
        "unique_sample_ids": len(set(sample_ids)),
        "status": "ok",
    }
