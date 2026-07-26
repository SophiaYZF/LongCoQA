import json
from pathlib import Path

import pytest

from longcoqa.rag.validation import (
    validate_generation_artifact,
    validate_ranking_artifact,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("ranking_level", "id_key"),
    [("child", "child_id"), ("parent", "parent_id")],
)
def test_validate_ranking_artifact_accepts_supported_schemas(
    tmp_path: Path, ranking_level: str, id_key: str
) -> None:
    path = tmp_path / "ranking.jsonl"
    _write_jsonl(
        path,
        [
            {"sample_id": "a", "query": "qa", "ranked": [{id_key: "x", "score": 1}]},
            {"id": "b", "question": "qb", "ranked": [{id_key: "y", "score": 0.5}]},
        ],
    )

    assert validate_ranking_artifact(
        path, ranking_level=ranking_level, expected_samples=2
    ) == {
        "examples": 2,
        "unique_sample_ids": 2,
        "ranking_level": ranking_level,
    }


def test_validate_ranking_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "ranking.jsonl"
    row = {"id": "same", "question": "q", "ranked": [{"parent_id": "p", "score": 1}]}
    _write_jsonl(path, [row, row])
    with pytest.raises(ValueError, match="Duplicate ranking sample IDs"):
        validate_ranking_artifact(path, ranking_level="parent", expected_samples=2)


def test_validate_ranking_rejects_wrong_count_and_empty_ranked(tmp_path: Path) -> None:
    path = tmp_path / "ranking.jsonl"
    _write_jsonl(path, [{"id": "a", "question": "q", "ranked": []}])
    with pytest.raises(ValueError, match="expected 2.*found 1"):
        validate_ranking_artifact(path, ranking_level="parent", expected_samples=2)
    with pytest.raises(ValueError, match="has no ranked results"):
        validate_ranking_artifact(path, ranking_level="parent", expected_samples=1)


def test_validate_generation_accepts_only_unique_ok_rows(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    _write_jsonl(path, [{"sample_id": "a", "status": "ok"}, {"sample_id": "b", "status": "ok"}])
    assert validate_generation_artifact(path, expected_samples=2) == {
        "examples": 2,
        "unique_sample_ids": 2,
        "status": "ok",
    }


def test_validate_generation_rejects_failure_duplicate_and_wrong_count(
    tmp_path: Path,
) -> None:
    path = tmp_path / "results.jsonl"
    _write_jsonl(path, [{"sample_id": "a", "status": "failed"}])
    with pytest.raises(ValueError, match="non-ok status"):
        validate_generation_artifact(path, expected_samples=1)

    _write_jsonl(path, [{"sample_id": "a", "status": "ok"}, {"sample_id": "a", "status": "ok"}])
    with pytest.raises(ValueError, match="Duplicate generation sample IDs"):
        validate_generation_artifact(path, expected_samples=2)

    _write_jsonl(path, [{"sample_id": "a", "status": "ok"}])
    with pytest.raises(ValueError, match="expected 2.*found 1"):
        validate_generation_artifact(path, expected_samples=2)
