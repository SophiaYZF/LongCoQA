import json
from pathlib import Path

import pytest

import longcoqa.rag.context as context_module
from longcoqa.rag.context import (
    build_rag_contexts,
    normalize_ranking_row,
    select_parent_contexts,
    select_ranked_parent_contexts,
)
from longcoqa.utils.io import read_jsonl

SAMPLES = [
    {
        "sample_id": "factoid:1:1",
        "question_type": "factoid",
        "dialogue_id": 1,
        "turn": 1,
        "question": "q1",
        "answer": "a1",
        "evidence_ids": [1],
    },
    {
        "sample_id": "factoid:1:2",
        "question_type": "factoid",
        "dialogue_id": 1,
        "turn": 2,
        "question": "q2",
        "answer": "a2",
        "evidence_ids": [],
    },
]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _context_assets(tmp_path: Path) -> tuple[Path, Path]:
    children = tmp_path / "children.jsonl"
    parents = tmp_path / "parents.jsonl"
    _write_jsonl(children, [{"child_id": "c1", "parent_id": "p1"}])
    _write_jsonl(
        parents,
        [
            {
                "parent_id": "p1",
                "book_id": 1,
                "chapter_id": "b01-c001",
                "book_title": "book",
                "chapter_title": "chapter",
                "text": "context",
            }
        ],
    )
    return children, parents


def _patch_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeDataset:
        def iter_samples(self):
            return iter(SAMPLES)

    monkeypatch.setattr(context_module, "LongCoQADataset", FakeDataset)


def test_parent_contexts_deduplicate_and_order() -> None:
    ranked = [
        {"child_id": "c2", "score": 2.0},
        {"child_id": "c1b", "score": 1.5},
        {"child_id": "c1", "score": 1.0},
    ]
    children = {
        "c2": {"parent_id": "p2"},
        "c1b": {"parent_id": "p1"},
        "c1": {"parent_id": "p1"},
    }
    parents = {
        "p1": {"parent_id": "p1", "book_id": 1, "chapter_id": "b01-c001"},
        "p2": {"parent_id": "p2", "book_id": 2, "chapter_id": "b02-c001"},
    }
    values = select_parent_contexts(
        ranked,
        children=children,
        parents=parents,
        top_k=2,
        chronological=True,
    )
    assert [row["parent_id"] for row in values] == ["p1", "p2"]
    assert values[0]["matched_child_id"] == "c1b"


def test_normalize_hipporag_parent_ranking() -> None:
    row = normalize_ranking_row(
        {
            "id": "factoid:1:1",
            "question": "q",
            "ranked": [
                {"parent_id": "p2", "score": 0.8},
                {"parent_id": "p2", "score": 0.7},
                {"parent_id": "p1", "score": 0.6},
            ],
        },
        "parent",
    )
    assert row["sample_id"] == "factoid:1:1"
    assert row["query"] == "q"
    assert [item["parent_id"] for item in row["ranked"]] == ["p2", "p2", "p1"]
    assert [item["score"] for item in row["ranked"]] == [0.8, 0.7, 0.6]


def test_parent_ranking_deduplicates_without_child_lookup() -> None:
    parents = {
        "p1": {"parent_id": "p1", "book_id": 1, "chapter_id": "b01-c001"},
        "p2": {"parent_id": "p2", "book_id": 2, "chapter_id": "b02-c001"},
    }
    values = select_ranked_parent_contexts(
        [
            {"parent_id": "p2", "score": 2.0},
            {"parent_id": "p2", "score": 1.0},
            {"parent_id": "p1", "score": 0.5},
        ],
        ranking_level="parent",
        children={},
        parents=parents,
        top_k=2,
        chronological=False,
    )
    assert [row["parent_id"] for row in values] == ["p2", "p1"]
    assert values[0]["matched_child_id"] is None


def test_normalize_ranking_rejects_unknown_level() -> None:
    try:
        normalize_ranking_row(
            {"sample_id": "factoid:1:1", "query": "q", "ranked": []},
            "paragraph",
        )
    except ValueError as error:
        assert "ranking_level" in str(error)
    else:
        raise AssertionError("unknown ranking level should fail")


def test_build_parent_contexts_includes_rows_without_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_samples(monkeypatch)
    children, parents = _context_assets(tmp_path)
    retrieval = tmp_path / "retrieval.jsonl"
    output = tmp_path / "contexts.jsonl"
    _write_jsonl(
        retrieval,
        [
            {
                "id": sample["sample_id"],
                "question": sample["question"],
                "ranked": [{"parent_id": "p1", "score": 1.0}],
            }
            for sample in SAMPLES
        ],
    )

    report = build_rag_contexts(
        retrieval_path=retrieval,
        children_path=children,
        parents_path=parents,
        output_path=output,
        top_k=10,
        chronological=False,
        ranking_level="parent",
        expected_samples=2,
    )

    rows = read_jsonl(output)
    assert [row["sample_id"] for row in rows] == ["factoid:1:1", "factoid:1:2"]
    assert rows[1]["evidence_ids"] == []
    assert rows[0]["retrieved_child_ids"] == []
    assert report["ranking_level"] == "parent"
    assert report["expected_samples"] == 2


def test_build_contexts_rejects_duplicate_retrieval_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_samples(monkeypatch)
    children, parents = _context_assets(tmp_path)
    retrieval = tmp_path / "retrieval.jsonl"
    row = {"id": "factoid:1:1", "question": "q", "ranked": [{"parent_id": "p1", "score": 1.0}]}
    _write_jsonl(retrieval, [row, row])

    with pytest.raises(ValueError, match="Duplicate retrieval sample IDs"):
        build_rag_contexts(
            retrieval_path=retrieval,
            children_path=children,
            parents_path=parents,
            output_path=tmp_path / "contexts.jsonl",
            ranking_level="parent",
            expected_samples=2,
        )


def test_build_contexts_rejects_wrong_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_samples(monkeypatch)
    children, parents = _context_assets(tmp_path)
    retrieval = tmp_path / "retrieval.jsonl"
    _write_jsonl(
        retrieval,
        [{"id": "factoid:1:1", "question": "q", "ranked": [{"parent_id": "p1", "score": 1.0}]}],
    )

    with pytest.raises(ValueError, match="Expected 2 retrieval samples, found 1"):
        build_rag_contexts(
            retrieval_path=retrieval,
            children_path=children,
            parents_path=parents,
            output_path=tmp_path / "contexts.jsonl",
            ranking_level="parent",
            expected_samples=2,
        )


def test_build_contexts_rejects_unknown_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_samples(monkeypatch)
    children, parents = _context_assets(tmp_path)
    retrieval = tmp_path / "retrieval.jsonl"
    _write_jsonl(
        retrieval,
        [
            {"id": sample["sample_id"], "question": "q", "ranked": [{"parent_id": "missing", "score": 1.0}]}
            for sample in SAMPLES
        ],
    )

    with pytest.raises(ValueError, match="Unknown parent_id: missing"):
        build_rag_contexts(
            retrieval_path=retrieval,
            children_path=children,
            parents_path=parents,
            output_path=tmp_path / "contexts.jsonl",
            ranking_level="parent",
            expected_samples=2,
        )
