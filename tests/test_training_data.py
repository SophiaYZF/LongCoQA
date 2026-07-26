from pathlib import Path

import pytest

from longcoqa.training.data import (
    _grounded_output,
    _has_annotated_evidence,
    _negative_candidates,
    build_cpt_dataset,
)


def test_grounded_output_contains_citations() -> None:
    value = _grounded_output("答案", ["1", "2"])
    assert value.startswith("答案")
    assert "EVIDENCE 1" in value
    assert "EVIDENCE 2" in value


def test_negative_candidates_exclude_gold_paragraphs() -> None:
    retrieval = {
        "x": {
            "ranked": [
                {"child_id": "gold"},
                {"child_id": "negative"},
            ]
        }
    }
    children = {
        "gold": {"child_id": "gold", "paragraph_ids": ["p1"], "text": "g"},
        "negative": {
            "child_id": "negative",
            "paragraph_ids": ["p2"],
            "text": "n",
        },
    }
    values = _negative_candidates(
        "x",
        retrieval=retrieval,
        children=children,
        gold_paragraph_ids={"p1"},
        count=4,
    )
    assert [row["child_id"] for row in values] == ["negative"]


def test_missing_evidence_annotation_is_not_an_abstention_label() -> None:
    assert not _has_annotated_evidence({"evidence_ids": []})
    assert not _has_annotated_evidence({})
    assert _has_annotated_evidence({"evidence_ids": [12]})


def test_cpt_builder_fails_before_writing_when_novels_are_absent(tmp_path: Path) -> None:
    output = tmp_path / "output"
    with pytest.raises(FileNotFoundError, match="user-supplied lawful source texts"):
        build_cpt_dataset(novel_dir=tmp_path / "novels", output_dir=output)
    assert not output.exists()
