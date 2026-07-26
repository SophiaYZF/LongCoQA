import json
from pathlib import Path

import pytest

from longcoqa.evaluation.reporting import (
    bleu_2,
    compare_inference_results,
    evaluate_inference_results,
    extended_score_answer,
)


def _row(sample_id: str, prediction: str, reference: str, turn: int = 1) -> dict:
    question_type, dialogue_id, _ = sample_id.split(":")
    return {
        "status": "ok",
        "sample_id": sample_id,
        "question_type": question_type,
        "dialogue_id": int(dialogue_id),
        "turn": turn,
        "prediction": prediction,
        "reference": reference,
    }


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_extended_metrics() -> None:
    scores = extended_score_answer("答案是小天狼星", "小天狼星")
    assert scores["exact_match"] == 0
    assert scores["exact_match_contain"] == 1
    assert scores["character_recall"] == 1
    assert 0 < bleu_2("小天狼星", "小天狼星") <= 1


def test_evaluate_and_compare(tmp_path: Path) -> None:
    left = [
        _row("factoid:1:1", "错误", "答案"),
        _row("non_factoid:2:1", "部分", "完整答案", turn=5),
    ]
    right = [
        _row("factoid:1:1", "答案", "答案"),
        _row("non_factoid:2:1", "完整答案", "完整答案", turn=5),
    ]
    left_path = tmp_path / "left.jsonl"
    right_path = tmp_path / "right.jsonl"
    _write(left_path, left)
    _write(right_path, right)
    report = evaluate_inference_results(
        right_path, tmp_path / "evaluation.json", bootstrap_iterations=100
    )
    assert report["overall"]["metrics"]["exact_match"] == 1
    assert report["by_turn"]["turn_5_plus"]["samples"] == 1
    comparison = compare_inference_results(
        left_path, right_path, tmp_path / "comparison.json", bootstrap_iterations=100
    )
    assert comparison["overall"]["exact_match"]["difference_right_minus_left"] == 1


def test_compare_requires_identical_ids(tmp_path: Path) -> None:
    left_path = tmp_path / "left.jsonl"
    right_path = tmp_path / "right.jsonl"
    _write(left_path, [_row("factoid:1:1", "a", "a")])
    _write(right_path, [_row("factoid:2:1", "a", "a")])
    with pytest.raises(ValueError, match="identical IDs"):
        compare_inference_results(
            left_path, right_path, tmp_path / "comparison.json", bootstrap_iterations=100
        )
