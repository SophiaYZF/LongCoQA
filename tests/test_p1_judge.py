from __future__ import annotations

import pytest

from longcoqa.evaluation.p1_judge import (
    expected_matrix_totals,
    normalize_p1_inputs,
    select_p1_judge_inputs,
    validate_p1_inputs,
    validate_p1_judgments,
)


def test_expected_matrix_totals_support_cpt_and_original_matrix() -> None:
    assert expected_matrix_totals(1, ("conversation", "grounded", "raft")) == {
        "successful_samples": 1680,
        "answer_quality_samples": 1625,
        "refusal_quality_samples": 55,
    }
    assert expected_matrix_totals(4, ("conversation", "grounded", "raft")) == {
        "successful_samples": 6720,
        "answer_quality_samples": 6500,
        "refusal_quality_samples": 220,
    }


def _row(sample_id: str, *, answerable: bool = True) -> dict[str, object]:
    return {
        "status": "ok",
        "sample_id": sample_id,
        "question_type": "factoid",
        "turn": 1,
        "model": "Qwen3.5-9B",
        "model_variant": "raft",
        "eval_variant": "raft",
        "reference": "参考答案",
        "prediction": "带引用的预测",
        "answer_text": "纯答案",
        "answerable": answerable,
        "messages": [{"role": "user", "content": "证据"}],
    }


def test_normalize_p1_inputs_restores_question_and_routes_refusal() -> None:
    rows = [_row("factoid:1:1"), _row("factoid:1:2", answerable=False)]

    normalized = normalize_p1_inputs(
        rows,
        question_index={"factoid:1:1": "问题一", "factoid:1:2": "问题二"},
    )

    assert normalized[0]["question"] == "问题一"
    assert normalized[0]["prediction"] == "纯答案"
    assert normalized[0]["judgment_type"] == "answer_quality"
    assert normalized[1]["judgment_type"] == "refusal_quality"


def test_select_p1_judge_inputs_preflight_covers_both_rubrics() -> None:
    rows = [
        {**_row("factoid:1:1"), "judgment_type": "answer_quality"},
        {**_row("factoid:1:2"), "judgment_type": "answer_quality"},
        {
            **_row("factoid:1:3", answerable=False),
            "judgment_type": "refusal_quality",
        },
    ]

    selected = select_p1_judge_inputs(rows, 2)

    assert {row["judgment_type"] for row in selected} == {
        "answer_quality",
        "refusal_quality",
    }


def test_validate_p1_inputs_enforces_count_and_unique_ids() -> None:
    rows = [_row("factoid:1:1"), _row("factoid:1:1")]
    with pytest.raises(ValueError, match="Duplicate sample IDs"):
        validate_p1_inputs(rows, expected_samples=2)
    with pytest.raises(ValueError, match="Expected 3 P1 inputs"):
        validate_p1_inputs(rows[:1], expected_samples=3)


def test_validate_p1_judgments_checks_model_and_rubric_counts() -> None:
    results = [
        {
            "status": "ok",
            "sample_id": "factoid:1:1",
            "judge_model": "judge-model",
            "judgment_type": "answer_quality",
        },
        {
            "status": "ok",
            "sample_id": "factoid:1:2",
            "judge_model": "judge-model",
            "judgment_type": "refusal_quality",
        },
    ]
    counts = validate_p1_judgments(
        results,
        expected_samples=2,
        expected_judge_model="judge-model",
        expected_answer_quality=1,
        expected_refusal_quality=1,
    )
    assert counts == {"answer_quality": 1, "refusal_quality": 1}

    results[1]["judge_model"] = "wrong-model"
    with pytest.raises(ValueError, match="Unexpected judge model"):
        validate_p1_judgments(
            results,
            expected_samples=2,
            expected_judge_model="judge-model",
            expected_answer_quality=1,
            expected_refusal_quality=1,
        )
