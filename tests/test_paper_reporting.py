from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from longcoqa.evaluation.paper_metric_profiles import (
    MetricCompatibility,
    MetricProfile,
)
from longcoqa.evaluation.paper_metrics import MetricScore, paper_lexical_scores
from longcoqa.evaluation.paper_reporting import (
    aggregate_paper_scores,
    compare_paper_scores,
    export_paper_table,
    score_rows,
)


def profile() -> MetricProfile:
    return MetricProfile(
        profile_id="paper-lexical-v1",
        metrics=("exact_match", "recall"),
        compatibility=MetricCompatibility.PAPER_NAMED_LOCKED,
        implementation="longcoqa.evaluation.paper_metrics",
        normalization="zh_alnum_lower",
        tokenization="character",
    )


def row(
    sample_id: str,
    prediction: str,
    reference: str,
    *,
    turn: int = 1,
    status: str = "ok",
) -> dict[str, object]:
    question_type, dialogue_id, _ = sample_id.split(":")
    return {
        "status": status,
        "sample_id": sample_id,
        "question_type": question_type,
        "dialogue_id": int(dialogue_id),
        "turn": turn,
        "prediction": prediction,
        "reference": reference,
    }


def scored(
    sample_id: str,
    exact_match: float,
    recall: float,
    *,
    turn: int = 1,
    status: str = "ok",
) -> dict[str, object]:
    question_type, dialogue_id, _ = sample_id.split(":")
    return {
        "schema_version": "longcoqa-paper-metric-score-v1",
        "status": status,
        "sample_id": sample_id,
        "question_type": question_type,
        "dialogue_id": int(dialogue_id),
        "turn": turn,
        "profile_id": "paper-lexical-v1",
        "profile_hash": profile().profile_hash,
        "compatibility": "paper_named_locked",
        "scores": {"exact_match": exact_match, "recall": recall},
    }


def test_score_rows_rejects_duplicate_ids_and_failed_inference(tmp_path: Path) -> None:
    duplicate = row("factoid:1:1", "a", "a")
    with pytest.raises(ValueError, match="Duplicate sample ID"):
        score_rows([duplicate, duplicate], profile(), tmp_path)
    with pytest.raises(ValueError, match="non-ok inference"):
        score_rows(
            [row("factoid:1:1", "a", "a", status="failed")],
            profile(),
            tmp_path,
        )


def test_score_rows_reuses_only_compatible_successful_checkpoint(tmp_path: Path) -> None:
    calls = 0

    def scorer(prediction: str, reference: str, metric_profile: MetricProfile) -> MetricScore:
        nonlocal calls
        calls += 1
        return paper_lexical_scores(prediction, reference, metric_profile)

    source = [row("factoid:1:1", "答案", "答案")]
    first = score_rows(source, profile(), tmp_path, scorer=scorer)
    second = score_rows(source, profile(), tmp_path, scorer=scorer)

    assert calls == 1
    assert first == second
    checkpoint = tmp_path / "checkpoints" / "factoid__1__1.json"
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["profile_hash"] == profile().profile_hash
    assert payload["input_hash"]
    assert payload["reference_hash"]

    changed = [row("factoid:1:1", "错误", "答案")]
    score_rows(changed, profile(), tmp_path, scorer=scorer)
    assert calls == 2


def test_score_rows_retries_failed_checkpoint(tmp_path: Path) -> None:
    source = [row("factoid:1:1", "答案", "答案")]

    def failing(*_: object) -> MetricScore:
        raise RuntimeError("scorer unavailable")

    failed = score_rows(source, profile(), tmp_path, scorer=failing)
    recovered = score_rows(source, profile(), tmp_path)

    assert failed[0]["status"] == "error"
    assert failed[0]["error_type"] == "RuntimeError"
    assert recovered[0]["status"] == "ok"


def test_aggregate_paper_scores_stratifies_and_bootstraps_deterministically() -> None:
    rows = [
        scored("factoid:1:1", 1.0, 1.0, turn=1),
        scored("factoid:1:2", 0.0, 0.5, turn=3),
        scored("non_factoid:2:1", 1.0, 1.0, turn=5),
        scored("non_factoid:3:1", 0.0, 0.0, turn=6),
    ]

    first = aggregate_paper_scores(rows, bootstrap_iterations=100, seed=7)
    second = aggregate_paper_scores(rows, bootstrap_iterations=100, seed=7)

    assert first == second
    assert first["requested_samples"] == 4
    assert first["successful_samples"] == 4
    assert first["overall"]["metrics"] == {"exact_match": 0.5, "recall": 0.625}
    assert first["by_question_type"]["factoid"]["samples"] == 2
    assert first["by_turn"]["turn_2_4"]["samples"] == 1
    assert first["bootstrap"]["unit"] == "question_type:dialogue_id"


def test_aggregate_refuses_failures_unless_diagnostic_mode() -> None:
    rows = [scored("factoid:1:1", 1.0, 1.0), scored("factoid:2:1", 0, 0, status="error")]

    with pytest.raises(ValueError, match="failed metric rows"):
        aggregate_paper_scores(rows, bootstrap_iterations=10)
    report = aggregate_paper_scores(rows, bootstrap_iterations=10, allow_failures=True)
    assert report["failed_samples"] == 1
    assert report["successful_samples"] == 1


def test_compare_paper_scores_requires_identical_ids_and_reports_difference() -> None:
    left = [scored("factoid:1:1", 0.0, 0.5), scored("non_factoid:2:1", 0.0, 0.0)]
    right = [scored("factoid:1:1", 1.0, 1.0), scored("non_factoid:2:1", 1.0, 0.5)]

    report = compare_paper_scores(left, right, bootstrap_iterations=20, seed=3)
    assert report["samples"] == 2
    assert report["overall"]["exact_match"]["difference_right_minus_left"] == 1.0
    with pytest.raises(ValueError, match="identical IDs"):
        compare_paper_scores(left, right[:1], bootstrap_iterations=20)


def test_export_paper_table_has_stable_columns_and_metadata(tmp_path: Path) -> None:
    report = aggregate_paper_scores(
        [scored("factoid:1:1", 1.0, 0.75)], bootstrap_iterations=10
    )
    csv_path = tmp_path / "table.csv"
    markdown_path = tmp_path / "table.md"
    metadata_path = tmp_path / "table.metadata.json"

    export_paper_table(
        [{"model": "model-a", "protocol": "Z1", "report": report}],
        metric_order=("exact_match", "recall", "bertscore_f1"),
        csv_path=csv_path,
        markdown_path=markdown_path,
        metadata_path=metadata_path,
        precision=4,
    )

    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == ["model", "protocol", "exact_match", "recall", "bertscore_f1"]
    assert rows[0]["exact_match"] == "1.0000"
    assert rows[0]["bertscore_f1"] == ""
    assert "| model-a | Z1 | 1.0000 | 0.7500 |  |" in markdown_path.read_text(
        encoding="utf-8"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["compatibility"]["exact_match"] == "paper_named_locked"
