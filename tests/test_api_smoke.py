from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from longcoqa.evaluation.api_smoke import (
    run_api_smoke,
    select_stratified_smoke12,
    turn_bucket,
    validate_api_smoke_request,
)


def fixture_rows() -> list[dict[str, object]]:
    rows = []
    dialogue_id = 1
    for question_type in ("factoid", "non_factoid"):
        for turn in (1, 3, 5):
            for index in range(4):
                rows.append(
                    {
                        "status": "ok",
                        "sample_id": f"{question_type}:{dialogue_id}:{index + 1}",
                        "question_type": question_type,
                        "dialogue_id": dialogue_id,
                        "turn": turn,
                        "prediction": f"answer-{index}",
                        "reference": f"reference-{index}",
                    }
                )
            dialogue_id += 1
    return rows


def test_turn_bucket_contract() -> None:
    assert turn_bucket(1) == "turn_1"
    assert turn_bucket(2) == "turn_2_4"
    assert turn_bucket(4) == "turn_2_4"
    assert turn_bucket(5) == "turn_5_plus"
    with pytest.raises(ValueError, match="positive"):
        turn_bucket(0)


def test_stratified_smoke_selects_two_stable_rows_per_stratum() -> None:
    rows = fixture_rows()
    shuffled = list(rows)
    random.Random(99).shuffle(shuffled)

    left = select_stratified_smoke12(rows)
    right = select_stratified_smoke12(shuffled)

    assert [row["sample_id"] for row in left] == [row["sample_id"] for row in right]
    assert len(left) == 12
    counts: dict[tuple[str, str], int] = {}
    for row in left:
        key = (str(row["question_type"]), turn_bucket(int(row["turn"])))
        counts[key] = counts.get(key, 0) + 1
    assert set(counts.values()) == {2}
    assert len(counts) == 6


def test_stratified_smoke_refuses_missing_or_small_stratum() -> None:
    rows = fixture_rows()
    missing = [row for row in rows if not (row["question_type"] == "factoid" and row["turn"] == 1)]
    with pytest.raises(ValueError, match="factoid:turn_1"):
        select_stratified_smoke12(missing)
    small = [
        row
        for row in rows
        if not (row["question_type"] == "factoid" and row["turn"] == 1)
    ] + [next(row for row in rows if row["question_type"] == "factoid" and row["turn"] == 1)]
    with pytest.raises(ValueError, match="at least 2"):
        select_stratified_smoke12(small)


def test_validate_api_smoke_request_guards_scope_and_output(tmp_path: Path) -> None:
    root = tmp_path / "runtime" / "paper_metrics_api_smoke"
    validate_api_smoke_request(smoke=True, limit=12, cells=("model/protocol",), output_root=root)
    with pytest.raises(ValueError, match="--smoke"):
        validate_api_smoke_request(
            smoke=False, limit=12, cells=("model/protocol",), output_root=root
        )
    with pytest.raises(ValueError, match="12"):
        validate_api_smoke_request(
            smoke=True, limit=13, cells=("model/protocol",), output_root=root
        )
    with pytest.raises(ValueError, match="one cell"):
        validate_api_smoke_request(
            smoke=True, limit=12, cells=("a", "b"), output_root=root
        )
    with pytest.raises(ValueError, match="runtime smoke root"):
        validate_api_smoke_request(
            smoke=True, limit=12, cells=("a",), output_root=tmp_path / "formal"
        )


def test_run_api_smoke_reuses_compatible_checkpoints(tmp_path: Path) -> None:
    calls = 0

    def evaluator(row: dict[str, object]) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"claims": [str(row["prediction"])], "supported": True}

    output_root = tmp_path / "runtime" / "paper_metrics_api_smoke"
    first = run_api_smoke(
        fixture_rows(),
        cell="model/protocol",
        output_root=output_root,
        evaluator=evaluator,
        smoke=True,
        limit=12,
        evaluator_identity={"provider": "fake", "model": "judge-v1"},
    )
    second = run_api_smoke(
        fixture_rows(),
        cell="model/protocol",
        output_root=output_root,
        evaluator=evaluator,
        smoke=True,
        limit=12,
        evaluator_identity={"provider": "fake", "model": "judge-v1"},
    )

    assert calls == 12
    assert first == second
    assert first["successful_samples"] == 12
    assert first["failed_samples"] == 0
    assert not any(path.name == ".complete" for path in output_root.rglob("*"))
    cell_root = output_root / "model__protocol"
    assert len(list((cell_root / "checkpoints").glob("*.json"))) == 12
    assert len((cell_root / "results.jsonl").read_text().splitlines()) == 12


def test_run_api_smoke_recomputes_when_evaluator_identity_changes(tmp_path: Path) -> None:
    calls = 0

    def evaluator(_: dict[str, object]) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"ok": True}

    output_root = tmp_path / "runtime" / "paper_metrics_api_smoke"
    for model in ("judge-v1", "judge-v2"):
        run_api_smoke(
            fixture_rows(),
            cell="model/protocol",
            output_root=output_root,
            evaluator=evaluator,
            smoke=True,
            limit=12,
            evaluator_identity={"provider": "fake", "model": model},
        )
    assert calls == 24


def test_run_api_smoke_checkpoints_structured_failure(tmp_path: Path) -> None:
    def evaluator(row: dict[str, object]) -> dict[str, object]:
        if str(row["sample_id"]).startswith("factoid"):
            raise TimeoutError("timed out")
        return {"ok": True}

    report = run_api_smoke(
        fixture_rows(),
        cell="model/protocol",
        output_root=tmp_path / "runtime" / "paper_metrics_api_smoke",
        evaluator=evaluator,
        smoke=True,
        limit=12,
        evaluator_identity={"provider": "fake", "model": "judge-v1"},
    )

    assert report["failed_samples"] == 6
    failure_path = next(
        path
        for path in (tmp_path / "runtime" / "paper_metrics_api_smoke").rglob("*.json")
        if path.parent.name == "checkpoints"
        and json.loads(path.read_text())["status"] == "error"
    )
    failure = json.loads(failure_path.read_text())
    assert failure["error_type"] == "TimeoutError"
