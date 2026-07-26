import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from longcoqa.inference.api_runner import (
    run_api_inference,
    run_api_suite,
    select_dialogue_samples,
    validate_api_protocol_results,
)

PROTOCOLS = (
    "z0_current",
    "z1_teacher_forced",
    "g0_released_gold",
    "g3_aligned_raw",
)


def _dialogue_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for question_type in ("factoid", "non_factoid"):
        for dialogue_id, turns in ((1, 2), (2, 3), (3, 2)):
            for turn in range(1, turns + 1):
                rows.append(
                    {
                        "sample_id": f"{question_type}:{dialogue_id}:{turn}",
                        "question_type": question_type,
                        "dialogue_id": dialogue_id,
                        "turn": turn,
                    }
                )
    return rows


def test_select_dialogue_samples_is_deterministic_and_keeps_complete_dialogues() -> None:
    selected = select_dialogue_samples(
        _dialogue_rows(), dialogues_per_type=2, seed=42
    )
    repeated = select_dialogue_samples(
        list(reversed(_dialogue_rows())), dialogues_per_type=2, seed=42
    )
    assert [row["sample_id"] for row in selected] == [
        row["sample_id"] for row in repeated
    ]
    keys = {
        (str(row["question_type"]), int(row["dialogue_id"]))
        for row in selected
    }
    assert sum(key[0] == "factoid" for key in keys) == 2
    assert sum(key[0] == "non_factoid" for key in keys) == 2
    for key in keys:
        expected = [
            row["sample_id"]
            for row in _dialogue_rows()
            if (row["question_type"], row["dialogue_id"]) == key
        ]
        actual = [
            row["sample_id"]
            for row in selected
            if (row["question_type"], row["dialogue_id"]) == key
        ]
        assert actual == expected


def test_select_dialogue_samples_rejects_invalid_or_unavailable_counts() -> None:
    with pytest.raises(ValueError, match="positive"):
        select_dialogue_samples(_dialogue_rows(), dialogues_per_type=0, seed=42)
    with pytest.raises(ValueError, match="only 3"):
        select_dialogue_samples(_dialogue_rows(), dialogues_per_type=4, seed=42)


def test_run_api_inference_rejects_row_and_dialogue_limits_together() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        run_api_inference(
            dataset=object(),
            settings=object(),
            protocol="z0_current",
            alignment_path=Path("unused"),
            cache_path=Path("unused"),
            output_dir=Path("unused"),
            limit=2,
            dialogues_per_type=1,
        )


def _rows(protocol: str) -> list[dict[str, object]]:
    return [
        {"status": "ok", "sample_id": "factoid:1:1", "protocol": protocol},
        {"status": "ok", "sample_id": "non_factoid:2:1", "protocol": protocol},
    ]


def test_validate_api_protocol_results_accepts_exact_unique_successes() -> None:
    summary = validate_api_protocol_results(
        _rows("z0_current"),
        protocol="z0_current",
        expected_samples=2,
    )

    assert summary == {
        "protocol": "z0_current",
        "expected_samples": 2,
        "result_rows": 2,
        "unique_sample_ids": 2,
        "successful_samples": 2,
        "failed_samples": 0,
        "status": "ok",
    }


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (_rows("z0_current")[:1], "expected 2 rows"),
        (
            [_rows("z0_current")[0], _rows("z0_current")[0]],
            "duplicate sample IDs",
        ),
        (
            [
                _rows("z0_current")[0],
                {
                    "status": "error",
                    "sample_id": "non_factoid:2:1",
                    "protocol": "z0_current",
                },
            ],
            "non-success rows",
        ),
        (_rows("z1_teacher_forced"), "protocol mismatch"),
    ],
)
def test_validate_api_protocol_results_rejects_invalid_formal_artifacts(
    rows: list[dict[str, object]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_api_protocol_results(
            rows,
            protocol="z0_current",
            expected_samples=2,
        )


def test_run_api_suite_uses_protocol_scoped_paths_and_writes_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Path] = []

    def fake_run_api_inference(*_args: object, **kwargs: object) -> dict[str, object]:
        protocol = str(kwargs["protocol"])
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True)
        rows = _rows(protocol)
        (output_dir / "results.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        calls.append(output_dir)
        return {
            "protocol": protocol,
            "requested_samples": 2,
            "successful_samples": 2,
            "failed_samples": 0,
        }

    monkeypatch.setattr(
        "longcoqa.inference.api_runner.run_api_inference",
        fake_run_api_inference,
    )
    output_root = tmp_path / "qwen37"
    summary = run_api_suite(
        dataset=object(),
        settings=SimpleNamespace(name="qwen37", safe_summary=lambda: {"name": "qwen37"}),
        protocols=PROTOCOLS,
        alignment_path=tmp_path / "alignment.jsonl",
        cache_path=tmp_path / "cache.sqlite3",
        output_root=output_root,
        seed=42,
        limit=2,
        temperature=0.0,
        max_tokens_factoid=256,
        max_tokens_non_factoid=768,
    )

    assert calls == [output_root / protocol / "smoke2" for protocol in PROTOCOLS]
    assert summary["status"] == "ok"
    assert summary["samples_per_protocol"] == 2
    assert summary["protocols"] == list(PROTOCOLS)
    assert (output_root / "suite_smoke2.json").is_file()


def test_run_api_suite_labels_dialogue_pilots_by_expanded_turn_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Dataset:
        def iter_samples(self):
            return iter(_dialogue_rows())

    calls: list[Path] = []

    def fake_run_api_inference(*_args: object, **kwargs: object) -> dict[str, object]:
        protocol = str(kwargs["protocol"])
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True)
        selected = select_dialogue_samples(
            _dialogue_rows(), dialogues_per_type=1, seed=42
        )
        rows = [
            {"status": "ok", "sample_id": row["sample_id"], "protocol": protocol}
            for row in selected
        ]
        (output_dir / "results.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        calls.append(output_dir)
        return {"requested_samples": len(rows)}

    monkeypatch.setattr(
        "longcoqa.inference.api_runner.run_api_inference",
        fake_run_api_inference,
    )
    output_root = tmp_path / "pilot"
    summary = run_api_suite(
        Dataset(),
        SimpleNamespace(name="fake", safe_summary=lambda: {"name": "fake"}),
        protocols=("z0_current",),
        alignment_path=tmp_path / "alignment.jsonl",
        cache_path=tmp_path / "cache.sqlite3",
        output_root=output_root,
        seed=42,
        dialogues_per_type=1,
    )

    assert calls == [output_root / "z0_current" / "dialogues1_per_type"]
    assert summary["dialogues_per_type"] == 1
    assert summary["samples_per_protocol"] == len(
        select_dialogue_samples(_dialogue_rows(), dialogues_per_type=1, seed=42)
    )
    assert (output_root / "suite_dialogues1_per_type.json").is_file()
