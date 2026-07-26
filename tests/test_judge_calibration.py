from __future__ import annotations

import csv
import json
import stat
from pathlib import Path

import pytest
from typer.testing import CliRunner

import longcoqa.cli as cli_module
from longcoqa.cli import app
from longcoqa.evaluation.judge_calibration import (
    JudgeProfile,
    calibrate_judges,
    export_human_annotation,
    import_human_annotations,
    judge_messages,
    load_judge_profiles,
    make_judge_evaluator,
    parse_judge_score,
)
from longcoqa.evaluation.paper_metric_profiles import MetricCompatibility


def profile(**overrides: object) -> JudgeProfile:
    values: dict[str, object] = {
        "profile_id": "paper-llm100-v1",
        "compatibility": MetricCompatibility.APPROXIMATE,
        "scale_min": 0.0,
        "scale_max": 100.0,
        "score_field": "score",
        "system_prompt": "Judge the answer and return JSON.",
        "transform": "identity",
        "scorer_model": "judge-model-v1",
    }
    values.update(overrides)
    return JudgeProfile(**values)


def test_judge_profile_is_locked_and_hashes_prompt() -> None:
    left = profile()
    right = profile()

    assert left.profile_hash == right.profile_hash
    assert left.to_dict()["compatibility"] == "approximate"
    assert left.prompt_hash
    with pytest.raises(ValueError, match="scale"):
        profile(scale_min=3, scale_max=3)
    with pytest.raises(ValueError, match="scorer_model"):
        profile(scorer_model="")


def test_parse_judge_score_supports_paper_and_qwen_transforms() -> None:
    llm100 = parse_judge_score('{"score": 73}', profile())
    qwen = parse_judge_score(
        '{"correctness": 4}',
        profile(
            profile_id="qwen-judge-v1",
            scale_min=1,
            scale_max=5,
            score_field="correctness",
            transform="linear_1_5_to_0_100",
        ),
    )

    assert llm100 == {"raw_score": 73.0, "score_100": 73.0}
    assert qwen == {"raw_score": 4.0, "score_100": 75.0}
    with pytest.raises(ValueError, match="range"):
        parse_judge_score('{"score": 101}', profile())


def test_locked_config_contains_all_judge_families() -> None:
    profiles = load_judge_profiles(Path("configs/metrics/judge_profiles.yaml"))

    assert set(profiles) == {
        "paper-llm3-v1",
        "paper-llm100-v1",
        "qwen-structured-v1",
        "cross-family-v1",
    }
    assert profiles["paper-llm3-v1"].compatibility is MetricCompatibility.APPROXIMATE
    assert profiles["paper-llm3-v1"].scale_max == 3


def test_judge_messages_include_profile_prompt_and_visible_fields() -> None:
    messages = judge_messages(
        {
            "question_type": "factoid",
            "question": "问题",
            "reference": "参考",
            "prediction": "候选",
            "evidence_text": "证据",
        },
        profile(),
    )

    assert messages[0]["content"] == profile().system_prompt
    assert "问题" in messages[1]["content"]
    assert "参考" in messages[1]["content"]
    assert "候选" in messages[1]["content"]


def test_make_judge_evaluator_uses_locked_profile() -> None:
    calls: list[dict[str, object]] = []

    class FakeClient:
        def generate(self, messages, **kwargs):
            calls.append({"messages": messages, **kwargs})
            return type("Response", (), {"text": '{"score": 75}'})()

    evaluator = make_judge_evaluator(FakeClient(), profile(), max_tokens=128)
    result = evaluator(annotation_rows()[0])

    assert result["score_100"] == 75
    assert calls[0]["temperature"] == 0.0
    assert calls[0]["max_tokens"] == 128


def annotation_rows() -> list[dict[str, object]]:
    return [
        {
            "sample_id": "factoid:1:1",
            "question_type": "factoid",
            "question": "问题1",
            "reference": "参考1",
            "prediction": "候选1",
            "evidence_text": "证据1",
            "model_variant": "secret-model",
            "method": "secret-method",
        },
        {
            "sample_id": "non_factoid:2:1",
            "question_type": "non_factoid",
            "question": "问题2",
            "reference": "参考2",
            "prediction": "候选2",
            "evidence_text": "证据2",
            "model_variant": "secret-model-2",
            "method": "secret-method-2",
        },
    ]


def test_human_export_is_blinded_and_has_private_key(tmp_path: Path) -> None:
    public_path = tmp_path / "human.csv"
    key_path = tmp_path / "human.private.json"

    export_human_annotation(
        annotation_rows(), public_path=public_path, key_path=key_path, blind_salt="test-salt"
    )

    public_text = public_path.read_text(encoding="utf-8")
    assert "secret-model" not in public_text
    assert "secret-method" not in public_text
    with public_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["presentation_id"]
    assert rows[0]["correctness"] == ""
    assert rows[0]["faithfulness"] == ""
    private = json.loads(key_path.read_text(encoding="utf-8"))
    assert private["rows"][0]["sample_id"] == "factoid:1:1"
    assert private["rows"][0]["model_variant"] == "secret-model"
    assert stat.S_IMODE(key_path.stat().st_mode) == 0o600


def write_annotations(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "presentation_id",
        "annotator_id",
        "question_type",
        "correctness",
        "completeness",
        "faithfulness",
        "citation_correctness",
        "pairwise_choice",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_human_import_validates_schema_labels_and_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "annotations.csv"
    valid = {
        "presentation_id": "p1",
        "annotator_id": "a1",
        "question_type": "factoid",
        "correctness": 5,
        "completeness": 4,
        "faithfulness": 5,
        "citation_correctness": 4,
        "pairwise_choice": "A",
    }
    write_annotations(path, [valid])
    assert import_human_annotations(path)[0]["correctness"] == 5
    write_annotations(path, [valid, valid])
    with pytest.raises(ValueError, match="Duplicate"):
        import_human_annotations(path)
    write_annotations(path, [{**valid, "correctness": 6}])
    with pytest.raises(ValueError, match="correctness"):
        import_human_annotations(path)
    write_annotations(path, [{**valid, "question_type": "other"}])
    with pytest.raises(ValueError, match="question_type"):
        import_human_annotations(path)
    write_annotations(path, [{**valid, "pairwise_choice": "C"}])
    with pytest.raises(ValueError, match="pairwise_choice"):
        import_human_annotations(path)


def test_calibration_reports_correlation_pairwise_accuracy_and_kappa() -> None:
    humans = []
    for annotator in ("a1", "a2"):
        humans.extend(
            [
                {
                    "presentation_id": "p1",
                    "annotator_id": annotator,
                    "question_type": "factoid",
                    "correctness": 5,
                    "pairwise_choice": "A",
                },
                {
                    "presentation_id": "p2",
                    "annotator_id": annotator,
                    "question_type": "non_factoid",
                    "correctness": 1,
                    "pairwise_choice": "B",
                },
            ]
        )
    judge = [
        {"presentation_id": "p1", "score_100": 100.0, "pairwise_choice": "A"},
        {"presentation_id": "p2", "score_100": 0.0, "pairwise_choice": "B"},
    ]

    report = calibrate_judges(judge, humans)

    assert report["samples"] == 2
    assert report["pearson"] == 1.0
    assert report["spearman"] == 1.0
    assert report["pairwise_accuracy"] == 1.0
    assert report["inter_annotator"]["cohen_kappa"] == 1.0
    assert report["by_question_type"]["factoid"]["exact_agreement"] == 1.0


def test_calibration_marks_zero_variance_correlation_unavailable() -> None:
    humans = [
        {
            "presentation_id": "p1",
            "annotator_id": "a1",
            "question_type": "factoid",
            "correctness": 3,
            "pairwise_choice": "",
        },
        {
            "presentation_id": "p2",
            "annotator_id": "a1",
            "question_type": "factoid",
            "correctness": 3,
            "pairwise_choice": "",
        },
    ]
    judge = [
        {"presentation_id": "p1", "score_100": 50.0},
        {"presentation_id": "p2", "score_100": 50.0},
    ]

    report = calibrate_judges(judge, humans)

    assert report["pearson"] is None
    assert report["correlation_unavailable_reason"] == "zero_variance"


def write_smoke_input(path: Path) -> None:
    rows = []
    dialogue = 1
    for question_type in ("factoid", "non_factoid"):
        for turn in (1, 3, 5):
            for index in range(2):
                row = annotation_rows()[index]
                row.update(
                    {
                        "sample_id": f"{question_type}:{dialogue}:{index + 1}",
                        "question_type": question_type,
                        "dialogue_id": dialogue,
                        "turn": turn,
                        "status": "ok",
                    }
                )
                rows.append(row)
            dialogue += 1
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_judge_cli_refuses_before_provider_construction(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "input.jsonl"
    write_smoke_input(input_path)
    monkeypatch.setattr(
        cli_module,
        "load_provider_settings",
        lambda *_: pytest.fail("provider constructed before smoke validation"),
    )

    result = CliRunner().invoke(
        app,
        [
            "evaluate-judge-smoke",
            "--input",
            str(input_path),
            "--provider",
            "fake",
            "--profile",
            "paper-llm100-v1",
            "--cell",
            "model/protocol",
            "--output-root",
            str(tmp_path / "runtime" / "paper_metrics_api_smoke"),
        ],
    )

    assert result.exit_code != 0
    assert "--smoke" in str(result.exception)


def test_judge_cli_runs_exactly_twelve_rows(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "input.jsonl"
    write_smoke_input(input_path)
    calls = 0

    class FakeSettings:
        model = "judge-v1"

        def safe_summary(self):
            return {"name": "fake", "model": self.model}

    class Context:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class FakeClient(Context):
        def generate(self, *_args, **_kwargs):
            nonlocal calls
            calls += 1
            return type("Response", (), {"text": '{"score": 75}'})()

    monkeypatch.setattr(cli_module, "load_provider_settings", lambda *_: FakeSettings())
    monkeypatch.setattr(cli_module, "ResponseCache", Context)
    monkeypatch.setattr(cli_module, "ProviderClient", FakeClient)
    output_root = tmp_path / "runtime" / "paper_metrics_api_smoke"

    result = CliRunner().invoke(
        app,
        [
            "evaluate-judge-smoke",
            "--input",
            str(input_path),
            "--provider",
            "fake",
            "--profile",
            "paper-llm100-v1",
            "--cell",
            "model/protocol",
            "--output-root",
            str(output_root),
            "--smoke",
            "--limit",
            "12",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == 12
    assert not any(path.name == ".complete" for path in output_root.rglob("*"))


def test_human_cli_exports_imports_and_calibrates(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in annotation_rows()),
        encoding="utf-8",
    )
    public = tmp_path / "human.csv"
    private = tmp_path / "human.private.json"
    export_result = CliRunner().invoke(
        app,
        [
            "export-human-annotation",
            "--input",
            str(source),
            "--public-output",
            str(public),
            "--private-key-output",
            str(private),
            "--blind-salt",
            "test-salt",
        ],
    )
    assert export_result.exit_code == 0, export_result.output

    with public.open(encoding="utf-8", newline="") as handle:
        public_rows = list(csv.DictReader(handle))
    for index, row in enumerate(public_rows):
        row.update(
            {
                "annotator_id": "a1",
                "correctness": 5 - index * 4,
                "completeness": 5 - index * 4,
                "faithfulness": 5 - index * 4,
                "citation_correctness": 5 - index * 4,
            }
        )
    with public.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=public_rows[0].keys())
        writer.writeheader()
        writer.writerows(public_rows)

    imported = tmp_path / "imported.jsonl"
    import_result = CliRunner().invoke(
        app,
        ["import-human-annotations", "--input", str(public), "--output", str(imported)],
    )
    assert import_result.exit_code == 0, import_result.output

    judge = tmp_path / "judge.jsonl"
    judge.write_text(
        "".join(
            json.dumps(
                {
                    "presentation_id": row["presentation_id"],
                    "score_100": 100 - index * 100,
                }
            )
            + "\n"
            for index, row in enumerate(public_rows)
        ),
        encoding="utf-8",
    )
    report = tmp_path / "calibration.json"
    calibration_result = CliRunner().invoke(
        app,
        [
            "calibrate-judges",
            "--judge-input",
            str(judge),
            "--human-input",
            str(imported),
            "--output",
            str(report),
        ],
    )
    assert calibration_result.exit_code == 0, calibration_result.output
    assert json.loads(report.read_text())["pearson"] == 1.0
