from __future__ import annotations

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

import longcoqa.cli as cli_module
from longcoqa.cli import app
from longcoqa.evaluation.paper_metric_profiles import load_metric_profiles
from longcoqa.evaluation.paper_metrics import BatchMetricOutput


def test_metric_profile_config_loads_locked_revisions(tmp_path: Path) -> None:
    config = tmp_path / "metrics.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "profiles": [
                    {
                        "profile_id": "bertscore-locked-v1",
                        "metrics": ["bertscore_f1"],
                        "compatibility": "paper_named_locked",
                        "implementation": "bert-score",
                        "normalization": "none",
                        "tokenization": "model_tokenizer",
                        "model_id": "example/bert",
                        "model_revision": "abcdef",
                        "options": {"max_length": 512},
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    profiles = load_metric_profiles(config)

    assert profiles["bertscore-locked-v1"].model_revision == "abcdef"


def test_evaluate_paper_metrics_cli_runs_lexical_profile(tmp_path: Path) -> None:
    input_path = tmp_path / "results.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "sample_id": "factoid:1:1",
                "question_type": "factoid",
                "dialogue_id": 1,
                "turn": 1,
                "prediction": "答案",
                "reference": "答案",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    config = tmp_path / "metrics.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "profiles": [
                    {
                        "profile_id": "lexical-v1",
                        "metrics": ["exact_match", "recall"],
                        "compatibility": "paper_named_locked",
                        "implementation": "longcoqa.evaluation.paper_metrics",
                        "normalization": "zh_alnum_lower",
                        "tokenization": "character",
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "paper-reports"

    result = CliRunner().invoke(
        app,
        [
            "evaluate-paper-metrics",
            "--input",
            str(input_path),
            "--config",
            str(config),
            "--output-root",
            str(output_root),
            "--profile",
            "lexical-v1",
            "--bootstrap-iterations",
            "10",
        ],
    )

    assert result.exit_code == 0, result.output
    report = json.loads((output_root / "lexical-v1" / "report.json").read_text())
    assert report["successful_samples"] == 1
    assert report["overall"]["metrics"]["exact_match"] == 1.0
    assert (output_root / "lexical-v1" / "scored_results.jsonl").is_file()
    assert (output_root / "lexical-v1" / "paper_table.csv").is_file()
    assert (output_root / "lexical-v1" / "paper_table.md").is_file()
    assert (output_root / "lexical-v1" / "paper_table.metadata.json").is_file()


def test_evaluate_paper_metrics_cli_keeps_legacy_reports_untouched(tmp_path: Path) -> None:
    legacy = tmp_path / "automatic.json"
    legacy.write_text("legacy", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "evaluate-paper-metrics",
            "--input",
            str(tmp_path / "missing.jsonl"),
            "--config",
            str(tmp_path / "missing.yaml"),
            "--output-root",
            str(tmp_path / "new-reports"),
        ],
    )

    assert result.exit_code != 0
    assert legacy.read_text(encoding="utf-8") == "legacy"


def test_evaluate_paper_metrics_cli_runs_learned_profile(
    tmp_path: Path, monkeypatch
) -> None:
    input_path = tmp_path / "results.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "sample_id": "factoid:1:1",
                "question_type": "factoid",
                "dialogue_id": 1,
                "turn": 1,
                "prediction": "answer",
                "reference": "reference",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = tmp_path / "metrics.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "profiles": [
                    {
                        "profile_id": "bleurt-v1",
                        "metrics": ["bleurt"],
                        "compatibility": "paper_named_locked",
                        "implementation": "transformers",
                        "normalization": "none",
                        "tokenization": "model_tokenizer",
                        "model_id": "example/bleurt",
                        "model_revision": "abcdef",
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    class FakeAdapter:
        def score_batch(self, predictions, references):
            return BatchMetricOutput(scores=({"bleurt": 0.75},), truncated=(False,))

    monkeypatch.setattr(cli_module, "build_learned_adapter", lambda *args, **kwargs: FakeAdapter())
    output_root = tmp_path / "reports"
    result = CliRunner().invoke(
        app,
        [
            "evaluate-paper-metrics",
            "--input",
            str(input_path),
            "--config",
            str(config),
            "--output-root",
            str(output_root),
            "--profile",
            "bleurt-v1",
            "--bootstrap-iterations",
            "10",
        ],
    )

    assert result.exit_code == 0, result.output
    report = json.loads((output_root / "bleurt-v1" / "report.json").read_text())
    assert report["overall"]["metrics"]["bleurt"] == 0.75
