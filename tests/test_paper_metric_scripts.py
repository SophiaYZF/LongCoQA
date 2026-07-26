from pathlib import Path

import pytest

SCRIPTS = (
    "scripts/evaluate_paper_metrics.sh",
    "scripts/run_p0_rag_formal.sh",
    "scripts/run_rag_quality_smoke.sh",
    "scripts/run_judge_calibration_smoke.sh",
)


@pytest.mark.parametrize("path", SCRIPTS)
def test_paper_metric_scripts_use_strict_shell_and_do_not_mark_complete(path: str) -> None:
    text = Path(path).read_text(encoding="utf-8")

    assert "set -euo pipefail" in text
    assert ".complete" not in text
    assert "outputs/formal" not in text


@pytest.mark.parametrize(
    "path,command",
    [
        ("scripts/run_rag_quality_smoke.sh", "evaluate-rag-quality-smoke"),
        ("scripts/run_judge_calibration_smoke.sh", "evaluate-judge-smoke"),
    ],
)
def test_api_scripts_are_pinned_to_twelve_runtime_smoke_rows(
    path: str, command: str
) -> None:
    text = Path(path).read_text(encoding="utf-8")

    assert command in text
    assert "--smoke" in text
    assert '--limit "12"' in text
    assert "runtime/paper_metrics_api_smoke" in text


def test_local_metric_script_writes_only_versioned_report_root() -> None:
    text = Path("scripts/evaluate_paper_metrics.sh").read_text(encoding="utf-8")

    assert "evaluate-paper-metrics" in text
    assert "reports/generated/evaluation/paper_metrics" in text
    assert "--input" in text
    assert "--output-root" in text
