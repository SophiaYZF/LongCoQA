from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from longcoqa.training.cpt_factorial import summarize_dev_results, summarize_dev_rows

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _row(
    sample_id: str,
    *,
    prediction: str = "答案",
    finish_reason: str = "stop",
    status: str = "ok",
    answerable: bool = True,
    character_f1: float = 0.5,
) -> dict:
    return {
        "sample_id": sample_id,
        "status": status,
        "prediction": prediction,
        "answer_text": prediction.strip(),
        "finish_reason": finish_reason,
        "answerable": answerable,
        "metrics": {"character_f1": character_f1} if answerable else None,
    }


def test_dev_gate_accepts_normal_rows() -> None:
    report = summarize_dev_rows([_row("a"), _row("b", character_f1=1.0)], expected=2)

    assert report["gate_passed"] is True
    assert report["normal_stop_rate"] == 1.0
    assert report["character_f1"] == 0.75


def test_dev_gate_rejects_space_loop() -> None:
    rows = [_row("a", prediction="答案\n" + "　" * 15, finish_reason="length")]

    report = summarize_dev_rows(rows, expected=1)

    assert report["max_u3000_run"] == 15
    assert report["gate_passed"] is False


def test_dev_gate_rejects_stripped_blank_and_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_dev_rows([_row("a"), _row("a")], expected=2)

    report = summarize_dev_rows([_row("a", prediction="　　")], expected=1)
    assert report["blank_answers"] == 1
    assert report["gate_passed"] is False


def test_summarize_dev_results_requires_all_twelve_cells(tmp_path: Path) -> None:
    models = ("A", "B", "C", "D")
    evals = ("conversation", "grounded", "raft")
    for model in models:
        for eval_variant in evals:
            target = tmp_path / model / eval_variant / "full"
            target.mkdir(parents=True)
            target.joinpath("results.jsonl").write_text(
                json.dumps(_row(f"{model}-{eval_variant}"), ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

    report = summarize_dev_results(
        tmp_path,
        expected_per_cell=1,
        model_variants=models,
        eval_variants=evals,
    )

    assert len(report["cells"]) == 12
    assert report["primary_candidate"] == "D"
    assert report["primary_gate_passed"] is True


def test_dev_config_and_launcher_never_reference_test_inputs() -> None:
    config_path = PROJECT_ROOT / "configs" / "experiments" / "phase_f_cpt_factorial_dev.yaml"
    script_path = PROJECT_ROOT / "scripts" / "run_cpt_factorial_dev_2gpu.sh"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    script = script_path.read_text(encoding="utf-8")

    eval_sets = config["p1_inference"]["eval_sets"]
    assert all("_dev.jsonl" in value["path"] for value in eval_sets.values())
    assert config["p1_inference"]["expected_split"] == "dev"
    assert "_test.jsonl" not in script
    assert "LONGCOQA_CPT_FACTORIAL_DEV_LIMIT:-16" in script
    assert "runtime/cpt_factorial_dev" in script
    assert "for marker in" in script
    assert '[[ -e "$marker" ]]' in script
