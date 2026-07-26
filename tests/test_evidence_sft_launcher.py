from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_evidence_sft_matrix_8gpu.sh"


def test_evidence_sft_launcher_has_double_execution_and_human_audit_gate() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert '"--execute"' in text
    assert "LONGCOQA_EVIDENCE_SFT_HUMAN_APPROVED" in text
    assert "READY_FOR_TRAINING" in text
    assert "training_allowed" in text
    assert "pretraining_status.json" in text


def test_evidence_sft_launcher_validates_inputs_and_never_deletes_outputs() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    for required in (
        "phase_g_evidence_sft.yaml",
        "manifest.json",
        "dataset_info.json",
        "quality_report.json",
        "qwen3_5_9b_cpt_conservative_lora/.complete",
        "sha256",
        "llamafactory-cli train",
    ):
        assert required in text
    assert "rm -rf" not in text
    assert "overwrite_output_dir: true" not in text


def test_evidence_sft_launcher_has_valid_shell_syntax() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
