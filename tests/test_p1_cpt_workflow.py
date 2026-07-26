from longcoqa.paths import PROJECT_ROOT
from longcoqa.utils.io import read_yaml


def test_cpt_phase_config_is_single_variant_and_complete() -> None:
    raw = read_yaml(PROJECT_ROOT / "configs/experiments/phase_f_cpt.yaml")
    p1 = raw["p1_inference"]

    assert p1["model_variants"] == {
        "cpt": {
            "adapter_path": "outputs/training/qwen3_5_9b_cpt_conservative_lora"
        }
    }
    assert {
        name: value["expected_samples"] for name, value in p1["eval_sets"].items()
    } == {
        "conversation": 602,
        "grounded": 539,
        "raft": 539,
    }
    assert raw["p1_judge"]["input_root"] == "outputs/formal/p1_cpt/merged"
    assert raw["p1_judge"]["output_root"] == "outputs/judgments/qwen37/p1_cpt"


def test_cpt_preflight_launcher_is_two_gpu_and_isolated() -> None:
    script = (PROJECT_ROOT / "scripts/run_p1_cpt_preflight_2gpu.sh").read_text(
        encoding="utf-8"
    )

    assert "set -Eeuo pipefail" in script
    assert "LONGCOQA_EXPECTED_GPUS=2" in script
    assert "LONGCOQA_SHARD_COUNT=2" in script
    assert "phase_f_cpt.yaml" in script
    assert "qwen3_5_9b_cpt_conservative_lora" in script
    assert "runtime/p1_cpt_preflight/cpt_raw_conservative_smoke" in script
    assert "summarize_dev_rows" in script
    assert "quality_summary.json" in script
    assert "P1 CPT candidate failed quality gates" in script
    assert script.index("quality_summary.json") < script.index('touch "$output_root/.complete"')
    assert "outputs/formal/p1/.complete" not in script


def test_cpt_judge_launcher_is_independent_and_supports_preflight() -> None:
    script = (PROJECT_ROOT / "scripts/run_p1_cpt_judge.sh").read_text(
        encoding="utf-8"
    )

    assert "outputs/formal/p1_cpt/.complete" in script
    assert "phase_f_cpt.yaml" in script
    assert "outputs/judgments/qwen37/p1_cpt" in script
    assert "runtime/p1_cpt_judge_preflight" in script
    assert "LONGCOQA_P1_CPT_JUDGE_LIMIT" in script
    assert "--model-variant cpt" in script
    assert "outputs/judgments/qwen37/p1/.complete" not in script
