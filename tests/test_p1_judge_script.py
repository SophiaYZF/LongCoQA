from longcoqa.paths import PROJECT_ROOT


def test_p1_judge_script_has_formal_guard_and_isolated_preflight() -> None:
    script = (PROJECT_ROOT / "scripts/run_p1_judge.sh").read_text(encoding="utf-8")

    assert "set -euo pipefail" in script
    assert "outputs/formal/p1/.complete" in script
    assert "DASHSCOPE_API_KEYS" in script
    assert "DASHSCOPE_BASE_URL" in script
    assert "DASHSCOPE_MODEL_ID" in script
    assert "runtime/p1_judge_preflight" in script
    assert "LONGCOQA_P1_JUDGE_MODEL_VARIANT" in script
    assert "LONGCOQA_P1_JUDGE_EVAL_VARIANT" in script


def test_phase_f_declares_full_p1_judge_matrix() -> None:
    config = (PROJECT_ROOT / "configs/experiments/phase_f.yaml").read_text(
        encoding="utf-8"
    )
    assert "p1_judge:" in config
    assert "provider: qwen37" in config
    assert "expected_samples:" in config
    assert "model_variants:" in config
    assert "eval_variants:" in config
