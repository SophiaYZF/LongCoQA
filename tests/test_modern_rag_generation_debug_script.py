import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts/run_modern_rag_generation_debug_2gpu.sh"


def test_debug_script_rejects_unknown_method_before_gpu_checks() -> None:
    env = os.environ.copy()
    env["LONGCOQA_MODERN_RAG_DEBUG_RUNS"] = "unknown"
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert "LONGCOQA_MODERN_RAG_DEBUG_RUNS" in result.stderr
    assert "Expected exactly" not in result.stderr


def test_debug_script_is_two_lane_and_smoke_isolated() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "runtime/modern_rag_debug" in text
    assert "CUDA_VISIBLE_DEVICES=$gpu" in text
    assert "--shard-count 2" in text
    assert "--expected-samples \"$expected\"" in text
    assert "outputs/rag/" not in text
    assert "runtime/dlc/.modern_rag" not in text
