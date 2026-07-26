import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from longcoqa.paths import PROJECT_ROOT


def _script(path: str) -> str:
    return (PROJECT_ROOT / path).read_text(encoding="utf-8")


def test_gpt_oss_smoke_is_two_gpu_harmony_gate() -> None:
    script = _script("scripts/run_gpt_oss_smoke_2gpu.sh")

    assert "set -Eeuo pipefail" in script
    assert "LONGCOQA_EXPECTED_GPUS=2" in script
    assert "LONGCOQA_SHARD_COUNT=2" in script
    assert "--samples 16" in script
    assert "run-gpt-oss-suite" in script
    assert "--shard-count 2" in script
    assert "runtime/gpt_oss_preflight/smoke16" in script
    assert "reasoning_effort" in script
    assert "Harmony control" in script
    assert "CJK" in script
    assert 'touch "$preflight_root/.complete"' in script


def test_gpt_oss_200_gate_requires_smoke_and_only_runs_z0_z1() -> None:
    script = _script("scripts/run_gpt_oss_preflight_200_2gpu.sh")

    assert "LONGCOQA_EXPECTED_GPUS=2" in script
    assert "smoke16/.complete" in script
    assert "--samples 200" in script
    assert "--protocol z0_current" in script
    assert "--protocol z1_teacher_forced" in script
    assert "g0_released_gold" not in script
    assert "g3_aligned_raw" not in script
    assert "runtime/gpt_oss_preflight/z0_z1_smoke200" in script
    assert "0.95" in script
    assert "diagnostics.json" in script
    assert 'touch "$preflight_root/.complete"' in script


@pytest.mark.parametrize(("failed", "expected"), [(0, True), (1, False)])
def test_gpt_oss_shard_reuse_requires_all_protocols_successful(
    tmp_path: Path, failed: int, expected: bool
) -> None:
    protocols = (
        "z0_current",
        "z1_teacher_forced",
        "g0_released_gold",
        "g3_aligned_raw",
    )
    suite = tmp_path / "suite_full.json"
    suite.write_text(
        json.dumps(
            {
                "schema_version": "longcoqa-gpt-oss-suite-v1",
                "protocols": list(protocols),
                "reports": {
                    protocol: {
                        "protocol": protocol,
                        "requested_samples": 377,
                        "successful_samples": 377 - failed,
                        "failed_samples": failed,
                    }
                    for protocol in protocols
                },
            }
        ),
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "LONGCOQA_PROJECT": str(PROJECT_ROOT),
        "LONGCOQA_VLLM_PYTHON": sys.executable,
    }
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1/scripts/gpt_oss_common.sh"; '
            'longcoqa_gpt_oss_shard_complete "$2"',
            "bash",
            str(PROJECT_ROOT),
            str(suite),
        ],
        env=env,
        check=False,
    )

    assert (result.returncode == 0) is expected
