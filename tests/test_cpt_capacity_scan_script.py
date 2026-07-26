from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_executable(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def test_scan_starts_at_two_and_stops_when_batch_three_hits_target(tmp_path: Path) -> None:
    project = tmp_path / "project"
    factory = tmp_path / "factory"
    scan_root = tmp_path / "scan"
    config = project / "configs" / "training" / "qwen3_5_9b_cpt_lora.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("stage: pt\ntemplate: empty\n", encoding="utf-8")
    (project / "artifacts" / "training").mkdir(parents=True)
    (project / "artifacts" / "training" / "dataset_info.json").write_text("{}\n", encoding="utf-8")

    calls = tmp_path / "calls.txt"
    fake_cli = factory / ".venv" / "bin" / "llamafactory-cli"
    _write_executable(
        fake_cli,
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s ' \"$@\" >> \"$FAKE_CALLS\"; printf '\\n' >> \"$FAKE_CALLS\"\n"
        "for arg in \"$@\"; do case \"$arg\" in output_dir=*) out=${arg#output_dir=};; esac; done\n"
        "sleep 1.2\n"
        "mkdir -p \"$out\"\n"
        "printf '{}\\n' > \"$out/adapter_config.json\"\n"
        "printf 'weights\\n' > \"$out/adapter_model.safetensors\"\n"
        "printf '{\"train_loss\":2.0,\"train_runtime\":1.0}\\n' > \"$out/train_results.json\"\n",
    )
    fake_nvidia = tmp_path / "nvidia-smi"
    _write_executable(
        fake_nvidia,
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "args=$*\n"
        "if [[ $args == *memory.total* ]]; then printf '0, 143771\\n1, 143771\\n'; exit; fi\n"
        "if [[ $args == *memory.used* && $args != *timestamp* ]]; then printf '0\\n0\\n'; exit; fi\n"
        "batch=$(cat \"$SCAN_ROOT/current_batch\" 2>/dev/null || printf 2)\n"
        "if [[ $batch == 2 ]]; then mem=110000; else mem=129000; fi\n"
        "printf '2026-07-20 00:00:00, 0, %s, 80, 500\\n' \"$mem\"\n"
        "printf '2026-07-20 00:00:00, 1, %s, 81, 501\\n' \"$((mem-100))\"\n",
    )

    env = os.environ.copy()
    env.update(
        {
            "LONGCOQA_PROJECT": str(project),
            "LLAMAFACTORY_ROOT": str(factory),
            "LONGCOQA_PYTHON": sys.executable,
            "LONGCOQA_CPT_SCAN_ROOT": str(scan_root),
            "LONGCOQA_CPT_SCAN_MAX_STEPS": "2",
            "LONGCOQA_CPT_SCAN_MAX_SAMPLES": "32",
            "NVIDIA_SMI_BIN": str(fake_nvidia),
            "FAKE_CALLS": str(calls),
            "SCAN_ROOT": str(scan_root),
            "PYTHONPATH": str(PROJECT_ROOT),
            "CUDA_VISIBLE_DEVICES": "0,1",
        }
    )
    subprocess.run(
        ["bash", str(PROJECT_ROOT / "scripts" / "run_cpt_capacity_scan_2gpu.sh")],
        check=True,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )

    call_lines = calls.read_text(encoding="utf-8").splitlines()
    assert len(call_lines) == 2
    assert "per_device_train_batch_size=2" in call_lines[0]
    assert "gradient_accumulation_steps=2" in call_lines[0]
    assert "per_device_train_batch_size=3" in call_lines[1]
    assert "gradient_accumulation_steps=1" in call_lines[1]
    assert all("packing=true" in line for line in call_lines)
    assert all("per_device_train_batch_size=4" not in line for line in call_lines)

    summary = json.loads((scan_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["selected"]["batch_size"] == 3
    assert summary["selected"]["selection_source"] == "scan"
