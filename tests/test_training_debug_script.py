from __future__ import annotations

import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_cpt_config_uses_plain_text_template() -> None:
    config = (PROJECT_ROOT / "configs" / "training" / "qwen3_5_9b_cpt_lora.yaml").read_text(
        encoding="utf-8"
    )
    assert "template: empty" in config.splitlines()


def test_cpt_config_uses_selected_capacity_scan_parameters() -> None:
    config = (PROJECT_ROOT / "configs" / "training" / "qwen3_5_9b_cpt_lora.yaml").read_text(
        encoding="utf-8"
    )
    lines = config.splitlines()
    assert "packing: true" in lines
    assert "per_device_train_batch_size: 3" in lines
    assert "gradient_accumulation_steps: 1" in lines


def test_cpt_debug_disables_evaluation_without_dev_set(tmp_path: Path) -> None:
    project = tmp_path / "project"
    factory = tmp_path / "factory"
    config_dir = project / "configs" / "training"
    cli_dir = factory / ".venv" / "bin"
    config_dir.mkdir(parents=True)
    cli_dir.mkdir(parents=True)
    (config_dir / "qwen3_5_9b_cpt_lora.yaml").write_text("stage: pt\n", encoding="utf-8")

    args_file = tmp_path / "args.txt"
    fake_cli = cli_dir / "llamafactory-cli"
    fake_cli.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s\\n' \"$@\" > \"$FAKE_ARGS_FILE\"\n"
        "for arg in \"$@\"; do\n"
        "  case \"$arg\" in output_dir=*) output=${arg#output_dir=};; esac\n"
        "done\n"
        "mkdir -p \"$output\"\n"
        "printf '{}\\n' > \"$output/adapter_config.json\"\n",
        encoding="utf-8",
    )
    fake_cli.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "LONGCOQA_PROJECT": str(project),
            "LLAMAFACTORY_ROOT": str(factory),
            "CUDA_VISIBLE_DEVICES": "0,1",
            "FAKE_ARGS_FILE": str(args_file),
        }
    )
    subprocess.run(
        ["bash", str(PROJECT_ROOT / "scripts" / "run_training_debug_2gpu.sh"), "cpt"],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )

    passed_args = args_file.read_text(encoding="utf-8").splitlines()
    assert 'eval_strategy="no"' in passed_args
    assert "eval_strategy=no" not in passed_args
    assert "eval_strategy=steps" not in passed_args
