from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_cpt_factorial_retraining_2gpu.sh"


def _write_executable(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _fixture(tmp_path: Path) -> tuple[dict[str, str], Path, Path]:
    project = tmp_path / "project"
    factory = tmp_path / "factory"
    runtime = tmp_path / "runtime"
    calls = tmp_path / "calls.jsonl"
    data_dir = project / "artifacts" / "training"
    config_dir = project / "configs" / "training"
    base_dir = project / "outputs" / "training" / "qwen3_5_9b_cpt_lora"
    data_dir.mkdir(parents=True)
    config_dir.mkdir(parents=True)
    base_dir.mkdir(parents=True)
    (base_dir / "adapter_model.safetensors").write_bytes(b"original-adapter")
    (base_dir / ".complete").touch()
    raw_rows = [{"text": "　　甲\n　　乙", "id": index} for index in range(293)]
    (data_dir / "novels_cpt.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in raw_rows),
        encoding="utf-8",
    )
    (data_dir / "dataset_info.json").write_text("{}\n", encoding="utf-8")

    specs = {
        "clean": ("longcoqa_novels_cpt_noindent", 5.0e-5, 8.0),
        "conservative": ("longcoqa_novels_cpt", 1.0e-5, None),
        "clean_conservative": ("longcoqa_novels_cpt_noindent", 1.0e-5, None),
    }
    for name, (dataset, learning_rate, ratio) in specs.items():
        config = {
            "model_name_or_path": "Qwen3.5-9B",
            "stage": "pt",
            "dataset": dataset,
            "output_dir": str(
                project / "outputs" / "training" / f"qwen3_5_9b_cpt_{name}_lora"
            ),
            "per_device_train_batch_size": 3,
            "gradient_accumulation_steps": 1,
            "learning_rate": learning_rate,
        }
        if ratio is not None:
            config["loraplus_lr_ratio"] = ratio
        (config_dir / f"qwen3_5_9b_cpt_{name}_lora.yaml").write_text(
            yaml.safe_dump(config), encoding="utf-8"
        )

    fake_cli = factory / ".venv" / "bin" / "llamafactory-cli"
    _write_executable(
        fake_cli,
        "#!/usr/bin/env python3\n"
        "import json, os, sys, yaml\n"
        "from pathlib import Path\n"
        "from safetensors.torch import save_file\n"
        "import torch\n"
        "config=yaml.safe_load(Path(sys.argv[2]).read_text())\n"
        "with Path(os.environ['FAKE_CALLS']).open('a') as h: "
        "h.write(json.dumps({'config':Path(sys.argv[2]).name,'args':sys.argv[3:]})+'\\n')\n"
        "fail=os.environ.get('FAKE_FAIL_CELL')\n"
        "if fail and fail in Path(sys.argv[2]).name: sys.exit(9)\n"
        "out=Path(config['output_dir']); out.mkdir(parents=True, exist_ok=True)\n"
        "(out/'adapter_config.json').write_text('{}\\n')\n"
        "(out/'trainer_state.json').write_text('{\"global_step\":7,\"epoch\":1.0}\\n')\n"
        "(out/'train_results.json').write_text('{\"train_loss\":3.0}\\n')\n"
        "save_file({'adapter.weight':torch.ones(1)}, out/'adapter_model.safetensors')\n",
    )
    fake_nvidia = tmp_path / "nvidia-smi"
    _write_executable(
        fake_nvidia,
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "if [[ $* == *compute-apps* ]]; then exit 0; fi\n"
        "if [[ $* == *memory.used* ]]; then printf '0\\n0\\n'; exit 0; fi\n"
        "if [[ $* == *timestamp* ]]; then printf '2026-01-01, 0, 0, 0\\n2026-01-01, 1, 0, 0\\n'; exit 0; fi\n"
        "printf '0, NVIDIA H200, 143771\\n1, NVIDIA H200, 143771\\n'\n",
    )
    env = os.environ.copy()
    env.update(
        {
            "LONGCOQA_PROJECT": str(project),
            "LONGCOQA_WORKSPACE": str(tmp_path),
            "LLAMAFACTORY_ROOT": str(factory),
            "LONGCOQA_PYTHON": sys.executable,
            "LONGCOQA_CPT_FACTORIAL_RUNTIME": str(runtime),
            "NVIDIA_SMI_BIN": str(fake_nvidia),
            "FAKE_CALLS": str(calls),
            "PYTHONPATH": str(PROJECT_ROOT),
            "CUDA_VISIBLE_DEVICES": "0,1",
            "PATH": f"{Path(sys.executable).parent}{os.pathsep}{env['PATH']}",
        }
    )
    return env, calls, project


def test_launcher_runs_b_c_d_with_equivalent_global_batch(tmp_path: Path) -> None:
    env, calls, project = _fixture(tmp_path)

    subprocess.run(["bash", str(SCRIPT)], check=True, env=env, timeout=30)

    rows = [json.loads(line) for line in calls.read_text().splitlines()]
    assert [row["config"] for row in rows] == [
        "qwen3_5_9b_cpt_clean_lora.yaml",
        "qwen3_5_9b_cpt_conservative_lora.yaml",
        "qwen3_5_9b_cpt_clean_conservative_lora.yaml",
    ]
    assert all("per_device_train_batch_size=3" in row["args"] for row in rows)
    assert all("gradient_accumulation_steps=4" in row["args"] for row in rows)
    for name in ("clean", "conservative", "clean_conservative"):
        output = project / "outputs" / "training" / f"qwen3_5_9b_cpt_{name}_lora"
        assert (output / ".complete").is_file()
        assert (output / "training_manifest.json").is_file()


def test_launcher_stops_after_first_failed_cell(tmp_path: Path) -> None:
    env, calls, _ = _fixture(tmp_path)
    env["FAKE_FAIL_CELL"] = "conservative_lora.yaml"

    result = subprocess.run(["bash", str(SCRIPT)], env=env, timeout=30)

    assert result.returncode != 0
    rows = [json.loads(line) for line in calls.read_text().splitlines()]
    assert len(rows) == 2


def test_launcher_dry_run_makes_no_training_calls(tmp_path: Path) -> None:
    env, calls, _ = _fixture(tmp_path)
    env["LONGCOQA_CPT_FACTORIAL_DRY_RUN"] = "1"

    subprocess.run(["bash", str(SCRIPT)], check=True, env=env, timeout=30)

    assert not calls.exists()


def test_launcher_dry_run_rejects_wrong_factor_config(tmp_path: Path) -> None:
    env, _, project = _fixture(tmp_path)
    env["LONGCOQA_CPT_FACTORIAL_DRY_RUN"] = "1"
    config_path = (
        project / "configs" / "training" / "qwen3_5_9b_cpt_clean_lora.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["learning_rate"] = 9.0e-5
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = subprocess.run(["bash", str(SCRIPT)], env=env, timeout=30)

    assert result.returncode != 0
