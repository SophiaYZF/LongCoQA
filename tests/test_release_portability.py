from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from longcoqa.paths import DEFAULT_ENV_FILES, PROJECT_ROOT

SCANNED_ROOTS = ("longcoqa", "configs", "scripts", "tests")
FORBIDDEN = (
    "/" + "Users" + "/",
    "/" + "lustre" + "01",
    "feng" + "qin",
    "yu" + "zhengfeng",
    "../" + "dlc/common.sh",
)


def _isolated_subprocess_environment(**overrides: str) -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("LONGCOQA_ENV_FILE", None)
    environment.update(overrides)
    return environment


def test_public_text_has_no_private_paths_or_dlc_dependency() -> None:
    failures: list[str] = []
    for root_name in SCANNED_ROOTS:
        for path in (PROJECT_ROOT / root_name).rglob("*"):
            if not path.is_file() or path.suffix in {".pyc", ".csv"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if any(token in text for token in FORBIDDEN):
                failures.append(str(path.relative_to(PROJECT_ROOT)))
    assert failures == []


def test_model_configs_use_project_relative_model_paths() -> None:
    for path in (PROJECT_ROOT / "configs/models").glob("*.yaml"):
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert value["path"].startswith("models/")
        assert not Path(value["path"]).is_absolute()


def test_default_environment_files_stay_in_project() -> None:
    assert DEFAULT_ENV_FILES == (PROJECT_ROOT / ".env",)


@pytest.mark.parametrize(
    ("script_name", "completion_marker"),
    (
        ("run_p1_judge.sh", "outputs/formal/p1/.complete"),
        ("run_p1_cpt_judge.sh", "outputs/formal/p1_cpt/.complete"),
    ),
)
def test_judge_scripts_default_to_project_environment_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    script_name: str,
    completion_marker: str,
) -> None:
    project = tmp_path / "LongCoQA"
    script_dir = project / "scripts"
    script_dir.mkdir(parents=True)
    shutil.copyfile(PROJECT_ROOT / "scripts" / script_name, script_dir / script_name)
    shutil.copyfile(PROJECT_ROOT / "scripts/common.sh", script_dir / "common.sh")
    marker = project / completion_marker
    marker.parent.mkdir(parents=True)
    marker.touch()
    external_env = tmp_path / "host.env"
    external_env.write_text(
        "DASHSCOPE_API_KEYS=host-key\n"
        "DASHSCOPE_BASE_URL=https://host.invalid/v1\n"
        "DASHSCOPE_MODEL_ID=host-model\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("LONGCOQA_ENV_FILE", str(external_env))

    result = subprocess.run(
        ["bash", str(script_dir / script_name)],
        env=_isolated_subprocess_environment(LONGCOQA_PYTHON="/bin/sh"),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert result.stderr.strip() == f"Missing environment file: {project / '.env'}"


@pytest.mark.parametrize(
    ("script_name", "completion_marker"),
    (
        ("run_qwen37_p0_preflight.sh", None),
        ("run_qwen37_p0_full.sh", "runtime/qwen37_p0_preflight/.complete"),
        ("run_p1_judge.sh", "outputs/formal/p1/.complete"),
        ("run_p1_cpt_judge.sh", "outputs/formal/p1_cpt/.complete"),
    ),
)
def test_api_launchers_accept_python_command_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    script_name: str,
    completion_marker: str | None,
) -> None:
    project = tmp_path / "LongCoQA"
    script_dir = project / "scripts"
    script_dir.mkdir(parents=True)
    shutil.copyfile(PROJECT_ROOT / "scripts" / script_name, script_dir / script_name)
    shutil.copyfile(PROJECT_ROOT / "scripts/common.sh", script_dir / "common.sh")
    if completion_marker is not None:
        marker = project / completion_marker
        marker.parent.mkdir(parents=True)
        marker.touch()
    external_env = tmp_path / "host.env"
    external_env.write_text(
        "DASHSCOPE_API_KEYS=host-key\n"
        "DASHSCOPE_BASE_URL=https://host.invalid/v1\n"
        "DASHSCOPE_MODEL_ID=host-model\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("LONGCOQA_ENV_FILE", str(external_env))

    result = subprocess.run(
        ["bash", str(script_dir / script_name)],
        env=_isolated_subprocess_environment(LONGCOQA_PYTHON="sh"),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Missing Python executable" not in result.stderr
    assert result.stderr.strip() == f"Missing environment file: {project / '.env'}"


def test_cpt_factorial_launcher_accepts_default_python_command(tmp_path: Path) -> None:
    project = tmp_path / "LongCoQA"
    factory = tmp_path / "LlamaFactory"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = bin_dir / "python"
    python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python.chmod(0o755)
    env = {
        **os.environ,
        "LONGCOQA_PROJECT": str(project),
        "LLAMAFACTORY_ROOT": str(factory),
        "LONGCOQA_CPT_FACTORIAL_RUNTIME": str(tmp_path / "runtime"),
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
    }
    env.pop("LONGCOQA_PYTHON", None)

    result = subprocess.run(
        ["bash", str(PROJECT_ROOT / "scripts/run_cpt_factorial_retraining_2gpu.sh")],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert result.stderr.strip() == (
        f"Missing CPT factorial asset: {factory / '.venv/bin/llamafactory-cli'}"
    )


def test_retained_two_gpu_scripts_use_public_common_helper() -> None:
    names = (
        "run_glm_leak_repair_2gpu.sh",
        "run_single_model_preflight_2gpu.sh",
        "run_modern_rag_generation_debug_2gpu.sh",
        "run_gpt_oss_smoke_2gpu.sh",
        "run_cpt_factorial_dev_2gpu.sh",
        "run_p1_cpt_preflight_2gpu.sh",
        "run_gpt_oss_preflight_200_2gpu.sh",
        "run_p1_debug_2gpu.sh",
    )
    for name in names:
        text = (PROJECT_ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert 'source "$script_dir/common.sh"' in text
