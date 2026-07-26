import importlib.util
from pathlib import Path

import yaml

from longcoqa.paths import PROJECT_ROOT


def _load_download_module():
    path = PROJECT_ROOT / "scripts/download_gpt_oss_120b.py"
    spec = importlib.util.spec_from_file_location("download_gpt_oss_120b", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gpt_oss_config_contract() -> None:
    path = PROJECT_ROOT / "configs/models/gpt_oss_120b.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert raw["path"] == "models/openai/gpt-oss-120b"
    assert not Path(raw["path"]).is_absolute()
    assert raw["revision"] == "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
    assert raw["dtype"] == "auto"
    assert raw["response_format"] == "harmony"
    assert raw["reasoning_effort"] == "low"
    assert raw["tensor_parallel_size"] == 1
    assert raw["gpu_memory_utilization"] == 0.88
    assert raw["max_model_len"] == 32768
    assert raw["max_tokens_factoid"] == 2048
    assert raw["max_tokens_non_factoid"] == 2048


def test_download_scope_excludes_duplicate_formats() -> None:
    module = _load_download_module()

    assert module.REPO_ID == "openai/gpt-oss-120b"
    assert module.REVISION == "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
    assert "model-*.safetensors" in module.ALLOW_PATTERNS
    assert "original/*" in module.IGNORE_PATTERNS
    assert "metal/*" in module.IGNORE_PATTERNS
    assert module.EXPECTED_SHARDS == 15
    assert module.EXPECTED_WEIGHT_BYTES == 65_248_893_184


def test_validate_snapshot_rejects_wrong_weight_set(tmp_path: Path) -> None:
    module = _load_download_module()
    (tmp_path / "model-00001-of-00015.safetensors").write_bytes(b"wrong")

    try:
        module.validate_snapshot(tmp_path)
    except RuntimeError as exc:
        assert "Invalid checkpoint" in str(exc)
    else:
        raise AssertionError("invalid checkpoint was accepted")
