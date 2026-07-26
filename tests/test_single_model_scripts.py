from pathlib import Path

import yaml

from longcoqa.paths import PROJECT_ROOT


def _script(path: str) -> str:
    return (PROJECT_ROOT / path).read_text(encoding="utf-8")


def test_single_model_configs_match_local_runner_contract() -> None:
    required = {
        "name",
        "path",
        "revision",
        "dtype",
        "cuda_visible_devices",
        "tensor_parallel_size",
        "gpu_memory_utilization",
        "max_model_len",
        "max_num_seqs",
        "trust_remote_code",
        "enforce_eager",
        "output_root",
    }
    expected = {
        "glm4_7_flash.yaml": (
            "models/zai-org/GLM-4.7-Flash",
            "7dd20894a642a0aa287e9827cb1a1f7f91386b67",
        ),
        "qwen3_8b.yaml": (
            "models/Qwen/Qwen3-8B",
            "b968826d9c46dd6066d109eabc6255188de91218",
        ),
        "qwen3_5_35b_a3b.yaml": (
            "models/Qwen/Qwen3.5-35B-A3B",
            "59d61f3ce65a6d9863b86d2e96597125219dc754",
        ),
    }
    for filename, (model_path, revision) in expected.items():
        config = yaml.safe_load(
            (PROJECT_ROOT / "configs/models" / filename).read_text(encoding="utf-8")
        )
        assert required <= set(config)
        assert config["path"] == model_path
        assert not Path(config["path"]).is_absolute()
        assert config["revision"] == revision


def test_single_model_preflight_is_isolated_and_strict() -> None:
    script = _script("scripts/run_single_model_preflight_2gpu.sh")

    assert "set -Eeuo pipefail" in script
    assert "LONGCOQA_SINGLE_MODEL_CONFIG" in script
    assert "LONGCOQA_SINGLE_MODEL_SLUG" in script
    assert "LONGCOQA_EXPECTED_GPUS=2" in script
    assert "LONGCOQA_SHARD_COUNT=2" in script
    assert "--samples \"$limit\"" in script
    assert "--shard-count 2" in script
    assert "runtime/single_model_preflight" in script
    assert "successful_samples" in script
    assert "failed_samples" in script
    assert 'touch "$preflight_root/.complete"' in script
