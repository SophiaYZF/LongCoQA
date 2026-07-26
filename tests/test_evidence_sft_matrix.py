from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "configs/experiments/phase_g_evidence_sft.yaml"


def _read_yaml(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_evidence_sft_matrix_has_the_preregistered_eight_cells() -> None:
    matrix = _read_yaml(MATRIX_PATH)
    cells = matrix["training"]["cells"]
    observed = {
        (cell["id"], cell["initialization"], cell["objective"], cell["seed"])
        for cell in cells
    }
    assert observed == {
        ("A1", "base", "raw_answer", 42),
        ("A2", "base", "clean_answer", 42),
        ("A3", "base", "evidence_aware", 42),
        ("A4", "cpt", "raw_answer", 42),
        ("A5", "cpt", "clean_answer", 42),
        ("A6", "cpt", "evidence_aware", 42),
        ("A7", "base", "evidence_aware", 43),
        ("A8", "cpt", "evidence_aware", 43),
    }
    assert len({cell["config"] for cell in cells}) == 8
    assert len({cell["output_dir"] for cell in cells}) == 8


def test_all_training_configs_share_the_fixed_conservative_recipe() -> None:
    matrix = _read_yaml(MATRIX_PATH)
    common = {
        "stage": "sft",
        "finetuning_type": "lora",
        "lora_rank": 64,
        "lora_alpha": 128,
        "lora_dropout": 0.05,
        "lora_target": "all",
        "use_rslora": True,
        "learning_rate": 2.0e-5,
        "num_train_epochs": 2.0,
        "cutoff_len": 4096,
        "mask_history": True,
        "load_best_model_at_end": False,
    }
    objective_datasets = {
        "raw_answer": "longcoqa_evidence_raw_answer_train",
        "clean_answer": "longcoqa_evidence_clean_answer_train",
        "evidence_aware": "longcoqa_evidence_evidence_aware_train",
    }
    cpt_cells = {"A4", "A5", "A6", "A8"}
    for cell in matrix["training"]["cells"]:
        config = _read_yaml(ROOT / cell["config"])
        for key, expected in common.items():
            assert config[key] == expected, (cell["id"], key)
        assert "loraplus_lr_ratio" not in config
        assert config["dataset"] == objective_datasets[cell["objective"]]
        assert config["eval_dataset"] == objective_datasets[cell["objective"]].replace(
            "_train", "_dev"
        )
        assert config["seed"] == config["data_seed"] == cell["seed"]
        assert config["output_dir"].endswith(cell["output_dir"])
        if cell["id"] in cpt_cells:
            assert config["adapter_name_or_path"].endswith(
                "outputs/training/qwen3_5_9b_cpt_conservative_lora"
            )
        else:
            assert "adapter_name_or_path" not in config
