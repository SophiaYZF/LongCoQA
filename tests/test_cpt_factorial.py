from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml
from safetensors.torch import save_file

from longcoqa.training.cpt_factorial import (
    build_training_manifest,
    prepare_clean_dataset,
    strip_leading_ideographic_indent,
    validate_adapter_output,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_strip_only_removes_line_leading_ideographic_spaces() -> None:
    raw = "　　第一行\n　　第二行含　空格\n普通行"

    clean, removed = strip_leading_ideographic_indent(raw)

    assert clean == "第一行\n第二行含　空格\n普通行"
    assert removed == 4


def test_prepare_clean_dataset_is_deterministic_and_preserves_payload(
    tmp_path: Path,
) -> None:
    source = tmp_path / "raw.jsonl"
    output = tmp_path / "clean.jsonl"
    source.write_text(
        json.dumps({"text": "　　甲\n　　乙", "id": 1}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    first = prepare_clean_dataset(source, output, expected_rows=1)
    first_bytes = output.read_bytes()
    second = prepare_clean_dataset(source, output, expected_rows=1)

    assert output.read_bytes() == first_bytes
    assert first == second
    assert first["removed_u3000"] == 4
    assert json.loads(output.read_text(encoding="utf-8"))["id"] == 1


def test_prepare_clean_dataset_rejects_invalid_rows(tmp_path: Path) -> None:
    source = tmp_path / "raw.jsonl"
    source.write_text('{"not_text": 1}\n', encoding="utf-8")

    try:
        prepare_clean_dataset(source, tmp_path / "clean.jsonl", expected_rows=1)
    except ValueError as error:
        assert "text" in str(error)
    else:
        raise AssertionError("missing text must be rejected")


def test_factorial_configs_change_only_registered_factors() -> None:
    paths = {
        "A": "qwen3_5_9b_cpt_lora.yaml",
        "B": "qwen3_5_9b_cpt_clean_lora.yaml",
        "C": "qwen3_5_9b_cpt_conservative_lora.yaml",
        "D": "qwen3_5_9b_cpt_clean_conservative_lora.yaml",
    }
    cells = {
        name: yaml.safe_load(
            (PROJECT_ROOT / "configs" / "training" / filename).read_text(
                encoding="utf-8"
            )
        )
        for name, filename in paths.items()
    }

    assert cells["A"]["dataset"] == "longcoqa_novels_cpt"
    assert cells["B"]["dataset"] == "longcoqa_novels_cpt_noindent"
    assert cells["B"]["learning_rate"] == 5.0e-5
    assert cells["B"]["loraplus_lr_ratio"] == 8.0
    assert cells["C"]["dataset"] == "longcoqa_novels_cpt"
    assert cells["C"]["learning_rate"] == 1.0e-5
    assert "loraplus_lr_ratio" not in cells["C"]
    assert cells["D"]["dataset"] == "longcoqa_novels_cpt_noindent"
    assert cells["D"]["learning_rate"] == 1.0e-5
    assert "loraplus_lr_ratio" not in cells["D"]
    assert len({cell["output_dir"] for cell in cells.values()}) == 4

    allowed = {"dataset", "learning_rate", "loraplus_lr_ratio", "output_dir"}
    baseline = {key: value for key, value in cells["A"].items() if key not in allowed}
    for name in ("B", "C", "D"):
        compared = {key: value for key, value in cells[name].items() if key not in allowed}
        assert compared == baseline


def _write_fake_adapter(
    output: Path,
    *,
    global_step: int = 7,
    train_loss: float = 3.0,
    tensor_value: float = 1.0,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (output / "trainer_state.json").write_text(
        json.dumps({"global_step": global_step, "epoch": 1.0}) + "\n",
        encoding="utf-8",
    )
    (output / "train_results.json").write_text(
        json.dumps({"train_loss": train_loss}) + "\n",
        encoding="utf-8",
    )
    save_file(
        {"adapter.weight": torch.tensor([tensor_value], dtype=torch.float32)},
        output / "adapter_model.safetensors",
    )


def test_validate_adapter_output_accepts_complete_finite_adapter(tmp_path: Path) -> None:
    _write_fake_adapter(tmp_path)

    report = validate_adapter_output(tmp_path, expected_steps=7)

    assert report["global_step"] == 7
    assert report["train_loss"] == 3.0
    assert report["tensor_count"] == 1
    assert len(report["adapter_sha256"]) == 64


def test_validate_adapter_output_rejects_wrong_global_step(tmp_path: Path) -> None:
    _write_fake_adapter(tmp_path, global_step=6)

    with pytest.raises(ValueError, match="global_step"):
        validate_adapter_output(tmp_path, expected_steps=7)


def test_validate_adapter_output_rejects_nonfinite_tensor(tmp_path: Path) -> None:
    _write_fake_adapter(tmp_path, tensor_value=float("nan"))

    with pytest.raises(ValueError, match="non-finite"):
        validate_adapter_output(tmp_path, expected_steps=7)


def test_build_training_manifest_records_registered_factors(tmp_path: Path) -> None:
    _write_fake_adapter(tmp_path)
    validation = validate_adapter_output(tmp_path)

    manifest = build_training_manifest(
        cell="D",
        dataset_level="clean",
        update_level="conservative",
        config_sha256="c" * 64,
        data_sha256="d" * 64,
        base_model="Qwen3.5-9B",
        validation=validation,
        global_batch=24,
        log_path=Path("runtime/train.log"),
    )

    assert manifest["cell"] == "D"
    assert manifest["factors"] == {
        "dataset": "clean",
        "update": "conservative",
    }
    assert manifest["optimizer_steps"] == 7
    assert manifest["global_batch"] == 24
