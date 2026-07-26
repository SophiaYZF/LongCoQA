"""Utilities for the preregistered CPT factorial retraining experiment."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors import safe_open

from longcoqa.utils.hashing import sha256_file
from longcoqa.utils.io import atomic_write_json, atomic_write_jsonl, read_jsonl

_LEADING_U3000 = re.compile(r"(?m)^\u3000+")


def strip_leading_ideographic_indent(text: str) -> tuple[str, int]:
    """Remove only U+3000 runs at the start of a logical line."""
    removed = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal removed
        removed += len(match.group(0))
        return ""

    return _LEADING_U3000.sub(replace, text), removed


def prepare_clean_dataset(
    input_path: Path,
    output_path: Path,
    *,
    expected_rows: int = 293,
) -> dict[str, Any]:
    """Create the deterministic no-indent CPT dataset and return its audit."""
    rows = read_jsonl(input_path)
    if len(rows) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} CPT rows, found {len(rows)} in {input_path}"
        )

    cleaned_rows: list[dict[str, Any]] = []
    removed_total = 0
    for index, row in enumerate(rows, start=1):
        text = row.get("text")
        if not isinstance(text, str):
            raise ValueError(f"Expected string text at {input_path}:{index}")
        cleaned, removed = strip_leading_ideographic_indent(text)
        if _LEADING_U3000.search(cleaned):
            raise ValueError(f"Line-leading U+3000 remains at {input_path}:{index}")
        normalized_raw, _ = strip_leading_ideographic_indent(text)
        if normalized_raw != cleaned:
            raise ValueError(f"Non-indent content changed at {input_path}:{index}")
        cleaned_rows.append({**row, "text": cleaned})
        removed_total += removed

    atomic_write_jsonl(output_path, cleaned_rows)
    return {
        "rows": len(cleaned_rows),
        "removed_u3000": removed_total,
        "remaining_line_leading_u3000": 0,
        "input_sha256": sha256_file(input_path),
        "output_sha256": sha256_file(output_path),
        "input": str(input_path),
        "output": str(output_path),
    }


def validate_adapter_output(
    output_dir: Path,
    *,
    expected_steps: int = 7,
) -> dict[str, Any]:
    """Validate a completed adapter without creating its completion marker."""
    required = {
        name: output_dir / name
        for name in (
            "adapter_config.json",
            "adapter_model.safetensors",
            "trainer_state.json",
            "train_results.json",
        )
    }
    for name, path in required.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty {name}: {path}")

    trainer_state = json.loads(required["trainer_state.json"].read_text(encoding="utf-8"))
    results = json.loads(required["train_results.json"].read_text(encoding="utf-8"))
    global_step = trainer_state.get("global_step")
    if global_step != expected_steps:
        raise ValueError(f"Expected global_step={expected_steps}, found {global_step}")
    epoch = trainer_state.get("epoch")
    if not isinstance(epoch, (int, float)) or not math.isfinite(epoch) or epoch < 1.0:
        raise ValueError(f"Expected completed finite epoch, found {epoch!r}")
    train_loss = results.get("train_loss")
    if not isinstance(train_loss, (int, float)) or not math.isfinite(train_loss):
        raise ValueError(f"Expected finite train_loss, found {train_loss!r}")

    tensor_count = 0
    weights = required["adapter_model.safetensors"]
    with safe_open(weights, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            tensor = handle.get_tensor(name)
            tensor_count += 1
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"Adapter contains non-finite tensor: {name}")
    if tensor_count == 0:
        raise ValueError("Adapter contains no tensors")

    return {
        "output_dir": str(output_dir),
        "global_step": int(global_step),
        "epoch": float(epoch),
        "train_loss": float(train_loss),
        "tensor_count": tensor_count,
        "adapter_bytes": weights.stat().st_size,
        "adapter_sha256": sha256_file(weights),
    }


def build_training_manifest(
    *,
    cell: str,
    dataset_level: str,
    update_level: str,
    config_sha256: str,
    data_sha256: str,
    base_model: str,
    validation: dict[str, Any],
    global_batch: int,
    log_path: Path,
) -> dict[str, Any]:
    """Build the immutable per-cell audit manifest."""
    return {
        "schema_version": "longcoqa-cpt-factorial-training-v1",
        "cell": cell,
        "factors": {"dataset": dataset_level, "update": update_level},
        "config_sha256": config_sha256,
        "data_sha256": data_sha256,
        "base_model": base_model,
        "global_batch": global_batch,
        "optimizer_steps": int(validation["global_step"]),
        "adapter_sha256": validation["adapter_sha256"],
        "train_loss": validation["train_loss"],
        "tensor_count": validation["tensor_count"],
        "log": str(log_path),
    }


def _max_u3000_run(text: str) -> int:
    return max((len(match.group(0)) for match in re.finditer(r"\u3000+", text)), default=0)


def summarize_dev_rows(
    rows: list[dict[str, Any]],
    *,
    expected: int,
) -> dict[str, Any]:
    """Summarize one model/protocol dev cell and apply collapse gates."""
    sample_ids = [str(row.get("sample_id")) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Duplicate sample IDs in CPT factorial dev cell")
    successful = [row for row in rows if row.get("status") == "ok"]
    predictions = [str(row.get("prediction", "")) for row in successful]
    blank_answers = sum(not prediction.strip() for prediction in predictions)
    length_count = sum(row.get("finish_reason") == "length" for row in successful)
    normal_stops = sum(row.get("finish_reason") == "stop" for row in successful)
    total_characters = sum(len(prediction) for prediction in predictions)
    u3000_characters = sum(prediction.count("\u3000") for prediction in predictions)
    answerable_scores = [
        float(row["metrics"]["character_f1"])
        for row in successful
        if row.get("answerable", True) and row.get("metrics") is not None
    ]
    length_ratio = length_count / len(successful) if successful else 1.0
    maximum_run = max((_max_u3000_run(text) for text in predictions), default=0)
    gate_passed = (
        len(rows) == expected
        and len(successful) == expected
        and blank_answers == 0
        and length_ratio <= 0.10
        and maximum_run <= 14
    )
    return {
        "requested_samples": len(rows),
        "successful_samples": len(successful),
        "failed_samples": len(rows) - len(successful),
        "unique_samples": len(set(sample_ids)),
        "blank_answers": blank_answers,
        "length_finishes": length_count,
        "length_ratio": length_ratio,
        "normal_stops": normal_stops,
        "normal_stop_rate": normal_stops / len(successful) if successful else 0.0,
        "u3000_characters": u3000_characters,
        "u3000_character_ratio": (
            u3000_characters / total_characters if total_characters else 0.0
        ),
        "max_u3000_run": maximum_run,
        "answerable_samples": len(answerable_scores),
        "character_f1": (
            sum(answerable_scores) / len(answerable_scores) if answerable_scores else None
        ),
        "gate_passed": gate_passed,
    }


def summarize_dev_results(
    root: Path,
    *,
    expected_per_cell: int = 16,
    model_variants: tuple[str, ...] = (
        "cpt_raw_aggressive",
        "cpt_clean_aggressive",
        "cpt_raw_conservative",
        "cpt_clean_conservative",
    ),
    eval_variants: tuple[str, ...] = ("conversation", "grounded", "raft"),
) -> dict[str, Any]:
    """Validate and summarize the complete four-by-three dev matrix."""
    cells: dict[str, Any] = {}
    for model_variant in model_variants:
        for eval_variant in eval_variants:
            path = root / model_variant / eval_variant / "full" / "results.jsonl"
            if not path.is_file():
                raise FileNotFoundError(f"Missing CPT factorial dev cell: {path}")
            cells[f"{model_variant}/{eval_variant}"] = summarize_dev_rows(
                read_jsonl(path),
                expected=expected_per_cell,
            )
    primary = model_variants[-1]
    primary_gate_passed = all(
        cells[f"{primary}/{eval_variant}"]["gate_passed"]
        for eval_variant in eval_variants
    )
    return {
        "schema_version": "longcoqa-cpt-factorial-dev-v1",
        "root": str(root),
        "expected_per_cell": expected_per_cell,
        "models": list(model_variants),
        "eval_variants": list(eval_variants),
        "cells": cells,
        "primary_candidate": primary,
        "primary_gate_passed": primary_gate_passed,
    }


def validate_factorial_config(
    config_path: Path,
    *,
    dataset_level: str,
    update_level: str,
    expected_output: Path,
) -> dict[str, Any]:
    """Validate the preregistered factors encoded by one training config."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Expected mapping in {config_path}")
    expected_dataset = (
        "longcoqa_novels_cpt_noindent" if dataset_level == "clean" else "longcoqa_novels_cpt"
    )
    if config.get("dataset") != expected_dataset:
        raise ValueError(
            f"Expected dataset={expected_dataset}, found {config.get('dataset')!r}"
        )
    expected_lr = 5.0e-5 if update_level == "aggressive" else 1.0e-5
    if float(config.get("learning_rate", -1)) != expected_lr:
        raise ValueError(
            f"Expected learning_rate={expected_lr}, found {config.get('learning_rate')!r}"
        )
    ratio = config.get("loraplus_lr_ratio")
    if update_level == "aggressive" and float(ratio or -1) != 8.0:
        raise ValueError(f"Expected loraplus_lr_ratio=8.0, found {ratio!r}")
    if update_level == "conservative" and ratio is not None:
        raise ValueError(f"Conservative config must disable LoRA+, found {ratio!r}")
    if Path(str(config.get("output_dir"))) != expected_output:
        raise ValueError(
            f"Expected output_dir={expected_output}, found {config.get('output_dir')!r}"
        )
    return {
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "dataset_level": dataset_level,
        "update_level": update_level,
        "output_dir": str(expected_output),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-data")
    prepare.add_argument("--input", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--expected-rows", type=int, default=293)
    validate = subparsers.add_parser("validate-output")
    validate.add_argument("--output-dir", type=Path, required=True)
    validate.add_argument("--expected-steps", type=int, default=7)
    manifest = subparsers.add_parser("write-manifest")
    manifest.add_argument("--output-dir", type=Path, required=True)
    manifest.add_argument("--cell", required=True)
    manifest.add_argument("--dataset-level", choices=("raw", "clean"), required=True)
    manifest.add_argument(
        "--update-level", choices=("aggressive", "conservative"), required=True
    )
    manifest.add_argument("--config", type=Path, required=True)
    manifest.add_argument("--data", type=Path, required=True)
    manifest.add_argument("--base-model", required=True)
    manifest.add_argument("--global-batch", type=int, default=24)
    manifest.add_argument("--log", type=Path, required=True)
    manifest.add_argument("--expected-steps", type=int, default=7)
    summarize = subparsers.add_parser("summarize-dev")
    summarize.add_argument("--root", type=Path, required=True)
    summarize.add_argument("--output", type=Path, required=True)
    summarize.add_argument("--expected-per-cell", type=int, default=16)
    check = subparsers.add_parser("validate-config")
    check.add_argument("--config", type=Path, required=True)
    check.add_argument("--dataset-level", choices=("raw", "clean"), required=True)
    check.add_argument(
        "--update-level", choices=("aggressive", "conservative"), required=True
    )
    check.add_argument("--expected-output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "prepare-data":
        report = prepare_clean_dataset(
            args.input,
            args.output,
            expected_rows=args.expected_rows,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "validate-output":
        report = validate_adapter_output(
            args.output_dir,
            expected_steps=args.expected_steps,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "write-manifest":
        validation = validate_adapter_output(
            args.output_dir,
            expected_steps=args.expected_steps,
        )
        report = build_training_manifest(
            cell=args.cell,
            dataset_level=args.dataset_level,
            update_level=args.update_level,
            config_sha256=sha256_file(args.config),
            data_sha256=sha256_file(args.data),
            base_model=args.base_model,
            validation=validation,
            global_batch=args.global_batch,
            log_path=args.log,
        )
        destination = args.output_dir / "training_manifest.json"
        atomic_write_json(destination, report)
        print(json.dumps({**report, "manifest": str(destination)}, indent=2))
    elif args.command == "summarize-dev":
        report = summarize_dev_results(
            args.root,
            expected_per_cell=args.expected_per_cell,
        )
        atomic_write_json(args.output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "validate-config":
        report = validate_factorial_config(
            args.config,
            dataset_level=args.dataset_level,
            update_level=args.update_level,
            expected_output=args.expected_output,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
