from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from longcoqa.data.loader import LongCoQADataset
from longcoqa.data.split import optimize_dialogue_split
from longcoqa.utils.hashing import stable_json_hash
from longcoqa.utils.io import atomic_write_json

SCHEMA_VERSION = "longcoqa-manifest-v1"


def _base_manifest(dataset: LongCoQADataset, protocol: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": protocol,
        "created_at": datetime.now(UTC).isoformat(),
        "source_checksums": dataset.source_checksums(),
    }


def build_p0_manifest(dataset: LongCoQADataset) -> dict[str, Any]:
    samples = []
    for sample in dataset.iter_samples():
        samples.append(
            {
                "sample_id": sample["sample_id"],
                "question_type": sample["question_type"],
                "dialogue_id": sample["dialogue_id"],
                "turn": sample["turn"],
                "split": "all",
            }
        )
    manifest = _base_manifest(dataset, "P0-full-benchmark")
    manifest["samples"] = samples
    manifest["counts"] = {
        "samples": len(samples),
        "dialogues": len(dataset.dialogues()),
    }
    manifest["content_hash"] = stable_json_hash(samples)
    return manifest


def build_p1_manifest(
    dataset: LongCoQADataset,
    *,
    seed: int = 42,
    ratios: dict[str, float] | None = None,
    restarts: int = 24,
    local_steps: int = 4000,
    evidence_cut_weight: float = 1.0,
    duplicate_question_penalty: float = 1000.0,
) -> dict[str, Any]:
    result = optimize_dialogue_split(
        dataset.dialogues(),
        ratios=ratios,
        seed=seed,
        restarts=restarts,
        local_steps=local_steps,
        evidence_cut_weight=evidence_cut_weight,
        duplicate_question_penalty=duplicate_question_penalty,
    )
    samples = []
    for sample in dataset.iter_samples():
        dialogue_key = f"{sample['question_type']}:{sample['dialogue_id']}"
        samples.append(
            {
                "sample_id": sample["sample_id"],
                "question_type": sample["question_type"],
                "dialogue_id": sample["dialogue_id"],
                "turn": sample["turn"],
                "split": result.assignments[dialogue_key],
            }
        )
    samples.sort(key=lambda value: value["sample_id"])
    manifest = _base_manifest(dataset, "P1-leakage-minimized-dialogue-split")
    manifest.update(
        {
            "seed": seed,
            "ratios": ratios or {"train": 0.70, "dev": 0.10, "test": 0.20},
            "objective": result.objective,
            "diagnostics": result.diagnostics,
            "dialogue_assignments": dict(sorted(result.assignments.items())),
            "samples": samples,
            "content_hash": stable_json_hash(
                {
                    "seed": seed,
                    "assignments": result.assignments,
                    "source_checksums": manifest["source_checksums"],
                }
            ),
        }
    )
    return manifest


def write_manifests(
    dataset: LongCoQADataset,
    output_dir: Path,
    **p1_kwargs: Any,
) -> tuple[Path, Path]:
    dataset.validate()
    p0 = build_p0_manifest(dataset)
    p1 = build_p1_manifest(dataset, **p1_kwargs)
    p0_path = output_dir / "p0_full.json"
    p1_path = output_dir / "p1_dialogue_split.json"
    atomic_write_json(p0_path, p0)
    atomic_write_json(p1_path, p1)
    return p0_path, p1_path
