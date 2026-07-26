from __future__ import annotations

import json
import math
import shutil
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from longcoqa.evaluation.p0_nonfactoid_formal import (
    TASKS,
    P0NonFactoidCell,
    aggregate_claim_recall,
    normalize_nonfactoid_row,
)
from longcoqa.paths import resolve_project_path
from longcoqa.utils.hashing import sha256_file, stable_json_hash
from longcoqa.utils.io import atomic_write_json, read_jsonl, read_yaml

MIGRATION_VERSION = "longcoqa-p0-claim-repair-migration-v2"
_FROZEN_CLAIM_SCHEMA_VERSION = "longcoqa-p0-nonfactoid-claim-v1"
_FROZEN_V2_REPAIR_POLICY = "longcoqa-p0-nonfactoid-claim-repair-policy-v2"


@dataclass(frozen=True)
class _CheckpointRewrite:
    path: Path
    value: dict[str, Any]


@dataclass(frozen=True)
class ClaimRepairMigrationPlan:
    config_path: Path
    output_root: Path
    expected_samples: int
    cell_ids: tuple[str, ...]
    ok_to_migrate: int
    errors_left_pending: int
    old_evaluator_hashes: tuple[str, ...]
    new_evaluator_identity: dict[str, Any]
    new_evaluator_hash: str
    rewrites: tuple[_CheckpointRewrite, ...]
    claim_complete_markers: tuple[Path, ...]
    matrix_complete_marker: Path
    matrix_complete_marker_present: bool
    source_files: tuple[Path, ...]
    source_hashes: Mapping[str, str]
    planning_input_hashes: Mapping[str, str]
    judge_hashes: Mapping[str, str]

    def as_report(self) -> dict[str, Any]:
        return {
            "schema_version": MIGRATION_VERSION,
            "status": "dry_run",
            "config": str(self.config_path),
            "output_root": str(self.output_root),
            "cell_count": len(self.cell_ids),
            "cells": list(self.cell_ids),
            "expected_samples_per_cell": self.expected_samples,
            "ok_to_migrate": self.ok_to_migrate,
            "errors_left_pending": self.errors_left_pending,
            "old_evaluator_hashes": list(self.old_evaluator_hashes),
            "new_evaluator_hash": self.new_evaluator_hash,
            "new_evaluator_identity": deepcopy(self.new_evaluator_identity),
            "intended_mutations": {
                "rewrite_ok_checkpoints": self.ok_to_migrate,
                "remove_claim_complete_markers": len(self.claim_complete_markers),
                "remove_matrix_root_complete_marker": self.matrix_complete_marker_present,
                "regenerate_summaries": False,
            },
        }


def _read_json_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Missing or invalid JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _load_migration_config(
    config: Path,
) -> tuple[Path, dict[str, Any], tuple[P0NonFactoidCell, ...]]:
    config_path = resolve_project_path(config)
    raw = read_yaml(config_path)
    section = raw.get("p0_nonfactoid_main")
    if not isinstance(section, dict):
        raise ValueError("Config is missing p0_nonfactoid_main mapping")
    if tuple(str(task) for task in section.get("tasks", ())) != TASKS:
        raise ValueError(f"p0_nonfactoid_main.tasks must be {TASKS}")
    if str(section.get("provider", "")).strip() != "qwen37":
        raise ValueError("Claim v2 migration requires provider qwen37")
    raw_cells = section.get("cells")
    if not isinstance(raw_cells, list) or not raw_cells:
        raise ValueError("p0_nonfactoid_main.cells must be a non-empty list")
    cells: list[P0NonFactoidCell] = []
    seen: set[str] = set()
    for index, value in enumerate(raw_cells):
        if not isinstance(value, dict):
            raise ValueError(f"cells[{index}] must be a mapping")
        cell_id = str(value.get("cell_id", "")).strip()
        input_path = str(value.get("input_path", "")).strip()
        expected_model = str(value.get("expected_model", "")).strip()
        protocol = str(value.get("protocol", "")).strip()
        if not all((cell_id, input_path, expected_model, protocol)):
            raise ValueError(f"cells[{index}] is missing a required value")
        if cell_id in seen:
            raise ValueError(f"Duplicate cell_id: {cell_id}")
        seen.add(cell_id)
        cells.append(
            P0NonFactoidCell(
                cell_id=cell_id,
                input_path=resolve_project_path(input_path),
                expected_model=expected_model,
                protocol=protocol,
            )
        )
    return config_path, section, tuple(cells)


def _compatible_v1_identity(provider: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "identity_schema_version": "longcoqa-p0-nonfactoid-evaluator-identity-v1",
        "task": "claim_recall",
        "model": "qwen3.7-max",
        "prompt_version": "longcoqa-p0-nonfactoid-claim-prompt-v1",
        "parser_version": "longcoqa-p0-nonfactoid-claim-parser-v1",
        "generation_config_version": "longcoqa-p0-nonfactoid-generation-config-v1",
        "generation_options": {
            "enable_thinking": False,
            "response_format": {"type": "json_object"},
        },
        "max_tokens": 1024,
        "provider": deepcopy(dict(provider)),
    }


def _frozen_v2_identity(provider: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "identity_schema_version": "longcoqa-p0-nonfactoid-evaluator-identity-v2",
        "task": "claim_recall",
        "model": "qwen3.7-max",
        "primary": {
            "prompt_version": "longcoqa-p0-nonfactoid-claim-prompt-v1",
            "parser_version": "longcoqa-p0-nonfactoid-claim-parser-v1",
            "generation_config_version": "longcoqa-p0-nonfactoid-generation-config-v1",
            "generation_options": {
                "enable_thinking": False,
                "response_format": {"type": "json_object"},
            },
            "max_tokens": 1024,
        },
        "repair": {
            "policy_version": _FROZEN_V2_REPAIR_POLICY,
            "prompt_version": "longcoqa-p0-nonfactoid-claim-repair-prompt-v1",
            "max_tokens": 4096,
            "max_attempts": 3,
        },
        "provider": deepcopy(dict(provider)),
    }


def _validate_claim_evaluation(evaluation: Any, *, sample_id: str, artifact: Path) -> None:
    if not isinstance(evaluation, Mapping):
        raise ValueError(f"Claim evaluation must be a mapping: {artifact}")
    if (
        evaluation.get("schema_version") != _FROZEN_CLAIM_SCHEMA_VERSION
        or evaluation.get("sample_id") != sample_id
    ):
        raise ValueError(f"Claim evaluation identity mismatch: {artifact}")
    references = evaluation.get("reference_claims")
    candidates = evaluation.get("candidate_claims")
    if (
        not isinstance(references, list)
        or not references
        or any(not isinstance(value, str) or not value.strip() for value in references)
    ):
        raise ValueError(f"Claim reference_claims are invalid: {artifact}")
    if not isinstance(candidates, list):
        raise ValueError(f"Claim candidate_claims are invalid: {artifact}")
    matched: set[int] = set()
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError(f"Claim candidate entry is invalid: {artifact}")
        text = candidate.get("text")
        reference = candidate.get("matched_reference")
        if not isinstance(text, str) or not text.strip() or "matched_reference" not in candidate:
            raise ValueError(f"Claim candidate entry is invalid: {artifact}")
        if reference is not None:
            if (
                isinstance(reference, bool)
                or not isinstance(reference, int)
                or not 0 <= reference < len(references)
            ):
                raise ValueError(f"Claim matched_reference is invalid: {artifact}")
            matched.add(reference)
    recall = evaluation.get("claim_recall")
    expected_recall = len(matched) / len(references)
    if (
        isinstance(recall, bool)
        or not isinstance(recall, (int, float))
        or not math.isfinite(float(recall))
        or not math.isclose(float(recall), expected_recall, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ValueError(f"Claim recall contribution is invalid: {artifact}")


def _validate_v1_checkpoint(
    checkpoint: dict[str, Any],
    *,
    path: Path,
    cell: P0NonFactoidCell,
    row: dict[str, Any],
) -> tuple[str, dict[str, Any], str]:
    sample_id = str(row["sample_id"])
    identity = checkpoint.get("evaluator_identity")
    if isinstance(identity, Mapping) and identity.get("identity_schema_version") == (
        "longcoqa-p0-nonfactoid-evaluator-identity-v2"
    ):
        raise ValueError(f"Checkpoint already uses the v2 evaluator identity: {path}")
    if not isinstance(identity, Mapping):
        raise ValueError(f"Checkpoint evaluator identity is invalid: {path}")
    provider = identity.get("provider")
    if not isinstance(provider, Mapping):
        raise ValueError(f"Checkpoint provider identity is invalid: {path}")
    expected_identity = _compatible_v1_identity(provider)
    expected_evaluator_hash = stable_json_hash(expected_identity)
    expected_fields = {
        "schema_version": "longcoqa-p0-rag-formal-checkpoint-v1",
        "cell_id": cell.cell_id,
        "task": "claim_recall",
        "sample_id": sample_id,
        "question_type": "non_factoid",
        "answer_model": cell.expected_model,
        "input_hash": stable_json_hash(row),
        "evaluator_hash": expected_evaluator_hash,
    }
    for field, expected in expected_fields.items():
        if checkpoint.get(field) != expected:
            raise ValueError(f"Checkpoint {field} mismatch: {path}")
    if dict(identity) != expected_identity:
        raise ValueError(f"Checkpoint is not an exact compatible v1 identity: {path}")
    status = checkpoint.get("status")
    if status == "ok":
        _validate_claim_evaluation(checkpoint.get("evaluation"), sample_id=sample_id, artifact=path)
    elif status != "error":
        raise ValueError(f"Non-ok checkpoint must have status=error: {path}")
    return str(status), expected_identity, expected_evaluator_hash


def _validate_claim_summary(
    directory: Path,
    *,
    cell_id: str,
    expected_samples: int,
    checkpoints: list[dict[str, Any]],
    v1_identity: dict[str, Any],
    v1_hash: str,
) -> tuple[dict[str, Any], Path | None]:
    summary_path = directory / "summary.json"
    summary = _read_json_mapping(summary_path)
    successful = sum(row["status"] == "ok" for row in checkpoints)
    failed = expected_samples - successful
    expected_fields = {
        "schema_version": "longcoqa-p0-rag-formal-summary-v1",
        "cell_id": cell_id,
        "task": "claim_recall",
        "requested_samples": expected_samples,
        "successful_samples": successful,
        "failed_samples": failed,
        "evaluator_hash": v1_hash,
        "evaluator_identity": v1_identity,
    }
    for field, expected in expected_fields.items():
        if summary.get(field) != expected:
            raise ValueError(f"Claim task summary {field} mismatch: {summary_path}")
    evaluations = [row["evaluation"] for row in checkpoints if row["status"] == "ok"]
    aggregate = dict(aggregate_claim_recall(evaluations))
    if not isinstance(summary.get("aggregate"), Mapping) or stable_json_hash(
        dict(summary["aggregate"])
    ) != stable_json_hash(aggregate):
        raise ValueError(f"Claim task summary aggregate mismatch: {summary_path}")
    marker = directory / ".complete"
    if marker.is_file():
        if stable_json_hash(_read_json_mapping(marker)) != stable_json_hash(summary):
            raise ValueError(f"Claim task marker/summary mismatch: {directory}")
        return summary, marker
    if failed == 0:
        raise ValueError(f"Complete Claim task is missing marker: {directory}")
    return summary, None


def _validate_matrix_summary(
    output_root: Path,
    *,
    expected_samples: int,
    claim_states: Mapping[str, tuple[int, int, str]],
) -> None:
    path = output_root / "matrix_summary.json"
    summary = _read_json_mapping(path)
    if summary.get("expected_samples_per_cell") != expected_samples:
        raise ValueError(f"Matrix summary expected sample count mismatch: {path}")
    task_cells = summary.get("task_cells")
    if not isinstance(task_cells, list):
        raise ValueError(f"Matrix summary task_cells are invalid: {path}")
    claim_entries: dict[str, Mapping[str, Any]] = {}
    for entry in task_cells:
        if isinstance(entry, Mapping) and entry.get("task") == "claim_recall":
            cell_id = str(entry.get("cell_id", ""))
            if not cell_id or cell_id in claim_entries:
                raise ValueError(f"Matrix summary Claim cells are duplicated: {path}")
            claim_entries[cell_id] = entry
    if set(claim_entries) != set(claim_states):
        raise ValueError(f"Matrix summary Claim cells mismatch: {path}")
    for cell_id, (successful, failed, evaluator_hash) in claim_states.items():
        entry = claim_entries[cell_id]
        status = "complete" if failed == 0 else "incomplete"
        expected_hash = evaluator_hash if failed == 0 else None
        if (
            entry.get("status") != status
            or entry.get("successful_samples") != successful
            or entry.get("failed_samples") != failed
            or entry.get("evaluator_hash") != expected_hash
        ):
            raise ValueError(f"Matrix summary Claim state mismatch for {cell_id}: {path}")
    root_marker = output_root / ".complete"
    if root_marker.is_file() and stable_json_hash(_read_json_mapping(root_marker)) != stable_json_hash(
        summary
    ):
        raise ValueError(f"Matrix root marker/summary mismatch: {output_root}")


def _file_hashes(paths: tuple[Path, ...], *, relative_to: Path) -> dict[str, str]:
    return {str(path.relative_to(relative_to)): sha256_file(path) for path in paths}


def _judge_hashes(output_root: Path) -> dict[str, str]:
    judge_root = output_root / "judge_100"
    if not judge_root.is_dir():
        raise ValueError(f"Judge tree is missing: {judge_root}")
    files = tuple(sorted(path for path in judge_root.rglob("*") if path.is_file()))
    if not files:
        raise ValueError(f"Judge tree contains no artifacts: {judge_root}")
    return _file_hashes(files, relative_to=judge_root)


def plan_claim_recall_v2_migration(config: Path) -> ClaimRepairMigrationPlan:
    config_path, section, cells = _load_migration_config(config)
    expected_samples = int(section.get("expected_nonfactoid_samples", 0))
    expected_source_samples = int(section.get("expected_source_samples", 0))
    if expected_samples <= 0 or expected_source_samples <= 0:
        raise ValueError("Configured source/non-factoid sample counts must be positive")
    output_root = resolve_project_path(str(section.get("output_root", "")))
    if not output_root.is_dir():
        raise ValueError(f"Migration output root is missing: {output_root}")

    rewrites: list[_CheckpointRewrite] = []
    markers: list[Path] = []
    old_hashes: set[str] = set()
    new_identities: dict[str, dict[str, Any]] = {}
    errors = 0
    claim_states: dict[str, tuple[int, int, str]] = {}
    locked_sample_ids: tuple[str, ...] | None = None

    for cell in cells:
        raw_rows = read_jsonl(cell.input_path)
        if len(raw_rows) != expected_source_samples:
            raise ValueError(f"{cell.cell_id} source row count mismatch")
        rows = [
            normalize_nonfactoid_row(row, cell)
            for row in raw_rows
            if row.get("question_type") == "non_factoid"
        ]
        sample_ids = [str(row["sample_id"]) for row in rows]
        if len(rows) != expected_samples or len(sample_ids) != len(set(sample_ids)):
            raise ValueError(f"{cell.cell_id} input sample set is partial or duplicated")
        ordered_sample_ids = tuple(sorted(sample_ids))
        if locked_sample_ids is None:
            locked_sample_ids = ordered_sample_ids
        elif ordered_sample_ids != locked_sample_ids:
            raise ValueError(f"{cell.cell_id} has cross-cell sample-ID drift")
        directory = output_root / "claim_recall" / cell.cell_id / "full"
        checkpoint_dir = directory / "checkpoints"
        checkpoint_paths = tuple(sorted(path for path in checkpoint_dir.glob("*.json") if path.is_file()))
        expected_paths = {checkpoint_dir / f"{stable_json_hash(sample_id)}.json" for sample_id in sample_ids}
        if len(checkpoint_paths) != expected_samples or set(checkpoint_paths) != expected_paths:
            raise ValueError(f"{cell.cell_id} checkpoint sample set is partial or incompatible")
        checkpoints: list[dict[str, Any]] = []
        cell_identity: dict[str, Any] | None = None
        cell_hash: str | None = None
        successful = 0
        for row, checkpoint_path in zip(
            rows,
            (checkpoint_dir / f"{stable_json_hash(sample_id)}.json" for sample_id in sample_ids),
            strict=True,
        ):
            checkpoint = _read_json_mapping(checkpoint_path)
            status, v1_identity, v1_hash = _validate_v1_checkpoint(
                checkpoint, path=checkpoint_path, cell=cell, row=row
            )
            if cell_identity is None:
                cell_identity, cell_hash = v1_identity, v1_hash
            elif cell_identity != v1_identity or cell_hash != v1_hash:
                raise ValueError(f"{cell.cell_id} contains mixed v1 evaluator identities")
            old_hashes.add(v1_hash)
            new_identity = _frozen_v2_identity(v1_identity["provider"])
            new_identities[stable_json_hash(new_identity)] = new_identity
            checkpoints.append(checkpoint)
            if status == "ok":
                successful += 1
                rewritten = deepcopy(checkpoint)
                rewritten["evaluator_hash"] = stable_json_hash(new_identity)
                rewritten["evaluator_identity"] = deepcopy(new_identity)
                rewritten_evaluation = rewritten["evaluation"]
                assert isinstance(rewritten_evaluation, dict)
                rewritten_evaluation["repair_audit"] = {
                    "policy_version": _FROZEN_V2_REPAIR_POLICY,
                    "repair_applied": False,
                    "successful_attempt": 1,
                    "primary_validation_error": None,
                }
                rewritten["migration_provenance"] = {
                    "migration_version": MIGRATION_VERSION,
                    "source_evaluator_hash": v1_hash,
                    "source_evaluator_identity": deepcopy(v1_identity),
                }
                rewrites.append(_CheckpointRewrite(checkpoint_path, rewritten))
            else:
                errors += 1
        assert cell_identity is not None and cell_hash is not None
        results = read_jsonl(directory / "results.jsonl")
        if results != checkpoints:
            raise ValueError(f"Claim results/checkpoints mismatch: {directory}")
        _summary, marker = _validate_claim_summary(
            directory,
            cell_id=cell.cell_id,
            expected_samples=expected_samples,
            checkpoints=checkpoints,
            v1_identity=cell_identity,
            v1_hash=cell_hash,
        )
        if marker is not None:
            markers.append(marker)
        claim_states[cell.cell_id] = (successful, expected_samples - successful, cell_hash)

    if len(new_identities) != 1:
        raise ValueError("All Claim cells must resolve to one uniform v2 evaluator hash")
    _validate_matrix_summary(output_root, expected_samples=expected_samples, claim_states=claim_states)
    new_hash, new_identity = next(iter(new_identities.items()))
    claim_root = output_root / "claim_recall"
    claim_files = tuple(sorted(path for path in claim_root.rglob("*") if path.is_file()))
    matrix_files = tuple(
        path for path in (output_root / "matrix_summary.json", output_root / ".complete") if path.is_file()
    )
    source_files = claim_files + matrix_files
    return ClaimRepairMigrationPlan(
        config_path=config_path,
        output_root=output_root,
        expected_samples=expected_samples,
        cell_ids=tuple(cell.cell_id for cell in cells),
        ok_to_migrate=len(rewrites),
        errors_left_pending=errors,
        old_evaluator_hashes=tuple(sorted(old_hashes)),
        new_evaluator_identity=deepcopy(new_identity),
        new_evaluator_hash=new_hash,
        rewrites=tuple(rewrites),
        claim_complete_markers=tuple(markers),
        matrix_complete_marker=output_root / ".complete",
        matrix_complete_marker_present=(output_root / ".complete").is_file(),
        source_files=source_files,
        source_hashes=_file_hashes(source_files, relative_to=output_root),
        planning_input_hashes={
            str(path): sha256_file(path)
            for path in (config_path, *(cell.input_path for cell in cells))
        },
        judge_hashes=_judge_hashes(output_root),
    )


def _assert_plan_is_current(plan: ClaimRepairMigrationPlan) -> None:
    current_planning_inputs: dict[str, str] = {}
    for raw_path in plan.planning_input_hashes:
        path = Path(raw_path)
        if not path.is_file():
            raise ValueError(
                "Migration input changed after planning; no backup or write performed"
            )
        current_planning_inputs[raw_path] = sha256_file(path)
    if current_planning_inputs != dict(plan.planning_input_hashes):
        raise ValueError(
            "Migration input changed after planning; no backup or write performed"
        )
    claim_root = plan.output_root / "claim_recall"
    current_claim_files = tuple(sorted(path for path in claim_root.rglob("*") if path.is_file()))
    current_matrix_files = tuple(
        path
        for path in (
            plan.output_root / "matrix_summary.json",
            plan.output_root / ".complete",
        )
        if path.is_file()
    )
    current_files = current_claim_files + current_matrix_files
    current_hashes = _file_hashes(current_files, relative_to=plan.output_root)
    if current_hashes != dict(plan.source_hashes):
        raise ValueError("Migration source changed after planning; no backup or write performed")
    if _judge_hashes(plan.output_root) != dict(plan.judge_hashes):
        raise ValueError("Judge tree changed after planning; no backup or write performed")


def _create_snapshot(plan: ClaimRepairMigrationPlan, backup_root: Path) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    backup = backup_root / f"claim-v2-{stamp}-{uuid4().hex[:8]}"
    backup.mkdir(parents=True, exist_ok=False)
    for source in plan.source_files:
        relative = source.relative_to(plan.output_root)
        if relative.parts[0] == "claim_recall":
            target = backup / relative
        else:
            target = backup / "matrix" / relative.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    atomic_write_json(
        backup / "snapshot_manifest.json",
        {
            "schema_version": MIGRATION_VERSION,
            "source_root": str(plan.output_root),
            "source_hashes": dict(plan.source_hashes),
            "matrix_root_complete_existed": plan.matrix_complete_marker.is_file(),
        },
    )
    return backup


def execute_claim_recall_v2_migration(plan: ClaimRepairMigrationPlan, backup_root: Path) -> dict[str, Any]:
    backup_root = resolve_project_path(backup_root)
    try:
        backup_root.relative_to(plan.output_root)
    except ValueError:
        pass
    else:
        raise ValueError("Backup root must be outside the migration output root")
    _assert_plan_is_current(plan)
    backup = _create_snapshot(plan, backup_root)
    _assert_plan_is_current(plan)
    judge_before = _judge_hashes(plan.output_root)
    for rewrite in plan.rewrites:
        atomic_write_json(rewrite.path, rewrite.value)
    for marker in plan.claim_complete_markers:
        marker.unlink(missing_ok=True)
    plan.matrix_complete_marker.unlink(missing_ok=True)
    judge_after = _judge_hashes(plan.output_root)
    if judge_after != judge_before:
        raise RuntimeError("Judge artifacts changed during Claim migration")
    report = plan.as_report()
    report.update(
        {
            "status": "executed",
            "migrated_ok_checkpoints": plan.ok_to_migrate,
            "backup_path": str(backup),
            "judge_hashes_before": judge_before,
            "judge_hashes_after": judge_after,
        }
    )
    report_path = backup / "migration_report.json"
    report["report_path"] = str(report_path)
    atomic_write_json(report_path, report)
    return report
