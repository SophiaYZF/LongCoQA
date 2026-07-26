from __future__ import annotations

import json
import math
import shutil
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from longcoqa.evaluation.p0_nonfactoid_formal import (
    CLAIM_RECALL_SCHEMA_VERSION,
    TASKS,
    P0NonFactoidCell,
    aggregate_claim_recall,
    aggregate_judge_100,
    build_claim_recall_evaluator_identity,
    load_p0_nonfactoid_config,
    normalize_nonfactoid_row,
)
from longcoqa.paths import resolve_project_path
from longcoqa.utils.hashing import sha256_file, stable_json_hash
from longcoqa.utils.io import atomic_write_json, read_jsonl, read_yaml

MIGRATION_VERSION = "longcoqa-p0-claim-repair-migration-v3"
V1_EVALUATOR_HASH = "44579355eaf1a658d96bbfd7b0c97c682f38d3bed898155a7358f083004ddaa7"
V2_EVALUATOR_HASH = "0077462ab48127be65f73d5f56b11d864a02a50dca69ae6f93c640c9be3487d3"
V3_EVALUATOR_HASH = "9b354d2aca09ca10c2c75ffd30da721cd18f2112f0216646a11f7b37663fc216"
V2_REPAIR_POLICY = "longcoqa-p0-nonfactoid-claim-repair-policy-v2"
V3_REPAIR_POLICY = "longcoqa-p0-nonfactoid-claim-repair-policy-v3"


@dataclass(frozen=True)
class ClaimRepairV3ExpectedState:
    cell_count: int
    source_rows_per_cell: int
    selected_rows_per_cell: int
    ok_checkpoints: int
    errors: int
    v1_errors: int
    v2_errors: int
    v2_error_locations: tuple[tuple[str, str], ...]
    no_repair_successes: int
    actual_v2_repairs: int
    claim_complete_markers: int
    matrix_root_complete_present: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "cell_count": self.cell_count,
            "source_rows_per_cell": self.source_rows_per_cell,
            "selected_rows_per_cell": self.selected_rows_per_cell,
            "ok_checkpoints": self.ok_checkpoints,
            "errors": self.errors,
            "v1_errors": self.v1_errors,
            "v2_errors": self.v2_errors,
            "v2_error_locations": [
                {"cell_id": cell_id, "sample_id": sample_id}
                for cell_id, sample_id in self.v2_error_locations
            ],
            "no_repair_successes": self.no_repair_successes,
            "actual_v2_repairs": self.actual_v2_repairs,
            "claim_complete_markers": self.claim_complete_markers,
            "matrix_root_complete_present": self.matrix_root_complete_present,
        }


PRODUCTION_EXPECTED_STATE = ClaimRepairV3ExpectedState(
    cell_count=14,
    source_rows_per_cell=3019,
    selected_rows_per_cell=1847,
    ok_checkpoints=25850,
    errors=8,
    v1_errors=7,
    v2_errors=1,
    v2_error_locations=(
        ("qwen3_5_9b/g3_aligned_raw", "non_factoid:81:6"),
    ),
    no_repair_successes=25848,
    actual_v2_repairs=2,
    claim_complete_markers=3,
    matrix_root_complete_present=False,
)


@dataclass(frozen=True)
class _CheckpointRewrite:
    path: Path
    value: dict[str, Any]


@dataclass(frozen=True)
class ClaimRepairV3MigrationPlan:
    expected_state: ClaimRepairV3ExpectedState
    config_path: Path
    output_root: Path
    expected_samples: int
    cell_ids: tuple[str, ...]
    ok_to_migrate: int
    errors_left_pending: int
    v1_errors_left_pending: int
    v2_errors_left_pending: int
    pending_errors: tuple[dict[str, Any], ...]
    actual_v2_repairs_preserved: int
    source_evaluator_identity: dict[str, Any]
    source_evaluator_hash: str
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
            "expected_state": self.expected_state.as_dict(),
            "config": str(self.config_path),
            "output_root": str(self.output_root),
            "cell_count": len(self.cell_ids),
            "cells": list(self.cell_ids),
            "expected_samples_per_cell": self.expected_samples,
            "ok_to_migrate": self.ok_to_migrate,
            "errors_left_pending": self.errors_left_pending,
            "v1_errors_left_pending": self.v1_errors_left_pending,
            "v2_errors_left_pending": self.v2_errors_left_pending,
            "pending_errors": deepcopy(list(self.pending_errors)),
            "actual_v2_repairs_preserved": self.actual_v2_repairs_preserved,
            "source_evaluator_hash": self.source_evaluator_hash,
            "source_evaluator_identity": deepcopy(self.source_evaluator_identity),
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
    loaded_section, cells = load_p0_nonfactoid_config(config_path)
    if loaded_section != section:
        raise ValueError("P0 Non-factoid config changed while loading")
    if tuple(str(task) for task in section.get("tasks", ())) != TASKS:
        raise ValueError(f"p0_nonfactoid_main.tasks must be {TASKS}")
    if str(section.get("provider", "")).strip() != "qwen37":
        raise ValueError("Claim v3 migration requires provider qwen37")
    return config_path, section, cells


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


def _compatible_v2_identity(provider: Mapping[str, Any]) -> dict[str, Any]:
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
            "policy_version": V2_REPAIR_POLICY,
            "prompt_version": "longcoqa-p0-nonfactoid-claim-repair-prompt-v1",
            "max_tokens": 4096,
            "max_attempts": 3,
        },
        "provider": deepcopy(dict(provider)),
    }


def _recognized_identity(
    identity: Any, evaluator_hash: Any, *, artifact: Path
) -> tuple[int, dict[str, Any]]:
    if not isinstance(identity, Mapping):
        raise ValueError(f"Evaluator identity is invalid: {artifact}")
    provider = identity.get("provider")
    if not isinstance(provider, Mapping):
        raise ValueError(f"Evaluator provider identity is invalid: {artifact}")
    candidates = (
        (1, _compatible_v1_identity(provider), V1_EVALUATOR_HASH),
        (2, _compatible_v2_identity(provider), V2_EVALUATOR_HASH),
    )
    for version, expected, expected_hash in candidates:
        if dict(identity) == expected and evaluator_hash == expected_hash:
            if stable_json_hash(expected) != expected_hash:
                raise ValueError(f"Evaluator identity does not match the locked hash: {artifact}")
            return version, expected
    raise ValueError(
        f"Artifact is not an exact compatible v2 identity or allowed v1 error: {artifact}"
    )


def _validate_claim_evaluation(
    evaluation: Any, *, sample_id: str, artifact: Path, require_v2_audit: bool
) -> tuple[dict[str, Any], bool]:
    if not isinstance(evaluation, Mapping):
        raise ValueError(f"Claim evaluation must be a mapping: {artifact}")
    expected_keys = {
        "schema_version",
        "sample_id",
        "reference_claims",
        "candidate_claims",
        "claim_recall",
    }
    if require_v2_audit:
        expected_keys.add("repair_audit")
    if set(evaluation) != expected_keys:
        raise ValueError(f"Claim evaluation is not a strict mapping: {artifact}")
    if (
        evaluation.get("schema_version") != CLAIM_RECALL_SCHEMA_VERSION
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
        if not isinstance(candidate, Mapping) or set(candidate) != {
            "text",
            "matched_reference",
        }:
            raise ValueError(f"Claim candidate entry is invalid: {artifact}")
        text = candidate.get("text")
        reference = candidate.get("matched_reference")
        if not isinstance(text, str) or not text.strip():
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

    repaired = False
    if require_v2_audit:
        audit = evaluation.get("repair_audit")
        if not isinstance(audit, Mapping) or set(audit) != {
            "policy_version",
            "repair_applied",
            "successful_attempt",
            "primary_validation_error",
        }:
            raise ValueError(f"Claim repair audit is invalid: {artifact}")
        repaired_value = audit.get("repair_applied")
        attempt = audit.get("successful_attempt")
        primary_error = audit.get("primary_validation_error")
        if audit.get("policy_version") != V2_REPAIR_POLICY or type(repaired_value) is not bool:
            raise ValueError(f"Claim repair audit is not exact v2: {artifact}")
        if type(attempt) is not int or not 1 <= attempt <= 4:
            raise ValueError(f"Claim repair audit attempt is invalid: {artifact}")
        repaired = bool(repaired_value)
        if repaired:
            if attempt <= 1 or not isinstance(primary_error, str) or not primary_error:
                raise ValueError(f"Claim repaired audit is inconsistent: {artifact}")
        elif attempt != 1 or primary_error is not None:
            raise ValueError(f"Claim no-repair audit is inconsistent: {artifact}")
    return deepcopy(dict(evaluation)), repaired


def _validate_checkpoint_common(
    checkpoint: Mapping[str, Any],
    *,
    path: Path,
    cell: P0NonFactoidCell,
    row: Mapping[str, Any],
) -> None:
    expected = {
        "schema_version": "longcoqa-p0-rag-formal-checkpoint-v1",
        "cell_id": cell.cell_id,
        "task": "claim_recall",
        "sample_id": str(row["sample_id"]),
        "question_type": "non_factoid",
        "answer_model": cell.expected_model,
        "input_hash": stable_json_hash(dict(row)),
    }
    for field, value in expected.items():
        if checkpoint.get(field) != value:
            raise ValueError(f"Checkpoint {field} mismatch: {path}")


def _rewrite_ok_checkpoint(
    checkpoint: dict[str, Any],
    *,
    path: Path,
    new_identity: Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
    identity = checkpoint.get("evaluator_identity")
    if isinstance(identity, Mapping) and identity.get("identity_schema_version") == (
        "longcoqa-p0-nonfactoid-evaluator-identity-v3"
    ):
        raise ValueError(f"Checkpoint already uses the v3 evaluator identity: {path}")
    version, source_identity = _recognized_identity(
        identity, checkpoint.get("evaluator_hash"), artifact=path
    )
    if version != 2:
        raise ValueError(f"Successful checkpoint is not an exact compatible v2 identity: {path}")
    evaluation, repaired = _validate_claim_evaluation(
        checkpoint.get("evaluation"),
        sample_id=str(checkpoint["sample_id"]),
        artifact=path,
        require_v2_audit=True,
    )
    rewritten = deepcopy(checkpoint)
    rewritten["evaluator_hash"] = V3_EVALUATOR_HASH
    rewritten["evaluator_identity"] = deepcopy(dict(new_identity))
    if not repaired:
        audit = evaluation["repair_audit"]
        assert isinstance(audit, dict)
        audit["policy_version"] = V3_REPAIR_POLICY
    rewritten["evaluation"] = evaluation
    provenance: dict[str, Any] = {
        "migration_version": MIGRATION_VERSION,
        "source_evaluator_hash": V2_EVALUATOR_HASH,
        "source_evaluator_identity": deepcopy(source_identity),
    }
    if "migration_provenance" in checkpoint:
        provenance["prior_provenance"] = deepcopy(checkpoint["migration_provenance"])
    rewritten["migration_provenance"] = provenance
    return rewritten, repaired


def _validate_error_checkpoint(checkpoint: Mapping[str, Any], *, path: Path) -> int:
    identity = checkpoint.get("evaluator_identity")
    if isinstance(identity, Mapping) and identity.get("identity_schema_version") == (
        "longcoqa-p0-nonfactoid-evaluator-identity-v3"
    ):
        raise ValueError(f"Checkpoint already uses the v3 evaluator identity: {path}")
    version, _expected_identity = _recognized_identity(
        identity, checkpoint.get("evaluator_hash"), artifact=path
    )
    if checkpoint.get("status") != "error":
        raise ValueError(f"Non-ok checkpoint must have status=error: {path}")
    return version


def _strip_retry_and_migration_fields(value: Mapping[str, Any]) -> dict[str, Any]:
    stripped = deepcopy(dict(value))
    for field in (
        "evaluator_identity",
        "evaluator_hash",
        "migration_provenance",
        "elapsed_seconds",
        "error",
        "error_type",
    ):
        stripped.pop(field, None)
    evaluation = stripped.get("evaluation")
    if isinstance(evaluation, dict):
        evaluation.pop("repair_audit", None)
    return stripped


def _validate_results_against_checkpoints(
    results_path: Path,
    *,
    checkpoints: Sequence[dict[str, Any]],
    completed: bool,
) -> tuple[list[dict[str, Any]], int]:
    try:
        results = read_jsonl(results_path)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Claim results are invalid: {results_path}") from error
    if len(results) != len(checkpoints):
        raise ValueError(f"Claim result count mismatch: {results_path}")
    result_ids = [str(result.get("sample_id", "")) for result in results]
    checkpoint_ids = [str(checkpoint["sample_id"]) for checkpoint in checkpoints]
    if result_ids != checkpoint_ids or len(result_ids) != len(set(result_ids)):
        raise ValueError(f"Claim result sample-ID drift: {results_path}")
    versions: set[int] = set()
    for result, checkpoint in zip(results, checkpoints, strict=True):
        version, _identity = _recognized_identity(
            result.get("evaluator_identity"), result.get("evaluator_hash"), artifact=results_path
        )
        versions.add(version)
        if _strip_retry_and_migration_fields(result) != _strip_retry_and_migration_fields(
            checkpoint
        ):
            raise ValueError(f"Claim results/checkpoints semantic drift: {results_path}")
        if result.get("status") == "ok":
            _validate_claim_evaluation(
                result.get("evaluation"),
                sample_id=str(result["sample_id"]),
                artifact=results_path,
                require_v2_audit=version == 2,
            )
        elif result.get("status") != "error":
            raise ValueError(f"Claim result status is invalid: {results_path}")
    if len(versions) != 1:
        raise ValueError(f"Claim results contain mixed evaluator identities: {results_path}")
    version = next(iter(versions))
    if completed and (version != 2 or results != list(checkpoints)):
        raise ValueError(f"Completed Claim results are not current v2: {results_path}")
    return results, version


def _validate_claim_summary(
    directory: Path,
    *,
    cell_id: str,
    expected_samples: int,
    checkpoints: Sequence[dict[str, Any]],
) -> Path | None:
    marker_path = directory / ".complete"
    completed = marker_path.is_file()
    results, result_version = _validate_results_against_checkpoints(
        directory / "results.jsonl", checkpoints=checkpoints, completed=completed
    )
    summary_path = directory / "summary.json"
    summary = _read_json_mapping(summary_path)
    summary_version, summary_identity = _recognized_identity(
        summary.get("evaluator_identity"), summary.get("evaluator_hash"), artifact=summary_path
    )
    if summary_version != result_version:
        raise ValueError(f"Claim summary/results evaluator identity mismatch: {summary_path}")
    successful = sum(checkpoint["status"] == "ok" for checkpoint in checkpoints)
    expected = {
        "schema_version": "longcoqa-p0-rag-formal-summary-v1",
        "cell_id": cell_id,
        "task": "claim_recall",
        "requested_samples": expected_samples,
        "successful_samples": successful,
        "failed_samples": expected_samples - successful,
        "evaluator_hash": V2_EVALUATOR_HASH if summary_version == 2 else V1_EVALUATOR_HASH,
        "evaluator_identity": summary_identity,
    }
    for field, value in expected.items():
        if summary.get(field) != value:
            raise ValueError(f"Claim task summary {field} mismatch: {summary_path}")
    evaluations = [checkpoint["evaluation"] for checkpoint in checkpoints if checkpoint["status"] == "ok"]
    aggregate = dict(aggregate_claim_recall(evaluations))
    if not isinstance(summary.get("aggregate"), Mapping) or dict(summary["aggregate"]) != aggregate:
        raise ValueError(f"Claim task summary aggregate mismatch: {summary_path}")
    if completed:
        if summary_version != 2 or stable_json_hash(_read_json_mapping(marker_path)) != stable_json_hash(
            summary
        ):
            raise ValueError(f"Completed Claim cell has invalid current v2 marker: {directory}")
        if successful != expected_samples:
            raise ValueError(f"Completed Claim cell contains failed checkpoints: {directory}")
        return marker_path
    return None


def _validate_complete_judge_cell(
    directory: Path, *, cell_id: str, expected_samples: int, sample_ids: tuple[str, ...]
) -> tuple[str, int]:
    summary_path = directory / "summary.json"
    marker_path = directory / ".complete"
    try:
        summary = _read_json_mapping(summary_path)
        marker = _read_json_mapping(marker_path)
    except ValueError as error:
        raise ValueError(f"Judge cell is not complete: {directory}") from error
    if stable_json_hash(summary) != stable_json_hash(marker):
        raise ValueError(f"Judge cell marker/summary is not complete: {directory}")
    evaluator_identity = summary.get("evaluator_identity")
    evaluator_hash = summary.get("evaluator_hash")
    if (
        not isinstance(evaluator_identity, Mapping)
        or not isinstance(evaluator_hash, str)
        or stable_json_hash(dict(evaluator_identity)) != evaluator_hash
    ):
        raise ValueError(f"Judge cell evaluator identity is invalid: {directory}")
    expected = {
        "schema_version": "longcoqa-p0-rag-formal-summary-v1",
        "cell_id": cell_id,
        "task": "judge_100",
        "requested_samples": expected_samples,
        "successful_samples": expected_samples,
        "failed_samples": 0,
    }
    for field, value in expected.items():
        if summary.get(field) != value:
            raise ValueError(f"Judge cell is not complete: {directory}")
    try:
        results = read_jsonl(directory / "results.jsonl")
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Judge cell results are invalid: {directory}") from error
    observed_sample_ids = tuple(
        sorted(str(row.get("sample_id", "")) for row in results)
    )
    if len(results) != expected_samples or observed_sample_ids != sample_ids:
        raise ValueError(f"Judge cell sample set is invalid: {directory}")
    for result in results:
        if (
            result.get("status") != "ok"
            or result.get("cell_id") != cell_id
            or result.get("task") != "judge_100"
            or result.get("evaluator_hash") != evaluator_hash
            or result.get("evaluator_identity") != evaluator_identity
            or not isinstance(result.get("evaluation"), Mapping)
        ):
            raise ValueError(f"Judge cell result identity/status is invalid: {directory}")
    aggregate = dict(aggregate_judge_100([result["evaluation"] for result in results]))
    if not isinstance(summary.get("aggregate"), Mapping) or dict(summary["aggregate"]) != aggregate:
        raise ValueError(f"Judge cell aggregate is invalid: {directory}")
    return evaluator_hash, expected_samples


def _validate_matrix_summary(
    output_root: Path,
    *,
    cells: Sequence[P0NonFactoidCell],
    expected_samples: int,
    sample_ids: tuple[str, ...],
    claim_states: Mapping[str, tuple[int, int, bool]],
) -> None:
    path = output_root / "matrix_summary.json"
    summary = _read_json_mapping(path)
    if (output_root / ".complete").is_file():
        if stable_json_hash(_read_json_mapping(output_root / ".complete")) != stable_json_hash(summary):
            raise ValueError(f"Matrix root marker/summary mismatch: {output_root}")
    judge_states: dict[str, tuple[str, int]] = {}
    for cell in cells:
        judge_states[cell.cell_id] = _validate_complete_judge_cell(
            output_root / "judge_100" / cell.cell_id / "full",
            cell_id=cell.cell_id,
            expected_samples=expected_samples,
            sample_ids=sample_ids,
        )
    expected_header = {
        "schema_version": "longcoqa-p0-nonfactoid-matrix-v1",
        "status": "incomplete",
        "expected_task_cells": len(cells) * len(TASKS),
        "completed_task_cells": len(cells)
        + sum(completed for _successful, _failed, completed in claim_states.values()),
        "expected_samples_per_cell": expected_samples,
        "successful_samples": len(cells) * expected_samples
        + sum(successful for successful, _failed, _completed in claim_states.values()),
        "failed_samples": sum(failed for _successful, failed, _completed in claim_states.values()),
        "locked_sample_hash": stable_json_hash(list(sample_ids)),
    }
    for field, value in expected_header.items():
        if summary.get(field) != value:
            raise ValueError(f"Matrix summary {field} mismatch: {path}")
    entries = summary.get("task_cells")
    if not isinstance(entries, list) or len(entries) != len(cells) * len(TASKS):
        raise ValueError(f"Matrix summary task_cells are invalid: {path}")
    indexed: dict[tuple[str, str], Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError(f"Matrix summary task cell is invalid: {path}")
        key = (str(entry.get("cell_id", "")), str(entry.get("task", "")))
        if key in indexed:
            raise ValueError(f"Matrix summary task cells are duplicated: {path}")
        indexed[key] = entry
    for cell in cells:
        successful, failed, completed = claim_states[cell.cell_id]
        claim_entry = indexed.get((cell.cell_id, "claim_recall"))
        judge_entry = indexed.get((cell.cell_id, "judge_100"))
        if claim_entry is None or judge_entry is None:
            raise ValueError(f"Matrix summary cell is missing: {cell.cell_id}")
        claim_expected = {
            "cell_id": cell.cell_id,
            "task": "claim_recall",
            "status": "complete" if completed else "incomplete",
            "output_dir": str(output_root / "claim_recall" / cell.cell_id / "full"),
            "evaluator_hash": V2_EVALUATOR_HASH if completed else None,
            "successful_samples": successful,
            "failed_samples": failed,
        }
        if any(claim_entry.get(field) != value for field, value in claim_expected.items()):
            raise ValueError(f"Matrix summary Claim state mismatch for {cell.cell_id}: {path}")
        if not completed and not isinstance(claim_entry.get("error"), str):
            raise ValueError(f"Matrix summary incomplete Claim error is missing: {path}")
        judge_hash, judge_successful = judge_states[cell.cell_id]
        judge_expected = {
            "cell_id": cell.cell_id,
            "task": "judge_100",
            "status": "complete",
            "output_dir": str(output_root / "judge_100" / cell.cell_id / "full"),
            "evaluator_hash": judge_hash,
            "successful_samples": judge_successful,
            "failed_samples": 0,
        }
        if any(judge_entry.get(field) != value for field, value in judge_expected.items()):
            raise ValueError(f"Matrix summary Judge state mismatch for {cell.cell_id}: {path}")


def _file_hashes(paths: Sequence[Path], *, relative_to: Path) -> dict[str, str]:
    return {str(path.relative_to(relative_to)): sha256_file(path) for path in paths}


def _judge_hashes(output_root: Path) -> dict[str, str]:
    judge_root = output_root / "judge_100"
    if not judge_root.is_dir():
        raise ValueError(f"Judge tree is missing: {judge_root}")
    files = tuple(sorted(path for path in judge_root.rglob("*") if path.is_file()))
    if not files:
        raise ValueError(f"Judge tree contains no artifacts: {judge_root}")
    return _file_hashes(files, relative_to=judge_root)


def _require_expected_state(field: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise ValueError(
            f"Claim v3 expected-state mismatch for {field}: "
            f"expected {expected!r}, found {observed!r}"
        )


def plan_claim_recall_v3_migration(
    config: Path,
    *,
    expected_state: ClaimRepairV3ExpectedState = PRODUCTION_EXPECTED_STATE,
) -> ClaimRepairV3MigrationPlan:
    config_path, section, cells = _load_migration_config(config)
    expected_samples = int(section.get("expected_nonfactoid_samples", 0))
    expected_source_samples = int(section.get("expected_source_samples", 0))
    _require_expected_state("cell_count", len(cells), expected_state.cell_count)
    _require_expected_state(
        "source_rows_per_cell",
        expected_source_samples,
        expected_state.source_rows_per_cell,
    )
    _require_expected_state(
        "selected_rows_per_cell",
        expected_samples,
        expected_state.selected_rows_per_cell,
    )
    if expected_samples <= 0 or expected_source_samples <= 0:
        raise ValueError("Configured source/non-factoid sample counts must be positive")
    output_root = resolve_project_path(str(section.get("output_root", "")))
    if not output_root.is_dir():
        raise ValueError(f"Migration output root is missing: {output_root}")

    rewrites: list[_CheckpointRewrite] = []
    markers: list[Path] = []
    errors = 0
    error_versions = {1: 0, 2: 0}
    pending_errors: list[dict[str, Any]] = []
    repaired_count = 0
    claim_states: dict[str, tuple[int, int, bool]] = {}
    locked_sample_ids: tuple[str, ...] | None = None
    source_identity: dict[str, Any] | None = None
    target_identity: dict[str, Any] | None = None

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
        ordered_paths = tuple(
            checkpoint_dir / f"{stable_json_hash(sample_id)}.json" for sample_id in sample_ids
        )
        if len(checkpoint_paths) != expected_samples or set(checkpoint_paths) != set(ordered_paths):
            raise ValueError(f"{cell.cell_id} checkpoint sample set is partial or incompatible")

        checkpoints: list[dict[str, Any]] = []
        successful = 0
        for row, checkpoint_path in zip(rows, ordered_paths, strict=True):
            checkpoint = _read_json_mapping(checkpoint_path)
            _validate_checkpoint_common(
                checkpoint, path=checkpoint_path, cell=cell, row=row
            )
            status = checkpoint.get("status")
            if status == "ok":
                identity = checkpoint.get("evaluator_identity")
                if isinstance(identity, Mapping):
                    provider = identity.get("provider")
                else:
                    provider = None
                if not isinstance(provider, Mapping):
                    raise ValueError(f"Checkpoint provider identity is invalid: {checkpoint_path}")
                expected_source_identity = _compatible_v2_identity(provider)
                if source_identity is None:
                    source_identity = expected_source_identity
                    target_identity = build_claim_recall_evaluator_identity(provider)
                    if stable_json_hash(target_identity) != V3_EVALUATOR_HASH:
                        raise ValueError("Approved v3 evaluator identity/hash mismatch")
                elif expected_source_identity != source_identity:
                    raise ValueError("Claim checkpoints contain mixed v2 evaluator identities")
                assert target_identity is not None
                rewritten, repaired = _rewrite_ok_checkpoint(
                    checkpoint,
                    path=checkpoint_path,
                    new_identity=target_identity,
                )
                rewrites.append(_CheckpointRewrite(checkpoint_path, rewritten))
                successful += 1
                repaired_count += int(repaired)
            else:
                error_version = _validate_error_checkpoint(
                    checkpoint, path=checkpoint_path
                )
                error_versions[error_version] += 1
                errors += 1
                pending_errors.append(
                    {
                        "cell_id": cell.cell_id,
                        "sample_id": str(checkpoint["sample_id"]),
                        "identity_version": f"v{error_version}",
                        "evaluator_hash": str(checkpoint["evaluator_hash"]),
                    }
                )
            checkpoints.append(checkpoint)
        marker = _validate_claim_summary(
            directory,
            cell_id=cell.cell_id,
            expected_samples=expected_samples,
            checkpoints=checkpoints,
        )
        if marker is not None:
            markers.append(marker)
        claim_states[cell.cell_id] = (
            successful,
            expected_samples - successful,
            marker is not None,
        )

    assert locked_sample_ids is not None and source_identity is not None and target_identity is not None
    _validate_matrix_summary(
        output_root,
        cells=cells,
        expected_samples=expected_samples,
        sample_ids=locked_sample_ids,
        claim_states=claim_states,
    )
    observed_v2_error_locations = tuple(
        (str(error["cell_id"]), str(error["sample_id"]))
        for error in pending_errors
        if error["identity_version"] == "v2"
    )
    observed_no_repairs = len(rewrites) - repaired_count
    root_marker_present = (output_root / ".complete").is_file()
    observed_values = {
        "ok_checkpoints": len(rewrites),
        "errors": errors,
        "v1_errors": error_versions[1],
        "v2_errors": error_versions[2],
        "v2_error_locations": observed_v2_error_locations,
        "no_repair_successes": observed_no_repairs,
        "actual_v2_repairs": repaired_count,
        "claim_complete_markers": len(markers),
        "matrix_root_complete_present": root_marker_present,
    }
    expected_values = {
        "ok_checkpoints": expected_state.ok_checkpoints,
        "errors": expected_state.errors,
        "v1_errors": expected_state.v1_errors,
        "v2_errors": expected_state.v2_errors,
        "v2_error_locations": expected_state.v2_error_locations,
        "no_repair_successes": expected_state.no_repair_successes,
        "actual_v2_repairs": expected_state.actual_v2_repairs,
        "claim_complete_markers": expected_state.claim_complete_markers,
        "matrix_root_complete_present": expected_state.matrix_root_complete_present,
    }
    for field, observed in observed_values.items():
        _require_expected_state(field, observed, expected_values[field])
    claim_root = output_root / "claim_recall"
    claim_files = tuple(sorted(path for path in claim_root.rglob("*") if path.is_file()))
    matrix_files = tuple(
        path
        for path in (output_root / "matrix_summary.json", output_root / ".complete")
        if path.is_file()
    )
    source_files = claim_files + matrix_files
    return ClaimRepairV3MigrationPlan(
        expected_state=expected_state,
        config_path=config_path,
        output_root=output_root,
        expected_samples=expected_samples,
        cell_ids=tuple(cell.cell_id for cell in cells),
        ok_to_migrate=len(rewrites),
        errors_left_pending=errors,
        v1_errors_left_pending=error_versions[1],
        v2_errors_left_pending=error_versions[2],
        pending_errors=tuple(pending_errors),
        actual_v2_repairs_preserved=repaired_count,
        source_evaluator_identity=deepcopy(source_identity),
        source_evaluator_hash=V2_EVALUATOR_HASH,
        new_evaluator_identity=deepcopy(target_identity),
        new_evaluator_hash=V3_EVALUATOR_HASH,
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


def _assert_plan_is_current(plan: ClaimRepairV3MigrationPlan) -> None:
    current_inputs: dict[str, str] = {}
    for raw_path in plan.planning_input_hashes:
        path = Path(raw_path)
        if not path.is_file():
            raise ValueError("Migration input changed after planning; no backup or write performed")
        current_inputs[raw_path] = sha256_file(path)
    if current_inputs != dict(plan.planning_input_hashes):
        raise ValueError("Migration input changed after planning; no backup or write performed")
    claim_root = plan.output_root / "claim_recall"
    claim_files = tuple(sorted(path for path in claim_root.rglob("*") if path.is_file()))
    matrix_files = tuple(
        path
        for path in (plan.output_root / "matrix_summary.json", plan.output_root / ".complete")
        if path.is_file()
    )
    current_files = claim_files + matrix_files
    if _file_hashes(current_files, relative_to=plan.output_root) != dict(plan.source_hashes):
        raise ValueError("Migration source changed after planning; no backup or write performed")
    if _judge_hashes(plan.output_root) != dict(plan.judge_hashes):
        raise ValueError("Judge tree changed after planning; no backup or write performed")


def _create_snapshot(plan: ClaimRepairV3MigrationPlan, backup_root: Path) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    backup = backup_root / f"claim-v3-{stamp}-{uuid4().hex[:8]}"
    backup.mkdir(parents=True, exist_ok=False)
    for source in plan.source_files:
        relative = source.relative_to(plan.output_root)
        target = (
            backup / relative
            if relative.parts[0] == "claim_recall"
            else backup / "matrix" / relative.name
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    atomic_write_json(
        backup / "snapshot_manifest.json",
        {
            "schema_version": MIGRATION_VERSION,
            "source_root": str(plan.output_root),
            "source_hashes": dict(plan.source_hashes),
            "planning_input_hashes": dict(plan.planning_input_hashes),
            "judge_hashes": dict(plan.judge_hashes),
            "matrix_root_complete_existed": plan.matrix_complete_marker_present,
        },
    )
    return backup


def execute_claim_recall_v3_migration(
    plan: ClaimRepairV3MigrationPlan, backup_root: Path
) -> dict[str, Any]:
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
