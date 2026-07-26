from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

import longcoqa.cli as cli_module
from longcoqa.evaluation.p0_claim_repair_v3_migration import (
    ClaimRepairV3ExpectedState,
    execute_claim_recall_v3_migration,
    plan_claim_recall_v3_migration,
)
from longcoqa.utils.hashing import sha256_file, stable_json_hash

MIGRATE_ENV = "LONGCOQA_P0_NONFACTOID_MIGRATE_V3"
V1_HASH = "44579355eaf1a658d96bbfd7b0c97c682f38d3bed898155a7358f083004ddaa7"
V2_HASH = "0077462ab48127be65f73d5f56b11d864a02a50dca69ae6f93c640c9be3487d3"
V3_HASH = "9b354d2aca09ca10c2c75ffd30da721cd18f2112f0216646a11f7b37663fc216"
V2_POLICY = "longcoqa-p0-nonfactoid-claim-repair-policy-v2"
V3_POLICY = "longcoqa-p0-nonfactoid-claim-repair-policy-v3"

CELL_SPECS = (
    ("qwen3_8b/z1_teacher_forced", "Qwen3-8B", "z1_teacher_forced"),
    ("qwen3_8b/g3_aligned_raw", "Qwen3-8B", "g3_aligned_raw"),
    ("qwen3_5_9b/z1_teacher_forced", "Qwen3.5-9B", "z1_teacher_forced"),
    ("qwen3_5_9b/g3_aligned_raw", "Qwen3.5-9B", "g3_aligned_raw"),
    ("qwen3_6_27b/z1_teacher_forced", "Qwen3.6-27B", "z1_teacher_forced"),
    ("qwen3_6_27b/g3_aligned_raw", "Qwen3.6-27B", "g3_aligned_raw"),
    ("glm4_7_flash/z1_teacher_forced", "GLM-4.7-Flash", "z1_teacher_forced"),
    ("glm4_7_flash/g3_aligned_raw", "GLM-4.7-Flash", "g3_aligned_raw"),
    ("qwen3_5_35b_a3b/z1_teacher_forced", "Qwen3.5-35B-A3B", "z1_teacher_forced"),
    ("qwen3_5_35b_a3b/g3_aligned_raw", "Qwen3.5-35B-A3B", "g3_aligned_raw"),
    ("gpt_oss_120b/z1_teacher_forced", "gpt-oss-120b", "z1_teacher_forced"),
    ("gpt_oss_120b/g3_aligned_raw", "gpt-oss-120b", "g3_aligned_raw"),
    ("qwen37/z1_teacher_forced", "qwen3.7-max", "z1_teacher_forced"),
    ("qwen37/g3_aligned_raw", "qwen3.7-max", "g3_aligned_raw"),
)

FIXTURE_EXPECTED_STATE = ClaimRepairV3ExpectedState(
    cell_count=14,
    source_rows_per_cell=2,
    selected_rows_per_cell=2,
    ok_checkpoints=26,
    errors=2,
    v1_errors=1,
    v2_errors=1,
    v2_error_locations=((CELL_SPECS[3][0], "non_factoid:1:2"),),
    no_repair_successes=25,
    actual_v2_repairs=1,
    claim_complete_markers=3,
    matrix_root_complete_present=False,
)


def _plan_fixture(
    config: Path,
    expected_state: ClaimRepairV3ExpectedState = FIXTURE_EXPECTED_STATE,
):
    return plan_claim_recall_v3_migration(config, expected_state=expected_state)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _provider() -> dict[str, object]:
    return {
        "api_key_count": 2,
        "base_url_configured": True,
        "concurrency": 4,
        "max_retries": 8,
        "model": "qwen3.7-max",
        "name": "qwen37",
        "protocol": "openai",
        "requests_per_minute": 30,
        "timeout_seconds": 180.0,
    }


def _v1_identity() -> dict[str, object]:
    identity = {
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
        "provider": _provider(),
    }
    assert stable_json_hash(identity) == V1_HASH
    return identity


def _v2_identity() -> dict[str, object]:
    identity = {
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
            "policy_version": V2_POLICY,
            "prompt_version": "longcoqa-p0-nonfactoid-claim-repair-prompt-v1",
            "max_tokens": 4096,
            "max_attempts": 3,
        },
        "provider": _provider(),
    }
    assert stable_json_hash(identity) == V2_HASH
    return identity


def _v3_identity() -> dict[str, object]:
    identity = deepcopy(_v2_identity())
    identity["identity_schema_version"] = "longcoqa-p0-nonfactoid-evaluator-identity-v3"
    repair = identity["repair"]
    assert isinstance(repair, dict)
    repair["policy_version"] = V3_POLICY
    repair["prompt_version"] = "longcoqa-p0-nonfactoid-claim-repair-prompt-v2"
    assert stable_json_hash(identity) == V3_HASH
    return identity


def _input_row(sample_id: str, *, model: str, protocol: str) -> dict[str, object]:
    return {
        "status": "ok",
        "sample_id": sample_id,
        "dialogue_id": sample_id.split(":")[1],
        "turn": int(sample_id.split(":")[2]),
        "question_type": "non_factoid",
        "question": f"question {sample_id}",
        "reference": "alpha; beta",
        "prediction": "alpha",
        "model": model,
        "protocol": protocol,
    }


def _normalized_input(row: dict[str, object]) -> dict[str, object]:
    return {
        "sample_id": row["sample_id"],
        "dialogue_id": row["dialogue_id"],
        "turn": row["turn"],
        "question_type": "non_factoid",
        "question": row["question"],
        "reference": row["reference"],
        "prediction": row["prediction"],
        "model": row["model"],
        "protocol": row["protocol"],
        "answerable": True,
    }


def _audit(*, repaired: bool) -> dict[str, object]:
    return {
        "policy_version": V2_POLICY,
        "repair_applied": repaired,
        "successful_attempt": 2 if repaired else 1,
        "primary_validation_error": "missing candidate_claims" if repaired else None,
    }


def _evaluation(sample_id: str, *, repaired: bool = False) -> dict[str, object]:
    return {
        "schema_version": "longcoqa-p0-nonfactoid-claim-v1",
        "sample_id": sample_id,
        "reference_claims": ["alpha", "beta"],
        "candidate_claims": [{"text": "alpha", "matched_reference": 0}],
        "claim_recall": 0.5,
        "repair_audit": _audit(repaired=repaired),
    }


def _v1_provenance() -> dict[str, object]:
    return {
        "migration_version": "longcoqa-p0-claim-repair-migration-v2",
        "source_evaluator_hash": V1_HASH,
        "source_evaluator_identity": _v1_identity(),
    }


def _checkpoint(
    row: dict[str, object],
    *,
    cell_id: str,
    status: str,
    identity_version: int,
    repaired: bool = False,
) -> dict[str, object]:
    identity = _v2_identity() if identity_version == 2 else _v1_identity()
    value: dict[str, object] = {
        "schema_version": "longcoqa-p0-rag-formal-checkpoint-v1",
        "status": status,
        "cell_id": cell_id,
        "task": "claim_recall",
        "sample_id": row["sample_id"],
        "question_type": "non_factoid",
        "answer_model": row["model"],
        "input_hash": stable_json_hash(_normalized_input(row)),
        "evaluator_hash": V2_HASH if identity_version == 2 else V1_HASH,
        "evaluator_identity": identity,
        "elapsed_seconds": 0.25,
    }
    if status == "ok":
        value["evaluation"] = _evaluation(str(row["sample_id"]), repaired=repaired)
        value["migration_provenance"] = _v1_provenance()
    else:
        value.update({"error_type": "ValueError", "error": "strict parse failed"})
    return value


def _stale_v1_result(checkpoint: dict[str, object]) -> dict[str, object]:
    stale = deepcopy(checkpoint)
    stale["evaluator_hash"] = V1_HASH
    stale["evaluator_identity"] = _v1_identity()
    stale["elapsed_seconds"] = 99.0
    stale.pop("migration_provenance", None)
    if stale["status"] == "ok":
        evaluation = stale["evaluation"]
        assert isinstance(evaluation, dict)
        evaluation.pop("repair_audit")
    else:
        stale["error_type"] = "RuntimeError"
        stale["error"] = "old retry failure text"
    return stale


def _summary(
    *, cell_id: str, results: list[dict[str, object]], identity_version: int
) -> dict[str, object]:
    successful = sum(result["status"] == "ok" for result in results)
    identity = _v2_identity() if identity_version == 2 else _v1_identity()
    return {
        "schema_version": "longcoqa-p0-rag-formal-summary-v1",
        "cell_id": cell_id,
        "task": "claim_recall",
        "requested_samples": 2,
        "successful_samples": successful,
        "failed_samples": 2 - successful,
        "reused_samples": successful,
        "evaluator_hash": V2_HASH if identity_version == 2 else V1_HASH,
        "evaluator_identity": identity,
        "aggregate": {"samples": successful, "claim_recall": 0.5 if successful else None},
    }


def _judge_identity() -> dict[str, object]:
    return {
        "identity_schema_version": "judge-fixture-v1",
        "task": "judge_100",
        "model": "qwen3.7-max",
    }


def _write_complete_judge_cell(output_root: Path, cell_id: str) -> dict[str, object]:
    directory = output_root / "judge_100" / cell_id / "full"
    identity = _judge_identity()
    evaluator_hash = stable_json_hash(identity)
    results = [
        {
            "schema_version": "longcoqa-p0-rag-formal-checkpoint-v1",
            "status": "ok",
            "cell_id": cell_id,
            "task": "judge_100",
            "sample_id": sample_id,
            "evaluator_hash": evaluator_hash,
            "evaluator_identity": identity,
            "evaluation": {
                "judgment_type": "answer_quality",
                "correctness": 3,
            },
        }
        for sample_id in ("non_factoid:1:1", "non_factoid:1:2")
    ]
    _write_jsonl(directory / "results.jsonl", results)
    summary = {
        "schema_version": "longcoqa-p0-rag-formal-summary-v1",
        "cell_id": cell_id,
        "task": "judge_100",
        "requested_samples": 2,
        "successful_samples": 2,
        "failed_samples": 0,
        "reused_samples": 2,
        "evaluator_hash": evaluator_hash,
        "evaluator_identity": identity,
        "aggregate": {"samples": 2, "mean_correctness": 3.0, "judge_100": 50.0},
    }
    _write_json(directory / "summary.json", summary)
    _write_json(directory / ".complete", summary)
    return {
        "cell_id": cell_id,
        "task": "judge_100",
        "status": "complete",
        "output_dir": str(directory),
        "evaluator_hash": evaluator_hash,
        "successful_samples": 2,
        "failed_samples": 0,
    }


def _build_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    output_root = tmp_path / "matrix"
    input_root = tmp_path / "inputs"
    sample_ids = ("non_factoid:1:1", "non_factoid:1:2")
    config_cells: list[dict[str, object]] = []
    matrix_cells: list[dict[str, object]] = []
    error_paths: list[Path] = []
    repaired_path: Path | None = None

    for index, (cell_id, model, protocol) in enumerate(CELL_SPECS):
        input_path = input_root / f"cell-{index}.jsonl"
        input_rows = [
            _input_row(sample_id, model=model, protocol=protocol) for sample_id in sample_ids
        ]
        _write_jsonl(input_path, input_rows)
        config_cells.append(
            {
                "cell_id": cell_id,
                "input_path": str(input_path),
                "expected_model": model,
                "protocol": protocol,
            }
        )

        statuses = ("ok", "error") if index in (3, 4) else ("ok", "ok")
        checkpoints: list[dict[str, object]] = []
        for sample_index, (row, status) in enumerate(zip(input_rows, statuses, strict=True)):
            identity_version = 1 if index == 4 and status == "error" else 2
            repaired = index == 0 and sample_index == 0
            checkpoint = _checkpoint(
                row,
                cell_id=cell_id,
                status=status,
                identity_version=identity_version,
                repaired=repaired,
            )
            checkpoint_path = (
                output_root
                / "claim_recall"
                / cell_id
                / "full"
                / "checkpoints"
                / f"{stable_json_hash(row['sample_id'])}.json"
            )
            _write_json(checkpoint_path, checkpoint)
            checkpoints.append(checkpoint)
            if status == "error":
                error_paths.append(checkpoint_path)
            if repaired:
                repaired_path = checkpoint_path

        directory = output_root / "claim_recall" / cell_id / "full"
        is_completed = index < 3
        current_incomplete = index == 3
        if is_completed or current_incomplete:
            results = deepcopy(checkpoints)
            summary_identity_version = 2
        else:
            results = [_stale_v1_result(checkpoint) for checkpoint in checkpoints]
            summary_identity_version = 1
        _write_jsonl(directory / "results.jsonl", results)
        summary = _summary(
            cell_id=cell_id,
            results=results,
            identity_version=summary_identity_version,
        )
        _write_json(directory / "summary.json", summary)
        if is_completed:
            _write_json(directory / ".complete", summary)

        successful = sum(checkpoint["status"] == "ok" for checkpoint in checkpoints)
        matrix_cells.append(
            {
                "cell_id": cell_id,
                "task": "claim_recall",
                "status": "complete" if is_completed else "incomplete",
                "output_dir": str(directory),
                "evaluator_hash": V2_HASH if is_completed else None,
                "successful_samples": successful,
                "failed_samples": 2 - successful,
                **({} if is_completed else {"error": "known incomplete Claim state"}),
            }
        )
        matrix_cells.append(_write_complete_judge_cell(output_root, cell_id))

    matrix_summary = {
        "schema_version": "longcoqa-p0-nonfactoid-matrix-v1",
        "status": "incomplete",
        "expected_task_cells": 28,
        "completed_task_cells": 17,
        "expected_samples_per_cell": 2,
        "successful_samples": 54,
        "failed_samples": 2,
        "task_cells": matrix_cells,
        "locked_sample_hash": stable_json_hash(list(sample_ids)),
    }
    _write_json(output_root / "matrix_summary.json", matrix_summary)

    config = {
        "p0_nonfactoid_main": {
            "provider": "qwen37",
            "expected_source_samples": 2,
            "expected_nonfactoid_samples": 2,
            "tasks": ["claim_recall", "judge_100"],
            "output_root": str(output_root),
            "cache_path": str(tmp_path / "never-open.sqlite3"),
            "cells": config_cells,
        }
    }
    config_path = tmp_path / "fixture.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    assert repaired_path is not None
    return config_path, output_root, {
        "errors": tuple(error_paths),
        "repaired": repaired_path,
        "inputs": tuple(Path(cell["input_path"]) for cell in config_cells),
    }


def test_production_default_rejects_a_compact_but_self_consistent_tree(
    tmp_path: Path,
) -> None:
    config, _output_root, _artifacts = _build_fixture(tmp_path)

    with pytest.raises(ValueError, match="expected-state.*source_rows_per_cell"):
        plan_claim_recall_v3_migration(config)


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("cell_count", 13),
        ("source_rows_per_cell", 3),
        ("selected_rows_per_cell", 3),
        ("ok_checkpoints", 25),
        ("errors", 3),
        ("v1_errors", 2),
        ("v2_errors", 2),
        ("no_repair_successes", 24),
        ("actual_v2_repairs", 2),
        ("claim_complete_markers", 4),
    ],
)
def test_expected_state_gate_rejects_every_count_drift(
    tmp_path: Path, field: str, wrong_value: int
) -> None:
    config, _output_root, _artifacts = _build_fixture(tmp_path)
    contract = replace(FIXTURE_EXPECTED_STATE, **{field: wrong_value})

    with pytest.raises(ValueError, match=rf"expected-state.*{field}"):
        _plan_fixture(config, contract)


def test_expected_state_gate_rejects_v2_error_location_drift(tmp_path: Path) -> None:
    config, _output_root, _artifacts = _build_fixture(tmp_path)
    contract = replace(
        FIXTURE_EXPECTED_STATE,
        v2_error_locations=((CELL_SPECS[3][0], "non_factoid:wrong"),),
    )

    with pytest.raises(ValueError, match="expected-state.*v2_error_locations"):
        _plan_fixture(config, contract)


@pytest.mark.parametrize(
    ("config_field", "expected_field"),
    [
        ("expected_source_samples", "source_rows_per_cell"),
        ("expected_nonfactoid_samples", "selected_rows_per_cell"),
    ],
)
def test_expected_state_gate_rejects_configured_row_count_drift(
    tmp_path: Path, config_field: str, expected_field: str
) -> None:
    config, _output_root, _artifacts = _build_fixture(tmp_path)
    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    raw["p0_nonfactoid_main"][config_field] = 3
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=rf"expected-state.*{expected_field}"):
        _plan_fixture(config)


def test_expected_state_gate_rejects_matrix_root_marker_presence(tmp_path: Path) -> None:
    config, output_root, _artifacts = _build_fixture(tmp_path)
    (output_root / ".complete").write_bytes(
        (output_root / "matrix_summary.json").read_bytes()
    )

    with pytest.raises(
        ValueError, match="expected-state.*matrix_root_complete_present"
    ):
        _plan_fixture(config)


def test_plan_is_provider_free_validates_current_state_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, output_root, _artifacts = _build_fixture(tmp_path)
    before = _tree_hashes(tmp_path)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("provider construction is forbidden during migration")

    monkeypatch.setattr(cli_module, "load_provider_settings", forbidden)
    monkeypatch.setattr(cli_module, "ProviderClient", forbidden)
    monkeypatch.setattr(cli_module, "ResponseCache", forbidden)
    plan = _plan_fixture(config)
    report = plan.as_report()

    assert report["cell_count"] == 14
    assert report["expected_samples_per_cell"] == 2
    assert report["ok_to_migrate"] == 26
    assert report["errors_left_pending"] == 2
    assert report["v1_errors_left_pending"] == 1
    assert report["v2_errors_left_pending"] == 1
    assert report["pending_errors"] == [
        {
            "cell_id": CELL_SPECS[3][0],
            "sample_id": "non_factoid:1:2",
            "identity_version": "v2",
            "evaluator_hash": V2_HASH,
        },
        {
            "cell_id": CELL_SPECS[4][0],
            "sample_id": "non_factoid:1:2",
            "identity_version": "v1",
            "evaluator_hash": V1_HASH,
        },
    ]
    assert report["actual_v2_repairs_preserved"] == 1
    assert report["source_evaluator_hash"] == V2_HASH
    assert report["new_evaluator_hash"] == V3_HASH
    assert report["new_evaluator_identity"] == _v3_identity()
    assert report["intended_mutations"] == {
        "rewrite_ok_checkpoints": 26,
        "remove_claim_complete_markers": 3,
        "remove_matrix_root_complete_marker": False,
        "regenerate_summaries": False,
    }
    assert _tree_hashes(tmp_path) == before
    assert not (output_root / "migration_report.json").exists()


def test_execute_snapshots_then_selectively_rewrites_and_preserves_judge(
    tmp_path: Path,
) -> None:
    config, output_root, artifacts = _build_fixture(tmp_path)
    plan = _plan_fixture(config)
    error_bytes = {path: path.read_bytes() for path in artifacts["errors"]}
    repaired_path = artifacts["repaired"]
    assert isinstance(repaired_path, Path)
    repaired_before = repaired_path.read_bytes()
    repaired_audit = deepcopy(_read_json(repaired_path)["evaluation"]["repair_audit"])
    judge_before = _tree_hashes(output_root / "judge_100")
    stale_cell = output_root / "claim_recall" / CELL_SPECS[5][0] / "full"
    stale_results_before = (stale_cell / "results.jsonl").read_bytes()
    stale_summary_before = (stale_cell / "summary.json").read_bytes()

    report = execute_claim_recall_v3_migration(plan, tmp_path / "backups")

    assert report["status"] == "executed"
    assert report["migrated_ok_checkpoints"] == 26
    assert report["errors_left_pending"] == 2
    assert report["judge_hashes_before"] == judge_before
    assert report["judge_hashes_after"] == judge_before
    backup = Path(report["backup_path"])
    assert backup.name.startswith("claim-v3-")
    assert (backup / "claim_recall" / CELL_SPECS[0][0] / "full" / "summary.json").is_file()
    assert (backup / "matrix" / "matrix_summary.json").is_file()
    assert (backup / "snapshot_manifest.json").is_file()
    assert (backup / "migration_report.json").is_file()
    assert (backup / repaired_path.relative_to(output_root)).read_bytes() == repaired_before

    for path in output_root.glob("claim_recall/**/checkpoints/*.json"):
        value = _read_json(path)
        if value["status"] == "error":
            continue
        assert value["evaluator_hash"] == V3_HASH
        assert value["evaluator_identity"] == _v3_identity()
        evaluation = value["evaluation"]
        assert isinstance(evaluation, dict)
        audit = evaluation["repair_audit"]
        assert isinstance(audit, dict)
        if path == repaired_path:
            assert audit == repaired_audit
        else:
            assert audit == {
                "policy_version": V3_POLICY,
                "repair_applied": False,
                "successful_attempt": 1,
                "primary_validation_error": None,
            }
        assert value["migration_provenance"] == {
            "migration_version": "longcoqa-p0-claim-repair-migration-v3",
            "source_evaluator_hash": V2_HASH,
            "source_evaluator_identity": _v2_identity(),
            "prior_provenance": _v1_provenance(),
        }

    assert {path: path.read_bytes() for path in artifacts["errors"]} == error_bytes
    assert not any(output_root.glob("claim_recall/**/.complete"))
    assert not (output_root / ".complete").exists()
    assert _tree_hashes(output_root / "judge_100") == judge_before
    assert (stale_cell / "results.jsonl").read_bytes() == stale_results_before
    assert (stale_cell / "summary.json").read_bytes() == stale_summary_before


@pytest.mark.parametrize("drift", ["checkpoint_sample", "checkpoint_input"])
def test_plan_rejects_checkpoint_sample_or_input_drift_without_writing(
    tmp_path: Path, drift: str
) -> None:
    config, output_root, _artifacts = _build_fixture(tmp_path)
    checkpoint_path = next(output_root.glob("claim_recall/**/checkpoints/*.json"))
    checkpoint = _read_json(checkpoint_path)
    checkpoint["sample_id" if drift == "checkpoint_sample" else "input_hash"] = "drift"
    _write_json(checkpoint_path, checkpoint)
    before = _tree_hashes(tmp_path)

    with pytest.raises(ValueError, match="sample_id|input_hash"):
        _plan_fixture(config)

    assert _tree_hashes(tmp_path) == before


def test_plan_rejects_unknown_checkpoint_identity_without_writing(tmp_path: Path) -> None:
    config, output_root, _artifacts = _build_fixture(tmp_path)
    checkpoint_path = next(output_root.glob("claim_recall/**/checkpoints/*.json"))
    checkpoint = _read_json(checkpoint_path)
    identity = checkpoint["evaluator_identity"]
    assert isinstance(identity, dict)
    identity["unexpected"] = True
    checkpoint["evaluator_hash"] = stable_json_hash(identity)
    _write_json(checkpoint_path, checkpoint)
    before = _tree_hashes(tmp_path)

    with pytest.raises(ValueError, match="exact compatible v2 identity"):
        _plan_fixture(config)

    assert _tree_hashes(tmp_path) == before


def test_plan_rejects_semantic_drift_in_stale_results_without_writing(tmp_path: Path) -> None:
    config, output_root, _artifacts = _build_fixture(tmp_path)
    results_path = output_root / "claim_recall" / CELL_SPECS[5][0] / "full" / "results.jsonl"
    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["evaluation"]["candidate_claims"][0]["matched_reference"] = 1
    _write_jsonl(results_path, rows)
    before = _tree_hashes(tmp_path)

    with pytest.raises(ValueError, match="semantic drift"):
        _plan_fixture(config)

    assert _tree_hashes(tmp_path) == before


def test_execute_rejects_input_drift_before_backup_or_output_mutation(
    tmp_path: Path,
) -> None:
    config, output_root, artifacts = _build_fixture(tmp_path)
    plan = _plan_fixture(config)
    input_path = artifacts["inputs"][0]
    assert isinstance(input_path, Path)
    input_path.write_bytes(input_path.read_bytes() + b"\n")
    before = _tree_hashes(tmp_path)

    with pytest.raises(ValueError, match="changed after planning"):
        execute_claim_recall_v3_migration(plan, tmp_path / "backups")

    assert _tree_hashes(tmp_path) == before
    assert not (tmp_path / "backups").exists()


def test_execute_rejects_judge_drift_before_backup_or_claim_mutation(
    tmp_path: Path,
) -> None:
    config, output_root, _artifacts = _build_fixture(tmp_path)
    plan = _plan_fixture(config)
    judge_path = output_root / "judge_100" / CELL_SPECS[0][0] / "full" / "summary.json"
    judge_path.write_bytes(judge_path.read_bytes() + b" ")
    before = _tree_hashes(tmp_path)

    with pytest.raises(ValueError, match="Judge tree changed after planning"):
        execute_claim_recall_v3_migration(plan, tmp_path / "backups")

    assert _tree_hashes(tmp_path) == before
    assert not (tmp_path / "backups").exists()


def test_plan_rejects_incomplete_judge_cell_and_matrix_drift(tmp_path: Path) -> None:
    config, output_root, _artifacts = _build_fixture(tmp_path)
    marker = output_root / "judge_100" / CELL_SPECS[0][0] / "full" / ".complete"
    marker.unlink()
    before = _tree_hashes(tmp_path)

    with pytest.raises(ValueError, match="Judge.*complete|matrix summary"):
        _plan_fixture(config)

    assert _tree_hashes(tmp_path) == before


def test_repeat_execute_and_already_v3_tree_are_rejected_without_second_backup(
    tmp_path: Path,
) -> None:
    config, _output_root, _artifacts = _build_fixture(tmp_path)
    backup_root = tmp_path / "backups"
    plan = _plan_fixture(config)
    execute_claim_recall_v3_migration(plan, backup_root)
    backups = tuple(backup_root.iterdir())

    with pytest.raises(ValueError, match="changed after planning"):
        execute_claim_recall_v3_migration(plan, backup_root)
    assert tuple(backup_root.iterdir()) == backups

    with pytest.raises(ValueError, match="already uses the v3 evaluator identity"):
        _plan_fixture(config)
    assert tuple(backup_root.iterdir()) == backups


def test_execute_snapshots_and_removes_a_matrix_root_marker(tmp_path: Path) -> None:
    config, output_root, _artifacts = _build_fixture(tmp_path)
    root_marker = output_root / ".complete"
    root_marker.write_bytes((output_root / "matrix_summary.json").read_bytes())
    marker_before = root_marker.read_bytes()
    plan = _plan_fixture(
        config,
        replace(FIXTURE_EXPECTED_STATE, matrix_root_complete_present=True),
    )

    report = execute_claim_recall_v3_migration(plan, tmp_path / "backups")

    assert report["intended_mutations"]["remove_matrix_root_complete_marker"] is True
    assert not root_marker.exists()
    backup = Path(report["backup_path"])
    assert (backup / "matrix" / ".complete").read_bytes() == marker_before


def test_cli_dry_run_has_zero_writes_no_provider_and_execute_gate_precedes_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, output_root, _artifacts = _build_fixture(tmp_path)
    backup_root = tmp_path / "backups"
    before = _tree_hashes(output_root)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("provider construction is forbidden during migration")

    monkeypatch.setattr(
        cli_module,
        "plan_claim_recall_v3_migration",
        lambda path: _plan_fixture(path),
    )
    monkeypatch.setattr(cli_module, "load_provider_settings", forbidden)
    monkeypatch.setattr(cli_module, "ProviderClient", forbidden)
    monkeypatch.setattr(cli_module, "ResponseCache", forbidden)
    dry_run = CliRunner().invoke(
        cli_module.app,
        [
            "migrate-p0-nonfactoid-claim-v3",
            "--config",
            str(config),
            "--backup-root",
            str(backup_root),
        ],
    )
    assert dry_run.exit_code == 0, dry_run.output
    assert '"status": "dry_run"' in dry_run.output
    assert V3_HASH in dry_run.output
    assert _tree_hashes(output_root) == before
    assert not backup_root.exists()

    monkeypatch.delenv(MIGRATE_ENV, raising=False)
    denied = CliRunner().invoke(
        cli_module.app,
        [
            "migrate-p0-nonfactoid-claim-v3",
            "--config",
            str(tmp_path / "missing.yaml"),
            "--backup-root",
            str(backup_root),
            "--execute",
        ],
    )
    assert denied.exit_code != 0
    assert MIGRATE_ENV in denied.output
    assert "Missing or invalid" not in denied.output
    assert _tree_hashes(output_root) == before
    assert not backup_root.exists()


def test_cli_does_not_expose_an_expected_state_override() -> None:
    result = CliRunner().invoke(
        cli_module.app,
        ["migrate-p0-nonfactoid-claim-v3", "--help"],
    )

    assert result.exit_code == 0, result.output
    assert "expected-state" not in result.output
