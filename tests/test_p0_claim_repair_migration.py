from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

import longcoqa.cli as cli_module
from longcoqa.evaluation.p0_claim_repair_migration import (
    execute_claim_recall_v2_migration,
    plan_claim_recall_v2_migration,
)
from longcoqa.utils.hashing import sha256_file, stable_json_hash

MIGRATE_ENV = "LONGCOQA_P0_NONFACTOID_MIGRATE_V2"
V2_HASH = "0077462ab48127be65f73d5f56b11d864a02a50dca69ae6f93c640c9be3487d3"
V2_POLICY = "longcoqa-p0-nonfactoid-claim-repair-policy-v2"


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


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path) for path in sorted(root.rglob("*")) if path.is_file()
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
        "provider": _provider(),
    }


def _v2_identity() -> dict[str, object]:
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
            "policy_version": V2_POLICY,
            "prompt_version": "longcoqa-p0-nonfactoid-claim-repair-prompt-v1",
            "max_tokens": 4096,
            "max_attempts": 3,
        },
        "provider": _provider(),
    }


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


def _evaluation(sample_id: str) -> dict[str, object]:
    return {
        "schema_version": "longcoqa-p0-nonfactoid-claim-v1",
        "sample_id": sample_id,
        "reference_claims": ["alpha", "beta"],
        "candidate_claims": [{"text": "alpha", "matched_reference": 0}],
        "claim_recall": 0.5,
    }


def _checkpoint(row: dict[str, object], *, cell_id: str, status: str) -> dict[str, object]:
    identity = _v1_identity()
    value: dict[str, object] = {
        "schema_version": "longcoqa-p0-rag-formal-checkpoint-v1",
        "status": status,
        "cell_id": cell_id,
        "task": "claim_recall",
        "sample_id": row["sample_id"],
        "question_type": "non_factoid",
        "answer_model": row["model"],
        "input_hash": stable_json_hash(_normalized_input(row)),
        "evaluator_hash": stable_json_hash(identity),
        "evaluator_identity": identity,
        "elapsed_seconds": 0.25,
    }
    if status == "ok":
        value["evaluation"] = _evaluation(str(row["sample_id"]))
    else:
        value.update({"error_type": "ValueError", "error": "strict parse failed"})
    return value


def _build_fixture(
    tmp_path: Path,
    *,
    second_cell_sample_ids: tuple[str, str] | None = None,
) -> tuple[Path, Path, dict[str, Path]]:
    output_root = tmp_path / "matrix"
    input_root = tmp_path / "inputs"
    sample_ids = ("non_factoid:1:1", "non_factoid:1:2")
    cell_specs = (
        ("model_a/z1", "Model A", "z1_teacher_forced", ("ok", "ok"), sample_ids),
        (
            "model_b/g3",
            "Model B",
            "g3_aligned_raw",
            ("ok", "error"),
            second_cell_sample_ids or sample_ids,
        ),
    )
    config_cells: list[dict[str, object]] = []
    error_paths: dict[str, Path] = {}
    claim_matrix_cells: list[dict[str, object]] = []
    for cell_id, model, protocol, statuses, cell_sample_ids in cell_specs:
        input_path = input_root / f"{cell_id.replace('/', '-')}.jsonl"
        input_rows = [
            _input_row(sample_id, model=model, protocol=protocol)
            for sample_id in cell_sample_ids
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
        cell_root = output_root / "claim_recall" / cell_id / "full"
        checkpoints = [
            _checkpoint(row, cell_id=cell_id, status=status)
            for row, status in zip(input_rows, statuses, strict=True)
        ]
        for checkpoint in checkpoints:
            checkpoint_path = cell_root / "checkpoints" / f"{stable_json_hash(checkpoint['sample_id'])}.json"
            _write_json(checkpoint_path, checkpoint)
            if checkpoint["status"] == "error":
                error_paths[cell_id] = checkpoint_path
        _write_jsonl(cell_root / "results.jsonl", checkpoints)
        successful = sum(status == "ok" for status in statuses)
        summary = {
            "schema_version": "longcoqa-p0-rag-formal-summary-v1",
            "cell_id": cell_id,
            "task": "claim_recall",
            "requested_samples": 2,
            "successful_samples": successful,
            "failed_samples": 2 - successful,
            "reused_samples": 0,
            "evaluator_hash": stable_json_hash(_v1_identity()),
            "evaluator_identity": _v1_identity(),
            "aggregate": {"samples": successful, "claim_recall": 0.5},
        }
        _write_json(cell_root / "summary.json", summary)
        if successful == 2:
            _write_json(cell_root / ".complete", summary)
        claim_matrix_cells.append(
            {
                "cell_id": cell_id,
                "task": "claim_recall",
                "status": "complete" if successful == 2 else "incomplete",
                "output_dir": str(cell_root),
                "evaluator_hash": (stable_json_hash(_v1_identity()) if successful == 2 else None),
                "successful_samples": successful,
                "failed_samples": 2 - successful,
            }
        )

    judge_root = output_root / "judge_100"
    for cell_id, *_rest in cell_specs:
        judge_cell = judge_root / cell_id / "full"
        (judge_cell / "results.jsonl").parent.mkdir(parents=True, exist_ok=True)
        (judge_cell / "results.jsonl").write_bytes(b'{"judge":"untouched"}\n')
        (judge_cell / "summary.json").write_bytes(b'{"judge":"summary"}\n')
        (judge_cell / ".complete").write_bytes(b'{"judge":"marker"}\n')

    matrix_summary = {
        "schema_version": "longcoqa-p0-nonfactoid-matrix-v1",
        "status": "incomplete",
        "expected_task_cells": 4,
        "completed_task_cells": 3,
        "expected_samples_per_cell": 2,
        "successful_samples": 7,
        "failed_samples": 1,
        "task_cells": claim_matrix_cells,
    }
    _write_json(output_root / "matrix_summary.json", matrix_summary)
    _write_json(output_root / ".complete", matrix_summary)

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
    return config_path, output_root, error_paths


def test_plan_is_provider_free_reports_exact_counts_and_preserves_bytes(
    tmp_path: Path,
) -> None:
    config, output_root, _errors = _build_fixture(tmp_path)
    before = _tree_hashes(tmp_path)

    plan = plan_claim_recall_v2_migration(config)
    report = plan.as_report()

    assert report["cell_count"] == 2
    assert report["ok_to_migrate"] == 3
    assert report["errors_left_pending"] == 1
    assert report["old_evaluator_hashes"] == [stable_json_hash(_v1_identity())]
    assert report["new_evaluator_hash"] == V2_HASH
    assert plan.new_evaluator_hash == V2_HASH
    assert plan.new_evaluator_identity == _v2_identity()
    assert report["intended_mutations"] == {
        "rewrite_ok_checkpoints": 3,
        "remove_claim_complete_markers": 1,
        "remove_matrix_root_complete_marker": True,
        "regenerate_summaries": False,
    }
    assert stable_json_hash(plan.new_evaluator_identity) == V2_HASH
    assert _tree_hashes(tmp_path) == before
    assert not (output_root / "migration_report.json").exists()


def test_execute_snapshots_selectively_rewrites_and_never_changes_judge(
    tmp_path: Path,
) -> None:
    config, output_root, errors = _build_fixture(tmp_path)
    plan = plan_claim_recall_v2_migration(config)
    error_bytes = {cell: path.read_bytes() for cell, path in errors.items()}
    judge_before = _tree_hashes(output_root / "judge_100")

    report = execute_claim_recall_v2_migration(plan, tmp_path / "backups")

    assert report["status"] == "executed"
    assert report["migrated_ok_checkpoints"] == 3
    assert report["errors_left_pending"] == 1
    assert report["intended_mutations"]["remove_matrix_root_complete_marker"] is True
    assert report["judge_hashes_before"] == judge_before
    assert report["judge_hashes_after"] == judge_before
    backup = Path(report["backup_path"])
    assert (backup / "claim_recall/model_a/z1/full/summary.json").is_file()
    assert (backup / "matrix/matrix_summary.json").is_file()
    assert (backup / "matrix/.complete").is_file()
    assert (backup / "snapshot_manifest.json").is_file()
    for path in output_root.glob("claim_recall/**/checkpoints/*.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value["status"] == "error":
            continue
        assert value["evaluator_hash"] == plan.new_evaluator_hash
        assert value["evaluator_identity"] == plan.new_evaluator_identity
        expected_evaluation = _evaluation(value["sample_id"])
        expected_evaluation["repair_audit"] = {
            "policy_version": V2_POLICY,
            "repair_applied": False,
            "successful_attempt": 1,
            "primary_validation_error": None,
        }
        assert value["evaluation"] == expected_evaluation
        assert "repair_audit" not in value
        assert value["migration_provenance"] == {
            "migration_version": "longcoqa-p0-claim-repair-migration-v2",
            "source_evaluator_hash": stable_json_hash(_v1_identity()),
            "source_evaluator_identity": _v1_identity(),
        }
    assert {cell: path.read_bytes() for cell, path in errors.items()} == error_bytes
    assert not any(output_root.glob("claim_recall/**/.complete"))
    assert not (output_root / ".complete").exists()
    assert _tree_hashes(output_root / "judge_100") == judge_before
    assert Path(report["report_path"]).is_file()


def test_execute_rejects_stale_plan_before_backup_or_output_mutation(
    tmp_path: Path,
) -> None:
    config, output_root, _errors = _build_fixture(tmp_path)
    plan = plan_claim_recall_v2_migration(config)
    summary_path = output_root / "claim_recall/model_b/g3/full/summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["aggregate"]["claim_recall"] = 0.75
    _write_json(summary_path, summary)
    before = _tree_hashes(output_root)

    with pytest.raises(ValueError, match="changed after planning"):
        execute_claim_recall_v2_migration(plan, tmp_path / "backups")

    assert _tree_hashes(output_root) == before
    assert not (tmp_path / "backups").exists()


def test_execute_rejects_input_changed_after_planning(tmp_path: Path) -> None:
    config, output_root, _errors = _build_fixture(tmp_path)
    plan = plan_claim_recall_v2_migration(config)
    input_path = tmp_path / "inputs/model_a-z1.jsonl"
    input_path.write_bytes(input_path.read_bytes() + b"\n")
    before = _tree_hashes(output_root)

    with pytest.raises(ValueError, match="changed after planning"):
        execute_claim_recall_v2_migration(plan, tmp_path / "backups")

    assert _tree_hashes(output_root) == before
    assert not (tmp_path / "backups").exists()


def test_plan_rejects_invalid_aggregate_without_writing(tmp_path: Path) -> None:
    config, output_root, _errors = _build_fixture(tmp_path)
    summary_path = output_root / "claim_recall/model_a/z1/full/summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["aggregate"]["claim_recall"] = 0.75
    _write_json(summary_path, summary)
    before = _tree_hashes(tmp_path)

    with pytest.raises(ValueError, match="aggregate"):
        plan_claim_recall_v2_migration(config)

    assert _tree_hashes(tmp_path) == before


def test_plan_rejects_cross_cell_sample_id_drift(tmp_path: Path) -> None:
    config, _output_root, _errors = _build_fixture(
        tmp_path,
        second_cell_sample_ids=("non_factoid:1:1", "non_factoid:1:3"),
    )

    with pytest.raises(ValueError, match="cross-cell sample-ID drift"):
        plan_claim_recall_v2_migration(config)


def test_repeated_migration_is_rejected_clearly(tmp_path: Path) -> None:
    config, _output_root, _errors = _build_fixture(tmp_path)
    plan = plan_claim_recall_v2_migration(config)
    execute_claim_recall_v2_migration(plan, tmp_path / "backups")

    with pytest.raises(ValueError, match="already uses the v2 evaluator identity"):
        plan_claim_recall_v2_migration(config)


def test_cli_defaults_to_dry_run_and_execute_requires_environment_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, output_root, _errors = _build_fixture(tmp_path)
    backup_root = tmp_path / "backups"
    before = _tree_hashes(output_root)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("provider construction is forbidden during migration")

    monkeypatch.setattr(cli_module, "load_provider_settings", forbidden)
    monkeypatch.setattr(cli_module, "ProviderClient", forbidden)
    monkeypatch.setattr(cli_module, "ResponseCache", forbidden)
    dry_run = CliRunner().invoke(
        cli_module.app,
        [
            "migrate-p0-nonfactoid-claim-v2",
            "--config",
            str(config),
            "--backup-root",
            str(backup_root),
        ],
    )
    assert dry_run.exit_code == 0, dry_run.output
    assert '"status": "dry_run"' in dry_run.output
    assert _tree_hashes(output_root) == before
    assert not backup_root.exists()

    monkeypatch.delenv(MIGRATE_ENV, raising=False)
    denied = CliRunner().invoke(
        cli_module.app,
        [
            "migrate-p0-nonfactoid-claim-v2",
            "--config",
            str(config),
            "--backup-root",
            str(backup_root),
            "--execute",
        ],
    )
    assert denied.exit_code != 0
    assert MIGRATE_ENV in denied.output
    assert _tree_hashes(output_root) == before
    assert not backup_root.exists()

    denied_before_validation = CliRunner().invoke(
        cli_module.app,
        [
            "migrate-p0-nonfactoid-claim-v2",
            "--config",
            str(tmp_path / "missing.yaml"),
            "--backup-root",
            str(backup_root),
            "--execute",
        ],
    )
    assert denied_before_validation.exit_code != 0
    assert MIGRATE_ENV in denied_before_validation.output
