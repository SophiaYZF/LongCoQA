from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import longcoqa.cli as cli_module
import longcoqa.evaluation.p0_nonfactoid_formal as formal_module
from longcoqa.config import ProviderSettings
from longcoqa.evaluation.p0_nonfactoid_formal import (
    CLAIM_RECALL_MAX_REPAIR_ATTEMPTS,
    CLAIM_RECALL_PARSER_VERSION,
    CLAIM_RECALL_PROMPT_VERSION,
    CLAIM_RECALL_REPAIR_MAX_TOKENS,
    JUDGE_100_PARSER_VERSION,
    JUDGE_100_PROMPT_VERSION,
    REQUIRED_CELL_CONTRACT,
    P0NonFactoidCell,
)
from longcoqa.utils.hashing import stable_json_hash

EXECUTE_ENV = "LONGCOQA_P0_NONFACTOID_EXECUTE"
WORKERS_ENV = "LONGCOQA_P0_NONFACTOID_MAX_WORKERS"
RPM_ENV = "LONGCOQA_P0_NONFACTOID_REQUESTS_PER_MINUTE"
CANARY_ENV = "LONGCOQA_P0_NONFACTOID_CLAIM_V3_CANARY"


def _cell(tmp_path: Path, cell_id: str = "model/z1") -> P0NonFactoidCell:
    input_path = tmp_path / f"{cell_id.replace('/', '-')}.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "sample_id": "non_factoid:7:2",
                "dialogue_id": 7,
                "turn": 2,
                "question_type": "non_factoid",
                "question": "Why?",
                "reference": "Because.",
                "prediction": "Therefore.",
                "model": "Model",
                "protocol": "z1_teacher_forced",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return P0NonFactoidCell(
        cell_id=cell_id,
        input_path=input_path,
        expected_model="Model",
        protocol="z1_teacher_forced",
    )


def _section(tmp_path: Path, *, provider: str = "qwen37") -> dict[str, object]:
    return {
        "provider": provider,
        "expected_source_samples": 1,
        "expected_nonfactoid_samples": 1,
        "cache_path": str(tmp_path / "cache.sqlite3"),
        "output_root": str(tmp_path / "outputs"),
    }


def _validation(total_calls: int = 2) -> dict[str, object]:
    return {
        "status": "valid",
        "cell_count": 1,
        "expected_task_cells": 2,
        "total_calls": total_calls,
    }


def _forbid_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("provider/cache constructed before safety gates")

    monkeypatch.setattr(cli_module, "load_provider_settings", forbidden)
    monkeypatch.setattr(cli_module, "ProviderClient", forbidden)
    monkeypatch.setattr(cli_module, "ResponseCache", forbidden)


def test_validate_command_is_provider_free_and_prints_planned_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cell = _cell(tmp_path)
    observed: list[tuple[int, int]] = []

    monkeypatch.setattr(
        cli_module,
        "load_p0_nonfactoid_config",
        lambda _path: (_section(tmp_path), (cell,)),
    )

    def validate(cells, expected_source_samples, expected_nonfactoid_samples):
        assert tuple(cells) == (cell,)
        observed.append((expected_source_samples, expected_nonfactoid_samples))
        return _validation(total_calls=2)

    monkeypatch.setattr(cli_module, "validate_p0_nonfactoid_inputs", validate)
    _forbid_provider(monkeypatch)

    result = CliRunner().invoke(
        cli_module.app,
        ["validate-p0-nonfactoid-main", "--config", str(tmp_path / "tiny.yaml")],
    )

    assert result.exit_code == 0, result.output
    assert observed == [(1, 1)]
    assert '"total_calls": 2' in result.output


@pytest.mark.parametrize(
    ("args", "environment"),
    [
        ([], {EXECUTE_ENV: "1"}),
        (["--execute"], {EXECUTE_ENV: "1"}),
        (
            ["--execute", "--cell", "model/z1", "--task", "claim_recall"],
            {EXECUTE_ENV: ""},
        ),
        (["--execute", "--cell", "model/z1"], {EXECUTE_ENV: "1"}),
        (["--execute", "--task", "claim_recall"], {EXECUTE_ENV: "1"}),
        (
            [
                "--execute",
                "--all-cells",
                "--cell",
                "model/z1",
                "--task",
                "claim_recall",
            ],
            {EXECUTE_ENV: "1"},
        ),
        (
            [
                "--execute",
                "--cell",
                "model/z1",
                "--task",
                "claim_recall",
                "--max-workers",
                "0",
            ],
            {EXECUTE_ENV: "1"},
        ),
        (
            ["--execute", "--cell", "model/z1", "--task", "claim_recall"],
            {EXECUTE_ENV: "1", WORKERS_ENV: "-2"},
        ),
        (
            ["--execute", "--cell", "model/z1", "--task", "claim_recall"],
            {EXECUTE_ENV: "1", RPM_ENV: "0"},
        ),
    ],
    ids=[
        "missing-cli-execute",
        "missing-selection",
        "missing-env-opt-in",
        "partial-cell-only",
        "partial-task-only",
        "mixed-selection",
        "nonpositive-cli-workers",
        "nonpositive-env-workers",
        "nonpositive-env-rpm",
    ],
)
def test_paid_execution_rejects_unsafe_invocations_before_provider(
    args: list[str], environment: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid_provider(monkeypatch)

    result = CliRunner().invoke(
        cli_module.app,
        ["evaluate-p0-nonfactoid-main", *args],
        env=environment,
    )

    assert result.exit_code != 0
    assert not isinstance(result.exception, AssertionError)
    assert "No such command" not in result.output


@pytest.mark.parametrize(
    ("selected_task", "prompt_version", "parser_version", "max_tokens"),
    [
        (
            "claim_recall",
            CLAIM_RECALL_PROMPT_VERSION,
            CLAIM_RECALL_PARSER_VERSION,
            1024,
        ),
        ("judge_100", JUDGE_100_PROMPT_VERSION, JUDGE_100_PARSER_VERSION, 512),
    ],
)
def test_valid_selected_execution_honors_worker_precedence_and_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selected_task: str,
    prompt_version: str,
    parser_version: str,
    max_tokens: int,
) -> None:
    cell = _cell(tmp_path)
    section = _section(tmp_path)
    events: list[str] = []
    runner_calls: list[dict[str, object]] = []
    matrix_calls: list[tuple[object, ...]] = []
    claim_evaluator_calls: list[tuple[object | None, object | None, object | None]] = []

    def load_config(_path: Path):
        events.append("config")
        return section, (cell,)

    def validate(*_args: object, **_kwargs: object):
        events.append("validation")
        return _validation()

    settings = ProviderSettings(
        name="qwen37",
        protocol="openai",
        model="qwen3.7-max",
        base_url="https://configured.invalid",
        api_keys=("secret",),
        concurrency=4,
        requests_per_minute=30,
        timeout_seconds=180.0,
        max_retries=8,
    )

    def load_settings(name: str):
        events.append("settings")
        assert name == "qwen37"
        return settings

    class FakeCache:
        def __init__(self, path: Path) -> None:
            events.append("cache")
            assert path == tmp_path / "cache.sqlite3"

        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    class FakeClient:
        def __init__(self, provider_settings, *, cache) -> None:
            events.append("client")
            assert provider_settings is not settings
            assert provider_settings.requests_per_minute == 60
            assert settings.requests_per_minute == 30
            assert isinstance(cache, FakeCache)

        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    evaluator = object()
    monkeypatch.setattr(cli_module, "load_p0_nonfactoid_config", load_config)
    monkeypatch.setattr(cli_module, "validate_p0_nonfactoid_inputs", validate)
    monkeypatch.setattr(cli_module, "load_provider_settings", load_settings)
    monkeypatch.setattr(cli_module, "ResponseCache", FakeCache)
    monkeypatch.setattr(cli_module, "ProviderClient", FakeClient)
    def make_claim_evaluator(
        _client: object,
        max_tokens: int,
        repair_max_tokens: int | None = None,
        max_repair_attempts: int | None = None,
    ) -> object:
        claim_evaluator_calls.append(
            (max_tokens, repair_max_tokens, max_repair_attempts)
        )
        return evaluator

    monkeypatch.setattr(cli_module, "make_claim_recall_evaluator", make_claim_evaluator)
    monkeypatch.setattr(
        cli_module, "make_judge_100_evaluator", lambda _client, max_tokens: evaluator
    )

    def run_task(*_args: object, **kwargs: object) -> dict[str, object]:
        runner_calls.append(kwargs)
        return {"cell_id": cell.cell_id, "task": selected_task, "failed_samples": 0}

    def write_matrix(cells, output_root, expected_samples):
        matrix_calls.append((tuple(cells), output_root, expected_samples))
        return {"status": "incomplete"}

    monkeypatch.setattr(cli_module, "run_p0_nonfactoid_task", run_task)
    monkeypatch.setattr(cli_module, "write_p0_nonfactoid_matrix_summary", write_matrix)

    result = CliRunner().invoke(
        cli_module.app,
        [
            "evaluate-p0-nonfactoid-main",
            "--config",
            str(tmp_path / "tiny.yaml"),
            "--execute",
            "--cell",
            cell.cell_id,
            "--task",
            selected_task,
            "--max-workers",
            "3",
        ],
        env={EXECUTE_ENV: "1", WORKERS_ENV: "2", RPM_ENV: "60"},
    )

    assert result.exit_code == 0, result.output
    assert events == ["config", "validation", "settings", "cache", "client"]
    assert len(runner_calls) == 1
    call = runner_calls[0]
    assert call["cell"] is cell
    assert call["task"] == selected_task
    assert call["evaluator"] is evaluator
    assert call["expected_samples"] == 1
    assert call["max_workers"] == 3
    assert len(call["rows"]) == 1
    identity = call["evaluator_identity"]
    if selected_task == "claim_recall":
        assert identity == {
            "identity_schema_version": "longcoqa-p0-nonfactoid-evaluator-identity-v3",
            "task": "claim_recall",
            "model": "qwen3.7-max",
            "primary": {
                "prompt_version": prompt_version,
                "parser_version": parser_version,
                "generation_config_version": formal_module.GENERATION_CONFIG_VERSION,
                "generation_options": formal_module.get_generation_options(),
                "max_tokens": max_tokens,
            },
            "repair": {
                "policy_version": "longcoqa-p0-nonfactoid-claim-repair-policy-v3",
                "prompt_version": "longcoqa-p0-nonfactoid-claim-repair-prompt-v2",
                "max_tokens": CLAIM_RECALL_REPAIR_MAX_TOKENS,
                "max_attempts": CLAIM_RECALL_MAX_REPAIR_ATTEMPTS,
            },
            "provider": settings.safe_summary(),
        }
        assert claim_evaluator_calls == [
            (1024, CLAIM_RECALL_REPAIR_MAX_TOKENS, CLAIM_RECALL_MAX_REPAIR_ATTEMPTS)
        ]
    else:
        assert identity == {
            "identity_schema_version": "longcoqa-p0-nonfactoid-evaluator-identity-v1",
            "task": selected_task,
            "model": "qwen3.7-max",
            "prompt_version": prompt_version,
            "parser_version": parser_version,
            "generation_config_version": formal_module.GENERATION_CONFIG_VERSION,
            "generation_options": dict(formal_module.GENERATION_OPTIONS),
            "max_tokens": max_tokens,
            "provider": settings.safe_summary(),
        }
        assert claim_evaluator_calls == []
    assert matrix_calls == [((cell,), tmp_path / "outputs", 1)]
    response = json.loads(result.output)
    assert response["runs"][0]["task"] == selected_task
    assert "secret" not in result.output


def test_claim_recall_evaluator_identity_is_stable_and_defensively_copied() -> None:
    provider = {
        "name": "qwen37",
        "nested": {"endpoint": "configured"},
        "api_key_count": 2,
    }

    first = formal_module.build_claim_recall_evaluator_identity(provider)
    second = formal_module.build_claim_recall_evaluator_identity(
        {"name": "qwen37", "nested": {"endpoint": "configured"}, "api_key_count": 2}
    )

    assert stable_json_hash(first) == stable_json_hash(second)
    assert first["provider"] == provider
    assert first["primary"]["generation_options"] == formal_module.get_generation_options()
    first["provider"]["nested"]["endpoint"] = "mutated"
    first["primary"]["generation_options"]["response_format"]["type"] = "mutated"

    later = formal_module.build_claim_recall_evaluator_identity(provider)
    assert provider["nested"]["endpoint"] == "configured"
    assert later["provider"]["nested"]["endpoint"] == "configured"
    assert later["primary"]["generation_options"] == formal_module.get_generation_options()


def test_claim_v3_canary_requires_gate_before_provider_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_provider(monkeypatch)

    result = CliRunner().invoke(
        cli_module.app,
        [
            "canary-p0-nonfactoid-claim-v3",
            "--config",
            "ignored.yaml",
            "--cell",
            "model/z1",
            "--sample-id",
            "non_factoid:7:2",
        ],
        env={CANARY_ENV: ""},
    )

    assert result.exit_code != 0
    assert not isinstance(result.exception, AssertionError)


def test_claim_v3_canary_rejects_unknown_cell_before_provider_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        cli_module,
        "load_p0_nonfactoid_config",
        lambda _path: (_section(tmp_path), (_cell(tmp_path),)),
    )
    _forbid_provider(monkeypatch)

    result = CliRunner().invoke(
        cli_module.app,
        [
            "canary-p0-nonfactoid-claim-v3",
            "--config",
            str(tmp_path / "tiny.yaml"),
            "--cell",
            "unknown",
            "--sample-id",
            "non_factoid:7:2",
        ],
        env={CANARY_ENV: "1"},
    )

    assert result.exit_code != 0
    assert not isinstance(result.exception, AssertionError)


def test_claim_v3_canary_normalizes_one_row_and_does_not_write_formal_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cell = _cell(tmp_path)
    section = _section(tmp_path)
    settings = ProviderSettings(
        name="qwen37",
        protocol="openai",
        model="qwen3.7-max",
        base_url="https://configured.invalid",
        api_keys=("secret-api-key",),
        concurrency=4,
        requests_per_minute=30,
        timeout_seconds=180.0,
        max_retries=8,
    )
    events: list[str] = []
    normalized_rows: list[dict[str, object]] = []
    evaluated_rows: list[dict[str, object]] = []

    class FakeCache:
        def __init__(self, path: Path) -> None:
            events.append("cache")
            assert path == tmp_path / "cache.sqlite3"

        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    class FakeClient:
        def __init__(self, configured_settings: ProviderSettings, *, cache: FakeCache) -> None:
            events.append("client")
            assert configured_settings is settings
            assert isinstance(cache, FakeCache)

        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    def normalize(row: dict[str, object], selected_cell: P0NonFactoidCell) -> dict[str, object]:
        assert selected_cell is cell
        normalized = formal_module.normalize_nonfactoid_row(row, selected_cell)
        normalized_rows.append(normalized)
        return normalized

    def make_evaluator(client: FakeClient, **_kwargs: object):
        assert isinstance(client, FakeClient)

        def evaluate(row: dict[str, object]) -> dict[str, object]:
            evaluated_rows.append(row)
            return {"claim_recall": 1.0}

        return evaluate

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("canary attempted a formal write path")

    monkeypatch.setattr(cli_module, "load_p0_nonfactoid_config", lambda _path: (section, (cell,)))
    monkeypatch.setattr(cli_module, "load_provider_settings", lambda _name: settings)
    monkeypatch.setattr(cli_module, "ResponseCache", FakeCache)
    monkeypatch.setattr(cli_module, "ProviderClient", FakeClient)
    monkeypatch.setattr(cli_module, "normalize_nonfactoid_row", normalize)
    monkeypatch.setattr(cli_module, "make_claim_recall_evaluator", make_evaluator)
    monkeypatch.setattr(cli_module, "run_p0_nonfactoid_task", forbidden)
    monkeypatch.setattr(cli_module, "write_p0_nonfactoid_matrix_summary", forbidden)
    monkeypatch.setattr(cli_module, "run_formal_task", forbidden)

    result = CliRunner().invoke(
        cli_module.app,
        [
            "canary-p0-nonfactoid-claim-v3",
            "--config",
            str(tmp_path / "tiny.yaml"),
            "--cell",
            cell.cell_id,
            "--sample-id",
            "non_factoid:7:2",
        ],
        env={CANARY_ENV: "1"},
    )

    assert result.exit_code == 0, result.output
    assert events == ["cache", "client"]
    assert len(normalized_rows) == 1
    assert evaluated_rows == normalized_rows
    response = json.loads(result.output)
    assert response == {
        "status": "ok",
        "cell": cell.cell_id,
        "sample_id": "non_factoid:7:2",
        "evaluator_hash": stable_json_hash(
            formal_module.build_claim_recall_evaluator_identity(settings.safe_summary())
        ),
        "evaluation": {"claim_recall": 1.0},
        "cache_path": str(tmp_path / "cache.sqlite3"),
    }
    assert "secret-api-key" not in result.output


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [
            {
                "status": "ok",
                "sample_id": "non_factoid:7:2",
                "dialogue_id": 7,
                "turn": 2,
                "question_type": "non_factoid",
                "question": "Why?",
                "reference": "Because.",
                "prediction": "Therefore.",
                "model": "Model",
                "protocol": "z1_teacher_forced",
            },
            {
                "status": "ok",
                "sample_id": "non_factoid:7:2",
                "dialogue_id": 7,
                "turn": 2,
                "question_type": "non_factoid",
                "question": "Why again?",
                "reference": "Because again.",
                "prediction": "Therefore again.",
                "model": "Model",
                "protocol": "z1_teacher_forced",
            },
        ],
    ],
    ids=["unknown-sample", "duplicate-sample"],
)
def test_claim_v3_canary_rejects_unknown_or_ambiguous_sample_before_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, object]]
) -> None:
    cell = _cell(tmp_path)
    cell.input_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    monkeypatch.setattr(
        cli_module, "load_p0_nonfactoid_config", lambda _path: (_section(tmp_path), (cell,))
    )
    _forbid_provider(monkeypatch)

    result = CliRunner().invoke(
        cli_module.app,
        [
            "canary-p0-nonfactoid-claim-v3",
            "--config",
            str(tmp_path / "tiny.yaml"),
            "--cell",
            cell.cell_id,
            "--sample-id",
            "non_factoid:7:2",
        ],
        env={CANARY_ENV: "1"},
    )

    assert result.exit_code != 0
    assert not isinstance(result.exception, AssertionError)


@pytest.mark.parametrize(
    ("worker_env", "expected_workers"),
    [("2", 2), (None, 4)],
    ids=["environment-override", "provider-default"],
)
def test_worker_precedence_falls_back_from_env_to_provider_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    worker_env: str | None,
    expected_workers: int,
) -> None:
    cell = _cell(tmp_path)
    section = _section(tmp_path)
    captured: list[int] = []
    settings = SimpleNamespace(
        model="qwen3.7-max",
        base_url="https://configured.invalid",
        api_keys=("secret",),
        concurrency=4,
        safe_summary=lambda: {"name": "qwen37", "model": "qwen3.7-max"},
    )
    monkeypatch.setattr(
        cli_module, "load_p0_nonfactoid_config", lambda _path: (section, (cell,))
    )
    monkeypatch.setattr(
        cli_module, "validate_p0_nonfactoid_inputs", lambda *_args, **_kwargs: _validation()
    )
    monkeypatch.setattr(cli_module, "load_provider_settings", lambda _name: settings)

    class Context:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(cli_module, "ResponseCache", Context)
    monkeypatch.setattr(cli_module, "ProviderClient", Context)
    monkeypatch.setattr(cli_module, "make_claim_recall_evaluator", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        cli_module,
        "run_p0_nonfactoid_task",
        lambda *_args, **kwargs: captured.append(kwargs["max_workers"])
        or {"failed_samples": 0},
    )
    monkeypatch.setattr(
        cli_module,
        "write_p0_nonfactoid_matrix_summary",
        lambda *_args, **_kwargs: {"status": "incomplete"},
    )

    result = CliRunner().invoke(
        cli_module.app,
        [
            "evaluate-p0-nonfactoid-main",
            "--execute",
            "--cell",
            cell.cell_id,
            "--task",
            "claim_recall",
        ],
        env={EXECUTE_ENV: "1", WORKERS_ENV: worker_env},
    )

    assert result.exit_code == 0, result.output
    assert captured == [expected_workers]


@pytest.mark.parametrize(
    ("cell_id", "task"),
    [("unknown", "claim_recall"), ("model/z1", "unknown")],
)
def test_unknown_cell_or_task_fails_without_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cell_id: str,
    task: str,
) -> None:
    cell = _cell(tmp_path)
    monkeypatch.setattr(
        cli_module,
        "load_p0_nonfactoid_config",
        lambda _path: (_section(tmp_path), (cell,)),
    )
    monkeypatch.setattr(
        cli_module, "validate_p0_nonfactoid_inputs", lambda *_args, **_kwargs: _validation()
    )
    _forbid_provider(monkeypatch)

    result = CliRunner().invoke(
        cli_module.app,
        [
            "evaluate-p0-nonfactoid-main",
            "--execute",
            "--cell",
            cell_id,
            "--task",
            task,
        ],
        env={EXECUTE_ENV: "1"},
    )

    assert result.exit_code != 0
    assert not isinstance(result.exception, AssertionError)


@pytest.mark.parametrize(
    ("provider", "model", "base_url", "api_keys"),
    [
        ("other", "qwen3.7-max", "https://configured.invalid", ("secret",)),
        ("qwen37", "not-qwen3.7-max", "https://configured.invalid", ("secret",)),
        ("qwen37", "qwen3.7-max", "", ("secret",)),
        ("qwen37", "qwen3.7-max", "https://configured.invalid", ()),
    ],
)
def test_provider_contract_fails_before_cache_or_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    model: str,
    base_url: str,
    api_keys: tuple[str, ...],
) -> None:
    cell = _cell(tmp_path)
    monkeypatch.setattr(
        cli_module,
        "load_p0_nonfactoid_config",
        lambda _path: (_section(tmp_path, provider=provider), (cell,)),
    )
    monkeypatch.setattr(
        cli_module, "validate_p0_nonfactoid_inputs", lambda *_args, **_kwargs: _validation()
    )
    settings = SimpleNamespace(model=model, base_url=base_url, api_keys=api_keys, concurrency=4)
    monkeypatch.setattr(cli_module, "load_provider_settings", lambda _name: settings)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("cache/client constructed for invalid provider contract")

    monkeypatch.setattr(cli_module, "ProviderClient", forbidden)
    monkeypatch.setattr(cli_module, "ResponseCache", forbidden)

    result = CliRunner().invoke(
        cli_module.app,
        [
            "evaluate-p0-nonfactoid-main",
            "--execute",
            "--cell",
            cell.cell_id,
            "--task",
            "claim_recall",
        ],
        env={EXECUTE_ENV: "1"},
    )

    assert result.exit_code != 0
    assert not isinstance(result.exception, AssertionError)


def test_shell_entrypoint_encodes_guarded_execution_contract() -> None:
    wrapper = Path("scripts/run_p0_nonfactoid_main.sh").read_text(encoding="utf-8")

    assert "set -euo pipefail" in wrapper
    assert 'PYTHON="${LONGCOQA_PYTHON:-python}"' in wrapper
    assert "validate-p0-nonfactoid-main" in wrapper
    assert "evaluate-p0-nonfactoid-main" in wrapper
    assert "LONGCOQA_P0_NONFACTOID_EXECUTE:-0" in wrapper
    assert 'ENV_FILE="${LONGCOQA_ENV_FILE:-${PROJECT_ROOT}/.env}"' in wrapper
    assert "source \"${ENV_FILE}\"" in wrapper
    assert "--execute" in wrapper

def test_default_wrapper_validation_does_not_source_environment_file(tmp_path: Path) -> None:
    wrapper = Path("scripts/run_p0_nonfactoid_main.sh")
    fake_python = tmp_path / "python"
    fake_python.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n", encoding="utf-8")
    fake_python.chmod(0o755)
    env_file = tmp_path / "must-not-source.env"
    env_file.write_text("exit 93\n", encoding="utf-8")

    result = subprocess.run(
        ["bash", str(wrapper)],
        cwd=Path.cwd(),
        env={
            "PATH": "/usr/bin:/bin",
            "LONGCOQA_PYTHON": str(fake_python),
            "LONGCOQA_ENV_FILE": str(env_file),
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "validate-p0-nonfactoid-main" in result.stdout
    assert "evaluate-p0-nonfactoid-main" not in result.stdout


def test_main_table_update_command_is_provider_free_and_uses_atomic_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = tmp_path / "paper.md"
    document.write_text("original\n", encoding="utf-8")
    cell = _cell(tmp_path)
    observed: list[tuple[str, Path, tuple[P0NonFactoidCell, ...]]] = []

    monkeypatch.setattr(
        cli_module,
        "load_p0_nonfactoid_config",
        lambda _path: (_section(tmp_path), (cell,)),
    )

    def update(text, matrix_root, cells):
        observed.append((text, matrix_root, tuple(cells)))
        return "updated\n"

    writes: list[tuple[Path, str]] = []

    def atomic_write(path: Path, text: str) -> None:
        writes.append((path, text))
        path.write_text(text, encoding="utf-8")

    monkeypatch.setattr(cli_module, "update_main_table_2", update)
    monkeypatch.setattr(cli_module, "_atomic_write_text", atomic_write)
    _forbid_provider(monkeypatch)

    result = CliRunner().invoke(
        cli_module.app,
        [
            "update-p0-nonfactoid-main-table",
            "--document",
            str(document),
            "--matrix-root",
            str(tmp_path / "matrix"),
            "--config",
            str(tmp_path / "config.yaml"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert observed == [("original\n", tmp_path / "matrix", (cell,))]
    assert writes == [(document, "updated\n")]
    assert json.loads(result.output) == {
        "document": str(document),
        "updated_metric_cells": 28,
        "updated_rows": 14,
    }


def test_main_table_update_command_failure_leaves_document_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = tmp_path / "paper.md"
    original = b"original\r\nbytes\n"
    document.write_bytes(original)
    cells = tuple(
        P0NonFactoidCell(
            cell_id=cell_id,
            input_path=tmp_path / "unused.jsonl",
            expected_model=model,
            protocol=protocol,
        )
        for cell_id, model, protocol in sorted(REQUIRED_CELL_CONTRACT)
    )
    monkeypatch.setattr(
        cli_module,
        "load_p0_nonfactoid_config",
        lambda _path: (_section(tmp_path), cells),
    )
    monkeypatch.setattr(
        cli_module,
        "_atomic_write_text",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("atomic write called after updater failure")
        ),
    )
    _forbid_provider(monkeypatch)

    result = CliRunner().invoke(
        cli_module.app,
        [
            "update-p0-nonfactoid-main-table",
            "--document",
            str(document),
            "--matrix-root",
            str(tmp_path / "matrix"),
            "--config",
            str(tmp_path / "config.yaml"),
        ],
    )

    assert result.exit_code != 0
    assert not isinstance(result.exception, AssertionError)
    assert document.read_bytes() == original
