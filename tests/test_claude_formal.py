from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from longcoqa.config import ProviderSettings
from longcoqa.inference.claude_formal import (
    FORMAL_PROTOCOLS,
    finalize_formal_suite,
    preflight_key_pool,
    run_formal_protocol,
    validate_formal_protocol,
)
from longcoqa.inference.free_running import (
    build_free_running_messages,
    history_hash,
)
from longcoqa.paths import PROJECT_ROOT
from longcoqa.providers.client import GenerationResult
from longcoqa.utils.io import read_yaml


@dataclass
class _Dataset:
    rows: list[dict[str, object]]

    def iter_samples(self):  # type: ignore[no-untyped-def]
        yield from self.rows


def _samples() -> list[dict[str, object]]:
    return [
        {
            "sample_id": "factoid:1:1",
            "question_type": "factoid",
            "dialogue_id": 1,
            "turn": 1,
            "question": "Q1",
            "answer": "GOLD1",
            "evidence_ids": [],
        },
        {
            "sample_id": "factoid:1:2",
            "question_type": "factoid",
            "dialogue_id": 1,
            "turn": 2,
            "question": "Q2",
            "answer": "GOLD2",
            "evidence_ids": [],
        },
    ]


def _rows(protocol: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    prior: list[tuple[str, str]] = []
    for sample in _samples():
        prediction = f"PRED:{sample['question']}"
        row: dict[str, object] = {
            "status": "ok",
            "sample_id": sample["sample_id"],
            "question_type": sample["question_type"],
            "dialogue_id": sample["dialogue_id"],
            "turn": sample["turn"],
            "protocol": protocol,
            "model": "claude-opus-4-6",
            "question": sample["question"],
            "reference": sample["answer"],
            "prediction": prediction,
        }
        if protocol == "z2_free_running":
            row.update(
                {
                    "model_requested": "claude-opus-4-6",
                    "history_hash": history_hash(prior),
                    "messages": build_free_running_messages(sample, prior),
                }
            )
        rows.append(row)
        prior.append((str(sample["question"]), prediction))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


@pytest.mark.parametrize("protocol", FORMAL_PROTOCOLS)
def test_validate_formal_protocol_accepts_exact_ordered_successes(protocol: str) -> None:
    report = validate_formal_protocol(
        _rows(protocol),
        expected_ids=[sample["sample_id"] for sample in _samples()],
        protocol=protocol,
        model="claude-opus-4-6",
    )

    assert report["status"] == "ok"
    assert report["successful_samples"] == 2


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda rows: list(reversed(rows)), "order or coverage"),
        (lambda rows: [rows[0], rows[0]], "order or coverage"),
        (
            lambda rows: [{**rows[0], "status": "error"}, rows[1]],
            "non-success",
        ),
        (
            lambda rows: [{**rows[0], "model": "wrong"}, rows[1]],
            "model mismatch",
        ),
    ],
)
def test_validate_formal_protocol_rejects_invalid_rows(mutation, message: str) -> None:  # noqa: ANN001
    with pytest.raises(ValueError, match=message):
        validate_formal_protocol(
            mutation(_rows("z1_teacher_forced")),
            expected_ids=[sample["sample_id"] for sample in _samples()],
            protocol="z1_teacher_forced",
            model="claude-opus-4-6",
        )


def test_validate_formal_protocol_rejects_broken_z2_chain() -> None:
    rows = _rows("z2_free_running")
    rows[1]["history_hash"] = "wrong"

    with pytest.raises(ValueError, match="history hash"):
        validate_formal_protocol(
            rows,
            expected_ids=[sample["sample_id"] for sample in _samples()],
            protocol="z2_free_running",
            model="claude-opus-4-6",
        )


def test_finalize_formal_suite_requires_all_protocol_markers(tmp_path: Path) -> None:
    output_root = tmp_path / "claude_opus_4_6"
    for protocol in FORMAL_PROTOCOLS:
        protocol_root = output_root / protocol / "full"
        _write_jsonl(protocol_root / "results.jsonl", _rows(protocol))
        (protocol_root / "summary.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="completion marker"):
        finalize_formal_suite(
            _Dataset(_samples()),
            output_root=output_root,
            model="claude-opus-4-6",
        )

    for protocol in FORMAL_PROTOCOLS:
        (output_root / protocol / "full" / ".complete").touch()
    summary = finalize_formal_suite(
        _Dataset(_samples()),
        output_root=output_root,
        model="claude-opus-4-6",
    )

    assert summary["status"] == "ok"
    assert (output_root / "suite_full.json").is_file()
    assert (output_root / ".complete").is_file()


def _settings() -> ProviderSettings:
    return ProviderSettings(
        name="anthropic_formal_claude_opus_4_6",
        protocol="openai",
        base_url="https://provider.invalid/v1",
        api_keys=("secret-1", "secret-2"),
        model="claude-opus-4-6",
        concurrency=2,
        requests_per_minute=1_000_000,
        timeout_seconds=5,
        max_retries=0,
    )


@pytest.mark.parametrize(
    ("protocol", "target"),
    [
        ("z1_teacher_forced", "longcoqa.inference.claude_formal.run_api_inference"),
        ("z2_free_running", "longcoqa.inference.claude_formal.run_free_running_inference"),
    ],
)
def test_run_formal_protocol_marks_only_strict_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    protocol: str,
    target: str,
) -> None:
    output_root = tmp_path / "formal"

    def fake_runner(*_args: object, **kwargs: object) -> dict[str, object]:
        output_dir = Path(kwargs["output_dir"])
        _write_jsonl(output_dir / "results.jsonl", _rows(protocol))
        summary = {"successful_samples": 2, "failed_samples": 0}
        (output_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return summary

    monkeypatch.setattr(target, fake_runner)
    summary = run_formal_protocol(
        _Dataset(_samples()),
        _settings(),
        protocol=protocol,
        alignment_path=tmp_path / "alignment.jsonl",
        cache_path=tmp_path / "cache.sqlite3",
        output_root=output_root,
    )

    assert summary["formal_validation"]["status"] == "ok"
    assert (output_root / protocol / "full" / ".complete").is_file()


def test_run_formal_protocol_does_not_mark_partial_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_runner(*_args: object, **kwargs: object) -> dict[str, object]:
        output_dir = Path(kwargs["output_dir"])
        rows = _rows("z1_teacher_forced")
        rows[1]["status"] = "error"
        _write_jsonl(output_dir / "results.jsonl", rows)
        summary = {"successful_samples": 1, "failed_samples": 1}
        (output_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return summary

    monkeypatch.setattr("longcoqa.inference.claude_formal.run_api_inference", fake_runner)
    output_root = tmp_path / "formal"
    with pytest.raises(ValueError, match="non-success"):
        run_formal_protocol(
            _Dataset(_samples()),
            _settings(),
            protocol="z1_teacher_forced",
            alignment_path=tmp_path / "alignment.jsonl",
            cache_path=tmp_path / "cache.sqlite3",
            output_root=output_root,
        )

    assert not (output_root / "z1_teacher_forced" / "full" / ".complete").exists()


def test_claude_formal_config_locks_scope_and_execution_tiers() -> None:
    config = read_yaml(PROJECT_ROOT / "configs/experiments/claude_opus_4_6_formal.yaml")

    assert config["schema_version"] == "longcoqa-claude-opus-46-formal-v1"
    assert config["provider"] == "anthropic"
    assert config["model"] == "claude-opus-4-6"
    assert config["expected_samples"] == 3019
    assert config["protocols"] == list(FORMAL_PROTOCOLS)
    assert config["output_root"] == "outputs/formal/models/claude_opus_4_6"
    assert config["runtime_root"] == "runtime/claude_opus_4_6_formal"
    assert config["generation"] == {
        "temperature": 0.0,
        "max_tokens_factoid": 256,
        "max_tokens_non_factoid": 768,
    }
    assert config["canary"]["dialogues_per_type"] == 2
    assert config["tiers"] == {
        "primary": {"concurrency": 8, "requests_per_minute": 60},
        "fallback": {"concurrency": 4, "requests_per_minute": 30},
    }


def test_claude_formal_cli_requires_explicit_paid_gate() -> None:
    script = PROJECT_ROOT / "scripts/run_claude_opus_46_formal.py"

    result = subprocess.run(
        [sys.executable, str(script), "--stage", "canary"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        timeout=10,
    )

    assert result.returncode != 0
    assert "--execute" in result.stderr


def test_claude_formal_cli_has_isolated_paths_and_no_secret_values() -> None:
    script = (PROJECT_ROOT / "scripts/run_claude_opus_46_formal.py").read_text(
        encoding="utf-8"
    )

    assert 'DEFAULT_ENV_FILE = ROOT / ".env"' in script
    assert "budget_pilot" not in script
    assert "read_env_file" in script
    assert "api_keys" in script
    assert "--execute" in script
    assert "finalize_formal_suite" in script
    assert "preflight_key_pool" in script


def test_claude_formal_supervisor_is_paid_gated_canary_first_and_resumable() -> None:
    script = (PROJECT_ROOT / "scripts/run_claude_opus_46_formal.sh").read_text(
        encoding="utf-8"
    )

    assert "set -Eeuo pipefail" in script
    assert "LONGCOQA_CLAUDE_FORMAL_EXECUTE" in script
    assert 'PYTHON="${LONGCOQA_PYTHON:-python}"' in script
    assert 'ENV_FILE="${LONGCOQA_ENV_FILE:-${PROJECT_ROOT}/.env}"' in script
    primary = "--concurrency 8 --requests-per-minute 60"
    fallback = "--concurrency 4 --requests-per-minute 30"
    assert primary in script
    assert fallback in script
    assert script.index("--stage canary") < script.index("--stage full")
    for protocol in FORMAL_PROTOCOLS:
        assert f'run_until_complete "{protocol}"' in script
    assert '[[ ! -f "${marker}" ]]' in script
    assert "--stage finalize" in script
    assert "outputs/formal/models/claude_opus_4_6/.complete" in script
    assert 'tee -a "${RUNTIME_ROOT}/supervisor.log"' in script
    assert '"${RUNTIME_ROOT}/supervisor.pid"' in script


def test_preflight_key_pool_keeps_only_keys_that_serve_claude() -> None:
    attempts: list[str] = []

    class Client:
        def __init__(self, settings: ProviderSettings) -> None:
            self.settings = settings

        def generate(self, *_args: object, **_kwargs: object) -> GenerationResult:
            key = self.settings.api_keys[0]
            attempts.append(key)
            if key == "bad-secret":
                raise RuntimeError("HTTP 400")
            return GenerationResult(
                provider=self.settings.name,
                model=self.settings.model,
                text="LONGCOQA_CLAUDE_FORMAL_OK",
                usage={},
                finish_reason="stop",
                request_hash="hash",
                cached=False,
                latency_seconds=0.1,
            )

        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    settings = _settings()
    settings = ProviderSettings(
        **{**settings.__dict__, "api_keys": ("bad-secret", "good-secret")}
    )

    verified, report = preflight_key_pool(settings, client_factory=Client)

    assert attempts == ["bad-secret", "good-secret"]
    assert verified.api_keys == ("good-secret",)
    assert report == {
        "configured_keys": 2,
        "working_key_indexes": [2],
        "working_keys": 1,
    }
    assert "secret" not in repr(report)
