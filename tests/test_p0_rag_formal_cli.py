from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

import longcoqa.cli as cli_module


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_tiny_config(tmp_path: Path) -> Path:
    input_path = tmp_path / "input.jsonl"
    alignments = tmp_path / "alignment.jsonl"
    parents = tmp_path / "parents.jsonl"
    write_jsonl(
        input_path,
        [
            {
                "status": "ok",
                "sample_id": "s1",
                "question_type": "factoid",
                "question": "q",
                "reference": "r",
                "prediction": "a",
                "model": "m",
            }
        ],
    )
    write_jsonl(alignments, [])
    write_jsonl(parents, [])
    config = tmp_path / "formal.yaml"
    config.write_text(
        f"""p0_rag_formal:
  provider: qwen37
  expected_samples: 1
  cache_path: {tmp_path / 'cache.sqlite3'}
  output_root: {tmp_path / 'outputs'}
  evidence_alignment_path: {alignments}
  parents_path: {parents}
  cells:
    - cell_id: m/z1
      input_path: {input_path}
      tasks: [answer_quality]
      evidence_mode: none
      citation_mode: none
""",
        encoding="utf-8",
    )
    return config


def test_validate_command_never_constructs_provider(tmp_path: Path, monkeypatch) -> None:
    config = write_tiny_config(tmp_path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("provider must not be constructed during validation")

    monkeypatch.setattr(cli_module, "load_provider_settings", forbidden)
    monkeypatch.setattr(cli_module, "ProviderClient", forbidden)
    result = CliRunner().invoke(
        cli_module.app,
        ["validate-p0-rag-formal", "--config", str(config)],
    )
    assert result.exit_code == 0, result.output
    assert '"status": "valid"' in result.output
    assert '"total_calls": 1' in result.output
    assert '"answer_rubric_calls": 1' in result.output
    assert '"refusal_rubric_calls": 0' in result.output


def test_evaluate_requires_execute_and_explicit_scope_before_provider(monkeypatch) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("provider constructed before safety gates")

    monkeypatch.setattr(cli_module, "load_provider_settings", forbidden)
    runner = CliRunner()
    result = runner.invoke(cli_module.app, ["evaluate-p0-rag-formal"])
    assert result.exit_code != 0
    assert "--execute" in result.output

    result = runner.invoke(cli_module.app, ["evaluate-p0-rag-formal", "--execute"])
    assert result.exit_code != 0
    assert "--cell" in result.output and "--task" in result.output


def test_formal_shell_wrapper_is_validate_by_default_and_execute_is_opt_in() -> None:
    text = Path("scripts/run_p0_rag_formal.sh").read_text(encoding="utf-8")
    assert "validate-p0-rag-formal" in text
    assert 'LONGCOQA_P0_RAG_EXECUTE:-0' in text
    assert "evaluate-p0-rag-formal" in text
    assert "--execute" in text
