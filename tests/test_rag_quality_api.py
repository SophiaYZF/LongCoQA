from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import longcoqa.cli as cli_module
from longcoqa.cli import app
from longcoqa.evaluation.rag_quality import (
    build_rag_quality_messages,
    make_rag_quality_evaluator,
    parse_rag_quality_response,
)


def sample_row() -> dict[str, object]:
    return {
        "sample_id": "factoid:1:1",
        "question_type": "factoid",
        "question": "谁发现了答案？",
        "prediction": "小天狼星。[RETRIEVED 1]",
        "reference": "小天狼星",
        "evidence_text": "[RETRIEVED 1] 小天狼星发现了答案。",
        "answerable": True,
        "cited_evidence_ids": ["1"],
        "gold_evidence_ids": ["1"],
    }


def valid_response() -> str:
    return json.dumps(
        {
            "reference_claims": ["小天狼星发现了答案"],
            "candidate_claims": [
                {
                    "text": "小天狼星发现了答案",
                    "material": True,
                    "supported": True,
                    "matched_reference": 0,
                }
            ],
            "citation_assessments": [{"citation_id": "1", "supported": True}],
        },
        ensure_ascii=False,
    )


def test_build_rag_quality_messages_contains_visible_inputs_only() -> None:
    messages = build_rag_quality_messages(sample_row())
    joined = "\n".join(message["content"] for message in messages)

    assert "谁发现了答案" in joined
    assert "小天狼星" in joined
    assert "RETRIEVED 1" in joined
    assert "api_key" not in joined


def test_parse_rag_quality_response_builds_claim_and_citation_metrics() -> None:
    result = parse_rag_quality_response(sample_row(), valid_response())

    assert result["status"] == "ok"
    assert result["claim_recall"] == 1.0
    assert result["claim_support_rate"] == 1.0
    assert result["answer_supported"] is True
    assert result["citation_semantic_precision"] == 1.0


@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        json.dumps({"reference_claims": [], "candidate_claims": []}),
        json.dumps(
            {
                "reference_claims": ["a"],
                "candidate_claims": [
                    {"text": "a", "material": "yes", "supported": True}
                ],
                "citation_assessments": [],
            }
        ),
    ],
)
def test_parse_rag_quality_response_rejects_invalid_schema(payload: str) -> None:
    with pytest.raises(ValueError):
        parse_rag_quality_response(sample_row(), payload)


def test_make_rag_quality_evaluator_uses_provider_and_parser() -> None:
    calls: list[dict[str, object]] = []

    class FakeClient:
        def generate(self, messages, **kwargs):
            calls.append({"messages": messages, **kwargs})
            return SimpleNamespace(text=valid_response())

    evaluator = make_rag_quality_evaluator(FakeClient(), max_tokens=600)
    result = evaluator(sample_row())

    assert result["answer_supported"] is True
    assert calls[0]["temperature"] == 0.0
    assert calls[0]["max_tokens"] == 600


def write_smoke_input(path: Path) -> None:
    rows = []
    dialogue = 1
    for question_type in ("factoid", "non_factoid"):
        for turn in (1, 3, 5):
            for index in range(2):
                item = sample_row()
                item.update(
                    {
                        "sample_id": f"{question_type}:{dialogue}:{index + 1}",
                        "question_type": question_type,
                        "dialogue_id": dialogue,
                        "turn": turn,
                        "status": "ok",
                    }
                )
                rows.append(item)
            dialogue += 1
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_rag_quality_cli_refuses_before_provider_construction(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "input.jsonl"
    write_smoke_input(input_path)
    monkeypatch.setattr(
        cli_module,
        "load_provider_settings",
        lambda *_: pytest.fail("provider must not be constructed before smoke validation"),
    )

    result = CliRunner().invoke(
        app,
        [
            "evaluate-rag-quality-smoke",
            "--input",
            str(input_path),
            "--provider",
            "fake",
            "--cell",
            "model/protocol",
            "--output-root",
            str(tmp_path / "runtime" / "paper_metrics_api_smoke"),
        ],
    )

    assert result.exit_code != 0
    assert "--smoke" in str(result.exception)


def test_rag_quality_cli_runs_exactly_twelve_rows(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "input.jsonl"
    write_smoke_input(input_path)
    calls = 0

    class FakeSettings:
        model = "judge-v1"

        def safe_summary(self):
            return {"name": "fake", "model": self.model}

    class FakeCache:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class FakeClient:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def generate(self, *_args, **_kwargs):
            nonlocal calls
            calls += 1
            return SimpleNamespace(text=valid_response())

    monkeypatch.setattr(cli_module, "load_provider_settings", lambda *_: FakeSettings())
    monkeypatch.setattr(cli_module, "ResponseCache", FakeCache)
    monkeypatch.setattr(cli_module, "ProviderClient", FakeClient)
    output_root = tmp_path / "runtime" / "paper_metrics_api_smoke"

    result = CliRunner().invoke(
        app,
        [
            "evaluate-rag-quality-smoke",
            "--input",
            str(input_path),
            "--provider",
            "fake",
            "--cell",
            "model/protocol",
            "--output-root",
            str(output_root),
            "--smoke",
            "--limit",
            "12",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == 12
    assert not any(path.name == ".complete" for path in output_root.rglob("*"))
