from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from longcoqa.evaluation.p0_nonfactoid_formal import aggregate_claim_recall
from longcoqa.evaluation.p0_rag_formal import (
    P0RagCell,
    citation_position_scores,
    load_evidence_parent_map,
    load_p0_rag_manifest,
    make_answer_quality_evaluator,
    make_formal_rag_quality_evaluator,
    normalize_formal_row,
    run_formal_task,
    validate_formal_task_output,
    write_matrix_summary,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_real_manifest_declares_expected_matrix() -> None:
    cells = load_p0_rag_manifest(Path("configs/experiments/p0_rag_formal.yaml"))
    assert len(cells) == 7
    assert sum("answer_quality" in cell.tasks for cell in cells) == 7
    assert sum("rag_quality" in cell.tasks for cell in cells) == 6
    assert sum(cell.citation_mode == "retrieved_positions" for cell in cells) == 5
    assert sum(cell.citation_mode == "not_requested" for cell in cells) == 1


def test_manifest_rejects_duplicate_cells_and_invalid_modes(tmp_path: Path) -> None:
    config = tmp_path / "manifest.yaml"
    config.write_text(
        """p0_rag_formal:
  expected_samples: 2
  cells:
    - cell_id: duplicate
      input_path: a.jsonl
      tasks: [answer_quality]
      evidence_mode: none
      citation_mode: none
    - cell_id: duplicate
      input_path: b.jsonl
      tasks: [rag_quality]
      evidence_mode: none
      citation_mode: retrieved_positions
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate cell_id|evidence_mode"):
        load_p0_rag_manifest(config)


def test_evidence_parent_mapping_and_citation_positions(tmp_path: Path) -> None:
    alignments = tmp_path / "alignments.jsonl"
    parents = tmp_path / "parents.jsonl"
    write_jsonl(
        alignments,
        [
            {"evidence_id": 7, "paragraph_ids": ["p1", "p2"]},
            {"evidence_id": 8, "paragraph_ids": ["p4"]},
        ],
    )
    write_jsonl(
        parents,
        [
            {"parent_id": "parent-a", "paragraph_ids": ["p1", "p2"]},
            {"parent_id": "parent-b", "paragraph_ids": ["p3", "p4"]},
        ],
    )
    mapping = load_evidence_parent_map(alignments, parents)
    assert mapping == {"7": ("parent-a",), "8": ("parent-b",)}

    score = citation_position_scores(
        prediction="答案。\n引用：[RETRIEVED 1], [RETRIEVED 3], [RETRIEVED 9]",
        evidence_ids=[7],
        retrieved_parent_ids=["parent-a", "noise", "parent-a"],
        evidence_parent_map=mapping,
    )
    assert score["gold_positions"] == ["1", "3"]
    assert score["cited_positions"] == ["1", "3", "9"]
    assert score["correct_positions"] == ["1", "3"]
    assert score["out_of_range_positions"] == ["9"]
    assert score["precision"] == pytest.approx(2 / 3)
    assert score["recall"] == 1.0
    assert score["f1"] == pytest.approx(0.8)


def test_normalize_prefers_answer_text_and_marks_g0_citation_inapplicable() -> None:
    cell = P0RagCell(
        cell_id="model/g0",
        input_path=Path("unused.jsonl"),
        tasks=("answer_quality", "rag_quality"),
        evidence_mode="released_gold",
        citation_mode="not_requested",
    )
    row = {
        "sample_id": "s1",
        "dialogue_id": "d1",
        "turn": 2,
        "question_type": "factoid",
        "question": "q",
        "reference": "r",
        "prediction": "fallback",
        "answer_text": "preferred",
        "model": "answer-model",
        "evidence_ids": [1],
        "messages": [{"role": "user", "content": "[EVIDENCE 1]\ne\n[QUESTION]\nq"}],
    }
    normalized = normalize_formal_row(row, cell, evidence_parent_map={})
    assert normalized["prediction"] == "preferred"
    assert normalized["evidence_text"].startswith("[EVIDENCE 1]")
    assert normalized["citation_status"] == "inapplicable"
    assert normalized["citation_reason"] == "not_requested"

    missing_gold = {**row, "sample_id": "s2", "evidence_ids": [], "messages": []}
    normalized_missing = normalize_formal_row(
        missing_gold, cell, evidence_parent_map={}
    )
    assert normalized_missing["answerable"] is False
    assert normalized_missing["evidence_text"] == ""


def _row(sample_id: str, prediction: str = "a") -> dict:
    return {
        "sample_id": sample_id,
        "question_type": "factoid",
        "question": "q",
        "reference": "r",
        "prediction": prediction,
        "model": "answer-model",
    }


def test_formal_runner_reuses_only_hash_compatible_success(tmp_path: Path) -> None:
    calls: list[str] = []

    def evaluator(row: dict) -> dict:
        calls.append(row["sample_id"])
        return {"verdict": "correct"}

    output = tmp_path / "answer_quality" / "cell" / "full"
    rows = [_row("s1"), _row("s2")]
    first = run_formal_task(
        rows,
        cell_id="cell",
        task="answer_quality",
        output_dir=output,
        evaluator=evaluator,
        evaluator_identity={"model": "judge-a"},
        expected_samples=2,
    )
    assert first["successful_samples"] == 2
    assert first["reused_samples"] == 0
    assert (output / ".complete").is_file()
    assert calls == ["s1", "s2"]

    calls.clear()
    second = run_formal_task(
        rows,
        cell_id="cell",
        task="answer_quality",
        output_dir=output,
        evaluator=evaluator,
        evaluator_identity={"model": "judge-a"},
        expected_samples=2,
    )
    assert second["reused_samples"] == 2
    assert calls == []

    changed = [_row("s1", "changed"), _row("s2")]
    run_formal_task(
        changed,
        cell_id="cell",
        task="answer_quality",
        output_dir=output,
        evaluator=evaluator,
        evaluator_identity={"model": "judge-a"},
        expected_samples=2,
    )
    assert calls == ["s1"]

    calls.clear()
    run_formal_task(
        changed,
        cell_id="cell",
        task="answer_quality",
        output_dir=output,
        evaluator=evaluator,
        evaluator_identity={"model": "judge-b"},
        expected_samples=2,
    )
    assert calls == ["s1", "s2"]


def test_custom_formal_aggregator_supports_claim_task(tmp_path: Path) -> None:
    output = tmp_path / "claim_recall" / "cell" / "full"

    summary = run_formal_task(
        [_row("s1")],
        cell_id="cell",
        task="claim_recall",
        output_dir=output,
        evaluator=lambda _row: {"claim_recall": 0.75},
        evaluator_identity={"task": "claim_recall", "version": 1},
        expected_samples=1,
        aggregate_evaluations=aggregate_claim_recall,
    )

    assert summary["aggregate"]["claim_recall"] == 0.75
    assert summary["successful_samples"] == 1
    assert (output / ".complete").is_file()


def test_custom_task_still_requires_an_aggregator(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="without aggregator"):
        run_formal_task(
            [_row("s1")],
            cell_id="cell",
            task="claim_recall",
            output_dir=tmp_path / "claim_recall" / "cell" / "full",
            evaluator=lambda _row: {"claim_recall": 0.75},
            evaluator_identity={"task": "claim_recall", "version": 1},
            expected_samples=1,
        )


def test_failed_checkpoint_is_retried_and_completion_is_strict(tmp_path: Path) -> None:
    attempts = 0

    def flaky(row: dict) -> dict:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary")
        return {"verdict": "correct"}

    output = tmp_path / "task"
    summary = run_formal_task(
        [_row("s1")],
        cell_id="cell",
        task="answer_quality",
        output_dir=output,
        evaluator=flaky,
        evaluator_identity={"model": "judge"},
        expected_samples=1,
    )
    assert summary["failed_samples"] == 1
    assert not (output / ".complete").exists()

    summary = run_formal_task(
        [_row("s1")],
        cell_id="cell",
        task="answer_quality",
        output_dir=output,
        evaluator=flaky,
        evaluator_identity={"model": "judge"},
        expected_samples=1,
    )
    assert summary["successful_samples"] == 1
    assert validate_formal_task_output(
        output,
        expected_sample_ids=("s1",),
        evaluator_identity={"model": "judge"},
    )["status"] == "complete"


class FakeClient:
    def __init__(self, text: str) -> None:
        self.text = text
        self.messages: list[list[dict[str, str]]] = []
        self.calls: list[dict[str, object]] = []

    def generate(
        self, messages: list[dict[str, str]], **kwargs: object
    ) -> SimpleNamespace:
        self.messages.append(messages)
        self.calls.append(kwargs)
        return SimpleNamespace(text=self.text, model="qwen3.7-max")


def test_answer_and_rag_adapters_use_existing_structured_parsers() -> None:
    answer_client = FakeClient(
        json.dumps(
            {
                "verdict": "correct",
                "correctness": 5,
                "completeness": 5,
                "relevance": 5,
                "conciseness": 4,
                "has_factual_conflict": False,
                "has_unsupported_claim": False,
                "missing_key_points": [],
                "error_summary": "",
            }
        )
    )
    answer = make_answer_quality_evaluator(answer_client)(_row("s1"))
    assert answer["verdict"] == "correct"
    assert answer["judgment_type"] == "answer_quality"
    assert "API" not in json.dumps(answer)
    assert answer_client.calls == [{"max_tokens": 512, "temperature": 0.0}]

    explicit_options = {
        "enable_thinking": False,
        "response_format": {"type": "json_object"},
    }
    configured_client = FakeClient(answer_client.text)
    configured_evaluator = make_answer_quality_evaluator(
        configured_client, extra_body=explicit_options
    )
    explicit_options["response_format"]["type"] = "mutated"
    configured_evaluator(_row("s1"))
    assert configured_client.calls == [
        {
            "max_tokens": 512,
            "temperature": 0.0,
            "extra_body": {
                "enable_thinking": False,
                "response_format": {"type": "json_object"},
            },
        }
    ]

    refusal_client = FakeClient(
        json.dumps(
            {
                "verdict": "appropriate_refusal",
                "refusal_appropriateness": 5,
                "grounding": 5,
                "relevance": 5,
                "conciseness": 5,
                "has_unsupported_answer": False,
                "claims_sufficient_evidence": False,
                "cites_irrelevant_evidence": False,
                "error_summary": "",
            }
        )
    )
    refusal_row = {**_row("s2"), "answerable": False, "evidence_text": ""}
    refusal = make_answer_quality_evaluator(refusal_client)(refusal_row)
    assert refusal["verdict"] == "appropriate_refusal"
    assert refusal["judgment_type"] == "refusal_quality"

    rag_client = FakeClient(
        json.dumps(
            {
                "reference_claims": ["r"],
                "candidate_claims": [
                    {
                        "text": "r",
                        "material": True,
                        "supported": True,
                        "matched_reference": 0,
                    }
                ],
                "citation_assessments": [{"citation_id": "1", "supported": True}],
            }
        )
    )
    row = {
        **_row("s1"),
        "evidence_text": "[RETRIEVED 1] evidence",
        "answerable": True,
        "citation": {
            "status": "ok",
            "precision": 1.0,
            "recall": 0.5,
            "f1": 2 / 3,
        },
    }
    rag = make_formal_rag_quality_evaluator(rag_client, citation_mode="retrieved_positions")(row)
    assert rag["claim_recall"] == 1.0
    assert rag["citation_semantic_precision"] == 1.0
    assert rag["citation_deterministic"]["recall"] == 0.5

    g0_row = {**row, "citation_status": "inapplicable", "citation_reason": "not_requested"}
    g0 = make_formal_rag_quality_evaluator(rag_client, citation_mode="not_requested")(g0_row)
    assert g0["citation_status"] == "inapplicable"
    assert g0["citation_reason"] == "not_requested"
    assert "citation_deterministic" not in g0


def test_matrix_complete_requires_all_declared_task_cells(tmp_path: Path) -> None:
    cells = (
        P0RagCell("m/z1", Path("z1"), ("answer_quality",), "none", "none"),
        P0RagCell(
            "m/g0",
            Path("g0"),
            ("answer_quality", "rag_quality"),
            "released_gold",
            "not_requested",
        ),
    )
    for cell in cells:
        for task in cell.tasks:
            directory = tmp_path / task / cell.cell_id / "full"
            directory.mkdir(parents=True)
            summary = {
                "cell_id": cell.cell_id,
                "task": task,
                "requested_samples": 2,
                "successful_samples": 2,
                "failed_samples": 0,
            }
            (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            (directory / ".complete").write_text(json.dumps(summary), encoding="utf-8")
    missing = tmp_path / "rag_quality" / "m/g0" / "full" / ".complete"
    missing.unlink()
    report = write_matrix_summary(cells, output_root=tmp_path, expected_samples=2)
    assert report["status"] == "incomplete"
    assert not (tmp_path / ".complete").exists()

    missing.write_text("{}", encoding="utf-8")
    report = write_matrix_summary(cells, output_root=tmp_path, expected_samples=2)
    assert report["status"] == "complete"
    assert report["completed_task_cells"] == 3
    assert (tmp_path / ".complete").is_file()
