import json
from pathlib import Path

from longcoqa.inference.p1_runner import (
    is_abstention,
    load_p1_records,
    parse_citation_ids,
    strip_citation_footer,
    summarize_p1_results,
    training_record_to_messages,
)


def _record(sample_id: str, question_type: str, raft_mode: str | None = None) -> dict:
    metadata = {
        "sample_id": sample_id,
        "split": "test",
        "variant": "raft",
        "question_type": question_type,
        "dialogue_id": 1,
        "turn": 2,
    }
    if raft_mode:
        metadata["raft_mode"] = raft_mode
    return {
        "system": "system",
        "history": [["q1", "a1"]],
        "instruction": "q2",
        "input": "",
        "output": "a2",
        "metadata": metadata,
    }


def test_training_record_to_messages() -> None:
    messages = training_record_to_messages(_record("factoid:1:2", "factoid"))
    assert messages == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ]


def test_parse_prediction_annotations() -> None:
    prediction = "答案。\n\n引用：[EVIDENCE 12, EVIDENCE 18, EVIDENCE NEG-2]"
    assert strip_citation_footer(prediction) == "答案。"
    assert parse_citation_ids(prediction) == ["12", "18", "NEG-2"]
    assert is_abstention("提供的证据不足以回答该问题。")
    assert not is_abstention(prediction)


def test_load_p1_records_stratifies_limit(tmp_path: Path) -> None:
    rows = [
        _record("factoid:1:1", "factoid", "gold_only"),
        _record("factoid:1:2", "factoid", "missing_gold_abstention"),
        _record("non_factoid:1:1", "non_factoid", "gold_plus_hard_negatives"),
        _record("non_factoid:1:2", "non_factoid", "missing_gold_abstention"),
    ]
    path = tmp_path / "raft_test.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    selected = load_p1_records(path, eval_variant="raft", limit=3, seed=42)
    assert len(selected) == 3
    assert len({row["metadata"]["question_type"] for row in selected}) == 2


def test_load_p1_records_accepts_explicit_dev_split(tmp_path: Path) -> None:
    row = _record("factoid:1:1", "factoid")
    row["metadata"]["split"] = "dev"
    path = tmp_path / "raft_dev.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    selected = load_p1_records(
        path,
        eval_variant="raft",
        expected_split="dev",
    )

    assert selected[0]["metadata"]["split"] == "dev"


def test_summarize_p1_results() -> None:
    results = [
        {
            "status": "ok",
            "sample_id": "factoid:1:1",
            "question_type": "factoid",
            "prediction": "答案\n引用：[EVIDENCE 1]",
            "metrics": {"exact_match": 1.0, "character_f1": 1.0, "rouge_l_f1": 1.0},
            "answerable": True,
            "abstained": False,
            "gold_evidence_ids": ["1", "2"],
            "cited_evidence_ids": ["1"],
            "raft_mode": "gold_only",
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        },
        {
            "status": "ok",
            "sample_id": "non_factoid:1:1",
            "question_type": "non_factoid",
            "prediction": "提供的证据不足以回答该问题。",
            "metrics": None,
            "answerable": False,
            "abstained": True,
            "gold_evidence_ids": ["3"],
            "cited_evidence_ids": [],
            "raft_mode": "missing_gold_abstention",
            "usage": {"prompt_tokens": 20, "completion_tokens": 4},
        },
    ]
    summary = summarize_p1_results(
        results,
        model={"name": "test"},
        model_variant="raft",
        eval_variant="raft",
        seed=42,
    )
    assert summary["metrics"]["character_f1"] == 1.0
    assert summary["citation"]["precision"] == 1.0
    assert summary["citation"]["recall"] == 1 / 3
    assert summary["abstention"]["f1"] == 1.0
    assert summary["usage"] == {"prompt_tokens": 30, "completion_tokens": 6}
