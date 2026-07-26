from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from longcoqa.training.evidence_reconstruction import (
    INSUFFICIENT_ANSWER,
    build_aligned_datasets,
    build_human_audit_packet,
    build_quality_report,
    build_teacher_messages,
    load_reconstruction_rows,
    parse_teacher_response,
    render_clean_output,
    render_evidence_aware_output,
    run_reconstruction,
    select_human_audit_rows,
)


def _teacher_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "sample_id": "factoid:1:1",
        "sufficiency": "sufficient",
        "selected_evidence_ids": ["1192"],
        "support_points": [
            {"evidence_id": "1192", "claim": "塞蒂娜·沃贝克是女巫歌唱家。"}
        ],
        "final_answer": "塞蒂娜·沃贝克是一名女巫歌唱家。",
        "risk_flags": [],
    }
    payload.update(overrides)
    return payload


def test_parse_teacher_response_accepts_fenced_json_and_normalizes_ids() -> None:
    payload = _teacher_payload(
        selected_evidence_ids=["EVIDENCE 1192"],
        support_points=[
            {"evidence_id": "[EVIDENCE 1192]", "claim": "她是女巫歌唱家。"}
        ],
    )
    parsed = parse_teacher_response(
        f"```json\n{json.dumps(payload, ensure_ascii=False)}\n```",
        expected_sample_id="factoid:1:1",
        available_evidence_ids={"1192", "NEG-1"},
    )
    assert parsed["selected_evidence_ids"] == ["1192"]
    assert parsed["support_points"][0]["evidence_id"] == "1192"


def test_parse_teacher_response_rejects_unknown_evidence_id() -> None:
    payload = _teacher_payload(selected_evidence_ids=["9999"])
    with pytest.raises(ValueError, match="unknown evidence"):
        parse_teacher_response(
            json.dumps(payload, ensure_ascii=False),
            expected_sample_id="factoid:1:1",
            available_evidence_ids={"1192"},
        )


def test_parse_teacher_response_requires_support_ids_to_match_selected_ids() -> None:
    payload = _teacher_payload(selected_evidence_ids=["1192", "NEG-1"])
    with pytest.raises(ValueError, match="support-point evidence IDs"):
        parse_teacher_response(
            json.dumps(payload, ensure_ascii=False),
            expected_sample_id="factoid:1:1",
            available_evidence_ids={"1192", "NEG-1"},
        )


def test_parse_teacher_response_enforces_insufficient_contract() -> None:
    payload = _teacher_payload(
        sufficiency="insufficient",
        selected_evidence_ids=[],
        support_points=[],
        final_answer=INSUFFICIENT_ANSWER,
    )
    parsed = parse_teacher_response(
        json.dumps(payload, ensure_ascii=False),
        expected_sample_id="factoid:1:1",
        available_evidence_ids={"1192"},
    )
    assert parsed["final_answer"] == INSUFFICIENT_ANSWER

    payload["selected_evidence_ids"] = ["1192"]
    with pytest.raises(ValueError, match="must not select evidence"):
        parse_teacher_response(
            json.dumps(payload, ensure_ascii=False),
            expected_sample_id="factoid:1:1",
            available_evidence_ids={"1192"},
        )


def test_renderers_share_the_exact_clean_final_answer() -> None:
    label = parse_teacher_response(
        json.dumps(_teacher_payload(), ensure_ascii=False),
        expected_sample_id="factoid:1:1",
        available_evidence_ids={"1192"},
    )
    clean = render_clean_output(label)
    planned = render_evidence_aware_output(label)
    assert clean == "塞蒂娜·沃贝克是一名女巫歌唱家。\n\n引用：[EVIDENCE 1192]"
    assert planned.endswith("最终答案: " + clean)


def test_teacher_prompt_marks_history_and_old_answer_as_untrusted() -> None:
    row = {
        "history": [["上一问", "旧回答"]],
        "instruction": "[EVIDENCE 1]\n文本\n\n[QUESTION]\n问题",
        "output": "原始答案",
        "metadata": {"sample_id": "factoid:1:1"},
    }
    messages = build_teacher_messages(row)
    text = "\n".join(message["content"] for message in messages)
    assert "仅用于指代消解" in text
    assert "不可信候选" in text
    assert "不得依赖模型记忆" in text
    assert "factoid:1:1" in text


def _raft_row(sample_id: str, *, split: str = "train", turn: int = 1) -> dict[str, object]:
    question_type, dialogue_id, _ = sample_id.split(":")
    return {
        "history": [],
        "input": "",
        "instruction": f"[EVIDENCE 1]\n证据文本\n\n[QUESTION]\n第{turn}问",
        "output": f"原始答案{turn}\n\n引用：[EVIDENCE 1]",
        "system": "raw system",
        "metadata": {
            "sample_id": sample_id,
            "split": split,
            "variant": "raft",
            "question_type": question_type,
            "dialogue_id": int(dialogue_id),
            "turn": turn,
            "raft_mode": "gold_only",
        },
    }


def _label(sample_id: str, answer: str) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "sufficiency": "sufficient",
        "selected_evidence_ids": ["1"],
        "support_points": [{"evidence_id": "1", "claim": answer}],
        "final_answer": answer,
        "risk_flags": [],
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_load_reconstruction_rows_rejects_duplicate_ids_and_test_metadata(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    dev = tmp_path / "dev.jsonl"
    duplicate = _raft_row("factoid:1:1")
    _write_jsonl(train, [duplicate])
    _write_jsonl(dev, [duplicate])
    with pytest.raises(ValueError, match="duplicate sample_id"):
        load_reconstruction_rows({"train": train, "dev": dev})

    forbidden = _raft_row("factoid:2:1", split="test")
    _write_jsonl(dev, [forbidden])
    with pytest.raises(ValueError, match="split mismatch"):
        load_reconstruction_rows({"train": train, "dev": dev})


def test_run_reconstruction_reuses_valid_checkpoint(tmp_path: Path) -> None:
    row = _raft_row("factoid:1:1")

    class FakeClient:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, *_args: object, **_kwargs: object) -> SimpleNamespace:
            self.calls += 1
            return SimpleNamespace(
                text=json.dumps(_label("factoid:1:1", "清洗答案"), ensure_ascii=False),
                provider="fake",
                model="teacher",
                usage={"total_tokens": 10},
                finish_reason="stop",
                request_hash="request-hash",
                cached=False,
                latency_seconds=0.1,
            )

    client = FakeClient()
    first = run_reconstruction(
        [row], client=client, checkpoint_dir=tmp_path / "checkpoints", max_workers=1
    )
    second = run_reconstruction(
        [row], client=client, checkpoint_dir=tmp_path / "checkpoints", max_workers=1
    )
    assert first["accepted"] == second["accepted"] == 1
    assert second["reused"] == 1
    assert client.calls == 1


def test_run_reconstruction_cancels_queued_work_before_propagating_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [_raft_row(f"factoid:{index}:1") for index in range(1, 21)]

    class InterruptingClient:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, messages: list[dict[str, str]], **_kwargs: object) -> SimpleNamespace:
            self.calls += 1
            sample_id = messages[1]["content"].split("[SAMPLE_ID]\n", 1)[1].splitlines()[0]
            return SimpleNamespace(
                text=json.dumps(_label(sample_id, "答案"), ensure_ascii=False),
                provider="fake",
                model="teacher",
                usage={},
                finish_reason="stop",
                request_hash=str(self.calls),
                cached=False,
                latency_seconds=0.0,
            )

    def interrupt_before_collecting(_futures: object) -> object:
        raise KeyboardInterrupt

    monkeypatch.setattr(
        "longcoqa.training.evidence_reconstruction.as_completed",
        interrupt_before_collecting,
    )
    client = InterruptingClient()
    with pytest.raises(KeyboardInterrupt):
        run_reconstruction(
            rows,
            client=client,
            checkpoint_dir=tmp_path / "checkpoints",
            max_workers=2,
        )
    assert client.calls <= 2


def test_build_aligned_datasets_rebuilds_clean_history_and_common_rejections(
    tmp_path: Path,
) -> None:
    rows = [
        _raft_row("factoid:1:1", turn=1),
        _raft_row("factoid:1:2", turn=2),
        _raft_row("factoid:2:1", turn=1),
    ]
    labels = {
        "factoid:1:1": _label("factoid:1:1", "清洗答案1"),
        "factoid:1:2": _label("factoid:1:2", "清洗答案2"),
    }
    report = build_aligned_datasets(rows, labels, output_dir=tmp_path / "out")
    assert report["counts"]["train"] == 2
    assert report["rejected_sample_ids"] == ["factoid:2:1"]

    variants = {}
    for variant in ("raw_answer", "clean_answer", "evidence_aware"):
        path = tmp_path / "out" / f"{variant}_train.jsonl"
        variants[variant] = [json.loads(line) for line in path.read_text().splitlines()]
        assert [row["metadata"]["sample_id"] for row in variants[variant]] == [
            "factoid:1:1",
            "factoid:1:2",
        ]
        assert variants[variant][1]["history"] == [["第1问", "清洗答案1\n\n引用：[EVIDENCE 1]"]]

    clean_answer = variants["clean_answer"][1]["output"]
    assert variants["evidence_aware"][1]["output"].endswith("最终答案: " + clean_answer)
    assert variants["raw_answer"][1]["output"].startswith("原始答案2")


def _audit_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    index = 0
    for question_type in ("factoid", "non_factoid"):
        for sufficiency in ("sufficient", "partial", "insufficient"):
            for repeat in range(3):
                index += 1
                records.append(
                    {
                        "sample_id": f"{question_type}:{index}:1",
                        "question_type": question_type,
                        "sufficiency": sufficiency,
                        "risk_flags": ["evidence_conflict"] if repeat == 0 else [],
                    }
                )
    return records


def test_human_audit_selector_is_deterministic_stratified_and_risk_first() -> None:
    selected = select_human_audit_rows(_audit_records(), count=12, seed=42)
    repeated = select_human_audit_rows(list(reversed(_audit_records())), count=12, seed=42)
    assert [row["sample_id"] for row in selected] == [row["sample_id"] for row in repeated]
    assert {row["question_type"] for row in selected} == {"factoid", "non_factoid"}
    assert {row["sufficiency"] for row in selected} == {
        "sufficient",
        "partial",
        "insufficient",
    }
    assert sum(bool(row["risk_flags"]) for row in selected) == 6


def test_quality_report_never_claims_human_or_semantic_pass() -> None:
    rows = [_raft_row("factoid:1:1")]
    labels = {"factoid:1:1": _label("factoid:1:1", "清洗答案")}
    report = build_quality_report(
        rows,
        labels,
        data_report={"accepted": 1, "rejected": 0, "counts": {"train": 1, "dev": 0}},
        input_paths={},
    )
    assert report["status"] == "AWAITING_HUMAN_AUDIT"
    assert report["semantic_support_verified"] is False
    assert report["human_review"]["completed"] is False
    assert report["automatic_checks"]["test_rows_read"] == 0


def test_human_audit_packet_has_blank_reviewer_fields(tmp_path: Path) -> None:
    rows = [_raft_row("factoid:1:1")]
    labels = {"factoid:1:1": _label("factoid:1:1", "清洗答案")}
    packet = build_human_audit_packet(
        rows, labels, output_dir=tmp_path, count=250, seed=42
    )
    assert packet["rows"] == 1
    exported = json.loads((tmp_path / "human_audit_250.jsonl").read_text().strip())
    assert exported["reviewer_sufficiency"] == ""
    assert exported["reviewer_evidence_correct"] == ""
    assert exported["reviewer_answer_supported"] == ""
    assert (tmp_path / "human_audit_250.csv").is_file()
    assert "AWAITING_HUMAN_AUDIT" in (tmp_path / "HUMAN_AUDIT.md").read_text()
