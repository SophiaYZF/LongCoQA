from __future__ import annotations

from longcoqa.inference.prompts import build_messages, index_samples


def _samples() -> list[dict]:
    return [
        {
            "sample_id": "factoid:1:1",
            "question_type": "factoid",
            "dialogue_id": 1,
            "turn": 1,
            "question": "第一问？",
            "answer": "第一答。",
            "evidence_ids": [1],
        },
        {
            "sample_id": "factoid:1:2",
            "question_type": "factoid",
            "dialogue_id": 1,
            "turn": 2,
            "question": "第二问？",
            "answer": "第二答。",
            "evidence_ids": [2],
        },
    ]


def test_teacher_forced_history_uses_reference_answer() -> None:
    samples = _samples()
    messages = build_messages(
        samples[1],
        protocol="z1_teacher_forced",
        dialogues=index_samples(samples),
        released_evidence={},
        aligned_source={},
    )

    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert messages[2]["content"] == "第一答。"
    assert messages[-1]["content"] == "第二问？"


def test_gold_protocol_uses_only_current_turn_evidence() -> None:
    samples = _samples()
    messages = build_messages(
        samples[1],
        protocol="g3_aligned_raw",
        dialogues=index_samples(samples),
        released_evidence={2: "released"},
        aligned_source={2: "raw source"},
    )

    assert "[EVIDENCE 2]" in messages[-1]["content"]
    assert "raw source" in messages[-1]["content"]
    assert "[EVIDENCE 1]" not in messages[-1]["content"]
