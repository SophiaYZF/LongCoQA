from longcoqa.retrieval.rewrite import build_rewrite_messages


def test_rewrite_messages_include_teacher_forced_history() -> None:
    samples = [
        {
            "sample_id": "factoid:1:1",
            "question_type": "factoid",
            "dialogue_id": 1,
            "turn": 1,
            "question": "甲是谁？",
            "answer": "乙。",
        },
        {
            "sample_id": "factoid:1:2",
            "question_type": "factoid",
            "dialogue_id": 1,
            "turn": 2,
            "question": "他在哪里？",
            "answer": "丙。",
        },
    ]
    _, messages = build_rewrite_messages(samples)
    assert "（无）" in messages[0][-1]["content"]
    assert "甲是谁" in messages[1][-1]["content"]
    assert "乙" in messages[1][-1]["content"]

