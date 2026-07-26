import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from longcoqa.evaluation.text_metrics import score_answer
from longcoqa.inference.gpt_oss_runner import (
    _gpt_oss_checkpoint_current,
    _run_gpt_oss_requests,
    build_gpt_oss_result,
    contains_harmony_control,
    parse_harmony_candidate,
)
from longcoqa.paths import PROJECT_ROOT


def test_parse_harmony_candidate_separates_final_from_reasoning() -> None:
    def parser(_: list[int]) -> tuple[str | None, str | None, bool]:
        return "private reasoning", "最终答案", False

    final, reasoning = parse_harmony_candidate([1, 2], parser=parser)
    assert final == "最终答案"
    assert reasoning == "private reasoning"
    assert score_answer(final, "最终答案")["exact_match"] == 1.0


def test_parse_harmony_candidate_rejects_missing_final() -> None:
    with pytest.raises(ValueError, match="missing final"):
        parse_harmony_candidate(
            [1], parser=lambda _: ("reasoning", None, False)
        )


def test_parse_harmony_candidate_rejects_control_tokens() -> None:
    assert contains_harmony_control("<|channel|>analysis")
    with pytest.raises(ValueError, match="control token"):
        parse_harmony_candidate(
            [], parser=lambda _: (None, "<|channel|>analysis", False)
        )


def test_gpt_oss_result_scores_only_final() -> None:
    sample = {
        "sample_id": "factoid:1:1",
        "question_type": "factoid",
        "dialogue_id": "1",
        "turn": 1,
        "question": "问题",
        "answer": "答案",
        "evidence_ids": ["e1"],
    }
    candidate = SimpleNamespace(
        token_ids=[1, 2], text="raw Harmony text", finish_reason="stop"
    )
    request_output = SimpleNamespace(
        prompt_token_ids=[10, 11, 12], outputs=[candidate]
    )

    result = build_gpt_oss_result(
        sample,
        [{"role": "user", "content": "问题"}],
        request_output,
        model={
            "name": "gpt-oss-120b",
            "response_format": "harmony",
            "reasoning_effort": "low",
        },
        protocol="z0_current",
        generation_max_tokens=1024,
        parser=lambda _: ("trace", "答案", False),
    )

    assert result["prediction"] == "答案"
    assert result["reasoning"] == "trace"
    assert result["reasoning_effort"] == "low"
    assert result["response_format"] == "harmony"
    assert result["generation_max_tokens"] == 1024
    assert result["metrics"]["exact_match"] == 1.0
    assert result["usage"] == {"prompt_tokens": 3, "completion_tokens": 2}


def test_gpt_oss_result_records_missing_final_without_raising() -> None:
    sample = {
        "sample_id": "factoid:1:1",
        "question_type": "factoid",
        "dialogue_id": "1",
        "turn": 1,
        "question": "问题",
        "answer": "答案",
        "evidence_ids": ["e1"],
    }
    candidate = SimpleNamespace(
        token_ids=[1, 2], text="raw Harmony text", finish_reason="length"
    )
    request_output = SimpleNamespace(
        prompt_token_ids=[10, 11, 12], outputs=[candidate]
    )

    result = build_gpt_oss_result(
        sample,
        [{"role": "user", "content": "问题"}],
        request_output,
        model={
            "name": "gpt-oss-120b",
            "response_format": "harmony",
            "reasoning_effort": "low",
        },
        protocol="z0_current",
        generation_max_tokens=1024,
        parser=lambda _: ("unfinished reasoning", None, False),
    )

    assert result["status"] == "failed"
    assert result["error"] == {
        "type": "ValueError",
        "message": "missing final Harmony channel",
    }
    assert result["finish_reason"] == "length"
    assert result["usage"]["completion_tokens"] == 2
    assert result["generation_max_tokens"] == 1024
    assert "metrics" not in result


def test_gpt_oss_result_records_harmony_runtime_error_without_raising() -> None:
    sample = {
        "sample_id": "factoid:59:12",
        "question_type": "factoid",
        "dialogue_id": "59",
        "turn": 12,
        "question": "另外的那个人曾摧毁过哪个魂器？",
        "answer": "纳吉尼。",
        "evidence_ids": ["e1"],
    }
    candidate = SimpleNamespace(
        token_ids=[1, 2], text="malformed Harmony", finish_reason="length"
    )
    request_output = SimpleNamespace(
        prompt_token_ids=[10, 11, 12], outputs=[candidate]
    )

    def malformed_harmony(_: list[int]) -> tuple[str | None, str | None, bool]:
        raise RuntimeError("unexpected tokens remaining in message header")

    result = build_gpt_oss_result(
        sample,
        [{"role": "user", "content": sample["question"]}],
        request_output,
        model={
            "name": "gpt-oss-120b",
            "response_format": "harmony",
            "reasoning_effort": "low",
        },
        protocol="z1_teacher_forced",
        generation_max_tokens=1024,
        parser=malformed_harmony,
    )

    assert result["status"] == "failed"
    assert result["error"] == {
        "type": "RuntimeError",
        "message": "unexpected tokens remaining in message header",
    }
    assert "prediction" not in result


def test_gpt_oss_requests_retry_only_malformed_outputs_individually() -> None:
    samples = [
        {
            "sample_id": f"factoid:1:{turn}",
            "question_type": "factoid",
            "dialogue_id": "1",
            "turn": turn,
            "question": f"问题{turn}",
            "answer": f"答案{turn}",
            "evidence_ids": [],
        }
        for turn in (1, 2)
    ]
    messages = [[{"role": "user", "content": sample["question"]}] for sample in samples]

    def output(token_id: int) -> SimpleNamespace:
        return SimpleNamespace(
            prompt_token_ids=[10],
            outputs=[
                SimpleNamespace(
                    token_ids=[token_id], text="raw", finish_reason="stop"
                )
            ],
        )

    class FakeEngine:
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def chat(self, batch: list, **_: object) -> list[SimpleNamespace]:
            self.batch_sizes.append(len(batch))
            return [output(0), output(2)] if len(batch) == 2 else [output(1)]

    def parser(token_ids: list[int]) -> tuple[str | None, str | None, bool]:
        if token_ids == [0]:
            raise RuntimeError("malformed Harmony header")
        return None, "答案", False

    engine = FakeEngine()
    results = _run_gpt_oss_requests(
        engine,
        samples,
        messages,
        [object(), object()],
        model={
            "name": "gpt-oss-120b",
            "response_format": "harmony",
            "reasoning_effort": "low",
        },
        protocol="z1_teacher_forced",
        generation_budget=lambda _: 1024,
        parser=parser,
    )

    assert engine.batch_sizes == [2, 1]
    assert [result["status"] for result in results] == ["ok", "ok"]
    assert [result["generation_attempt"] for result in results] == [2, 1]


def test_gpt_oss_requests_use_format_repair_after_plain_retry_fails() -> None:
    sample = {
        "sample_id": "factoid:59:12",
        "question_type": "factoid",
        "dialogue_id": "59",
        "turn": 12,
        "question": "另外的那个人曾摧毁过哪个魂器？",
        "answer": "纳吉尼。",
        "evidence_ids": [],
    }

    def output(token_id: int) -> SimpleNamespace:
        return SimpleNamespace(
            prompt_token_ids=[10],
            outputs=[
                SimpleNamespace(
                    token_ids=[token_id], text="raw", finish_reason="stop"
                )
            ],
        )

    class FakeEngine:
        def __init__(self) -> None:
            self.messages: list[list[dict[str, str]]] = []

        def chat(self, batch: list, **_: object) -> list[SimpleNamespace]:
            self.messages.append(batch[0])
            return [output(0 if len(self.messages) < 3 else 1)]

    def parser(token_ids: list[int]) -> tuple[str | None, str | None, bool]:
        if token_ids == [0]:
            raise RuntimeError("malformed Harmony header")
        return None, "纳吉尼。", False

    engine = FakeEngine()
    results = _run_gpt_oss_requests(
        engine,
        [sample],
        [[{"role": "user", "content": sample["question"]}]],
        [object()],
        model={
            "name": "gpt-oss-120b",
            "response_format": "harmony",
            "reasoning_effort": "low",
        },
        protocol="z1_teacher_forced",
        generation_budget=lambda _: 2048,
        parser=parser,
    )

    assert results[0]["status"] == "ok"
    assert results[0]["generation_attempt"] == 3
    assert engine.messages[0] == engine.messages[1]
    assert engine.messages[2][-1]["content"].endswith(
        "[格式修复] 请仅输出简短的中文最终答案。"
    )


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"status": "failed", "generation_max_tokens": 1024}, False),
        ({"status": "ok"}, False),
        ({"status": "ok", "generation_max_tokens": 256}, False),
        ({"status": "ok", "generation_max_tokens": 1024}, True),
    ],
)
def test_gpt_oss_checkpoint_requires_successful_matching_budget(
    tmp_path: Path, payload: dict[str, object], expected: bool
) -> None:
    path = tmp_path / "checkpoint.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert _gpt_oss_checkpoint_current(path, 1024) is expected


def test_gpt_oss_cli_exposes_protocol_and_sharding_options() -> None:
    source = (PROJECT_ROOT / "longcoqa/cli.py").read_text(encoding="utf-8")

    assert '@app.command("run-gpt-oss-suite")' in source
    assert 'typer.Option(None, "--protocol")' in source
    assert "run_gpt_oss_suite(" in source
    assert 'model["max_tokens_factoid"]' in source
    assert 'model["max_tokens_non_factoid"]' in source
    assert "shard_index=shard_index" in source
    assert "shard_count=shard_count" in source
