from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import pytest

from longcoqa.config import ProviderSettings
from longcoqa.inference.free_running import (
    build_free_running_messages,
    group_dialogue_samples,
    history_hash,
    run_free_running_inference,
    validate_free_running_chain,
)
from longcoqa.providers.client import GenerationResult
from longcoqa.utils.io import read_jsonl


def _sample(
    *,
    sample_id: str,
    question_type: str = "factoid",
    dialogue_id: int = 1,
    turn: int = 1,
    question: str | None = None,
    answer: str | None = None,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "question_type": question_type,
        "dialogue_id": dialogue_id,
        "turn": turn,
        "question": question or f"Q{turn}",
        "answer": answer or f"GOLD{turn}",
        "evidence_ids": [],
    }


def _result(
    *,
    sample_id: str,
    turn: int,
    question: str,
    prediction: str,
    prior: list[tuple[str, str]],
) -> dict[str, object]:
    return {
        "status": "ok",
        "sample_id": sample_id,
        "question_type": "factoid",
        "dialogue_id": 1,
        "turn": turn,
        "protocol": "z2_free_running",
        "model_requested": "claude-opus-4-6",
        "question": question,
        "reference": f"GOLD{turn}",
        "prediction": prediction,
        "history_hash": history_hash(prior),
        "messages": build_free_running_messages(
            {"question": question},
            prior,
        ),
    }


def test_build_free_running_messages_uses_predictions_not_gold() -> None:
    sample = {"question": "Q3", "answer": "GOLD3"}
    prior = [("Q1", "PRED1"), ("Q2", "PRED2")]

    messages = build_free_running_messages(sample, prior)

    assert messages[1:] == [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "PRED1"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "PRED2"},
        {"role": "user", "content": "Q3"},
    ]
    assert "GOLD3" not in repr(messages)


def test_group_dialogue_samples_orders_dialogues_and_turns() -> None:
    rows = [
        _sample(sample_id="non_factoid:2:2", question_type="non_factoid", dialogue_id=2, turn=2),
        _sample(sample_id="factoid:1:2", turn=2),
        _sample(sample_id="factoid:1:1", turn=1),
        _sample(sample_id="non_factoid:2:1", question_type="non_factoid", dialogue_id=2, turn=1),
    ]

    grouped = group_dialogue_samples(rows)

    assert list(grouped) == [("factoid", 1), ("non_factoid", 2)]
    assert [[row["turn"] for row in dialogue] for dialogue in grouped.values()] == [
        [1, 2],
        [1, 2],
    ]


def test_validate_free_running_chain_accepts_prediction_history() -> None:
    first = _result(
        sample_id="factoid:1:1",
        turn=1,
        question="Q1",
        prediction="PRED1",
        prior=[],
    )
    second = _result(
        sample_id="factoid:1:2",
        turn=2,
        question="Q2",
        prediction="PRED2",
        prior=[("Q1", "PRED1")],
    )

    report = validate_free_running_chain([first, second])

    assert report == {"dialogues": 1, "rows": 2, "gold_history_messages": 0}


def test_validate_free_running_chain_rejects_gold_history() -> None:
    first = _result(
        sample_id="factoid:1:1",
        turn=1,
        question="Q1",
        prediction="PRED1",
        prior=[],
    )
    second = _result(
        sample_id="factoid:1:2",
        turn=2,
        question="Q2",
        prediction="PRED2",
        prior=[("Q1", "GOLD1")],
    )

    with pytest.raises(ValueError, match="history hash|assistant history"):
        validate_free_running_chain([first, second])


@dataclass
class _Dataset:
    rows: list[dict[str, object]]

    def iter_samples(self):  # type: ignore[no-untyped-def]
        yield from self.rows


class _FakeClient:
    def __init__(
        self,
        settings: ProviderSettings,
        *,
        cache=None,  # noqa: ANN001
        fail_once: set[str] | None = None,
        calls: list[list[dict[str, str]]] | None = None,
    ) -> None:
        self.settings = settings
        self.fail_once = fail_once if fail_once is not None else set()
        self.calls = calls if calls is not None else []
        self._lock = threading.Lock()

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
    ) -> GenerationResult:
        del max_tokens, temperature
        question = messages[-1]["content"]
        with self._lock:
            self.calls.append(messages)
            if question in self.fail_once:
                self.fail_once.remove(question)
                raise RuntimeError(f"transient: {question}")
        return GenerationResult(
            provider=self.settings.name,
            model=self.settings.model,
            text=f"PRED:{question}",
            usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            finish_reason="stop",
            request_hash=f"hash:{question}",
            cached=False,
            latency_seconds=0.1,
        )

    def __enter__(self) -> _FakeClient:
        return self

    def __exit__(self, *_: object) -> None:
        return None


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


def _dialogue_rows() -> list[dict[str, object]]:
    return [
        _sample(sample_id="factoid:1:1", dialogue_id=1, turn=1, question="A1"),
        _sample(sample_id="factoid:1:2", dialogue_id=1, turn=2, question="A2"),
        _sample(sample_id="factoid:1:3", dialogue_id=1, turn=3, question="A3"),
        _sample(sample_id="factoid:2:1", dialogue_id=2, turn=1, question="B1"),
        _sample(sample_id="factoid:2:2", dialogue_id=2, turn=2, question="B2"),
        _sample(sample_id="factoid:2:3", dialogue_id=2, turn=3, question="B3"),
    ]


def test_runner_uses_serial_turns_with_parallel_dialogues(tmp_path: Path) -> None:
    calls: list[list[dict[str, str]]] = []

    summary = run_free_running_inference(
        _Dataset(_dialogue_rows()),
        _settings(),
        cache_path=tmp_path / "cache.sqlite3",
        output_dir=tmp_path / "formal",
        client_factory=lambda settings, cache: _FakeClient(settings, cache=cache, calls=calls),
    )

    results = read_jsonl(tmp_path / "formal" / "results.jsonl")
    assert summary["successful_samples"] == 6
    assert summary["failed_samples"] == 0
    assert [row["sample_id"] for row in results] == [row["sample_id"] for row in _dialogue_rows()]
    a2 = next(messages for messages in calls if messages[-1]["content"] == "A2")
    assert {"role": "assistant", "content": "PRED:A1"} in a2
    b3 = next(messages for messages in calls if messages[-1]["content"] == "B3")
    assert {"role": "assistant", "content": "PRED:B2"} in b3


def test_runner_resume_blocks_failed_dialogue_and_only_fills_missing_turns(
    tmp_path: Path,
) -> None:
    first_calls: list[list[dict[str, str]]] = []
    output_dir = tmp_path / "formal"
    run_free_running_inference(
        _Dataset(_dialogue_rows()),
        _settings(),
        cache_path=tmp_path / "cache.sqlite3",
        output_dir=output_dir,
        client_factory=lambda settings, cache: _FakeClient(
            settings,
            cache=cache,
            fail_once={"A2"},
            calls=first_calls,
        ),
    )
    assert "A3" not in {messages[-1]["content"] for messages in first_calls}

    second_calls: list[list[dict[str, str]]] = []
    summary = run_free_running_inference(
        _Dataset(_dialogue_rows()),
        _settings(),
        cache_path=tmp_path / "cache.sqlite3",
        output_dir=output_dir,
        client_factory=lambda settings, cache: _FakeClient(
            settings,
            cache=cache,
            calls=second_calls,
        ),
    )

    assert [messages[-1]["content"] for messages in second_calls] == ["A2", "A3"]
    assert summary["successful_samples"] == 6
    assert summary["failed_samples"] == 0
