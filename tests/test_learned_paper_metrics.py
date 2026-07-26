from __future__ import annotations

from collections.abc import Sequence

import pytest

from longcoqa.evaluation.paper_metric_profiles import (
    MetricCompatibility,
    MetricProfile,
)
from longcoqa.evaluation.paper_metrics import (
    BARTScoreAdapter,
    BatchMetricOutput,
    BERTScoreAdapter,
    BLEURTAdapter,
)
from longcoqa.evaluation.paper_reporting import score_batch_rows


def learned_profile(metric: str, **options: object) -> MetricProfile:
    defaults: dict[str, object] = {
        "profile_id": f"{metric}-locked-v1",
        "metrics": (metric,),
        "compatibility": MetricCompatibility.PAPER_NAMED_LOCKED,
        "implementation": f"longcoqa.{metric}",
        "normalization": "none",
        "tokenization": "model_tokenizer",
        "model_id": f"example/{metric}",
        "model_revision": "0123456789abcdef",
        "options": {"max_length": 16, **options},
    }
    return MetricProfile(**defaults)


def test_bertscore_adapter_preserves_precision_recall_f1() -> None:
    calls: list[dict[str, object]] = []

    def backend(
        predictions: Sequence[str], references: Sequence[str], **kwargs: object
    ) -> tuple[list[float], list[float], list[float]]:
        calls.append({"predictions": list(predictions), "references": list(references), **kwargs})
        return [0.2, 0.3], [0.4, 0.5], [0.6, 0.7]

    profile = learned_profile(
        "bertscore_f1",
        num_layers=9,
        precision_metric="bertscore_precision",
        recall_metric="bertscore_recall",
        f1_metric="bertscore_f1",
    )
    adapter = BERTScoreAdapter(profile, backend=backend, device="cuda:0", batch_size=8)
    output = adapter.score_batch(["a", "b"], ["x", "y"])

    assert output.scores == (
        {"bertscore_precision": 0.2, "bertscore_recall": 0.4, "bertscore_f1": 0.6},
        {"bertscore_precision": 0.3, "bertscore_recall": 0.5, "bertscore_f1": 0.7},
    )
    assert calls[0]["model_type"] == profile.model_id
    assert calls[0]["device"] == "cuda:0"
    assert calls[0]["batch_size"] == 8
    assert calls[0]["num_layers"] == 9


def test_bartscore_adapter_passes_locked_direction_and_marks_truncation() -> None:
    calls: list[dict[str, object]] = []

    def backend(
        predictions: Sequence[str], references: Sequence[str], **kwargs: object
    ) -> list[float]:
        calls.append({"predictions": list(predictions), "references": list(references), **kwargs})
        return [-1.5, -2.0]

    profile = learned_profile("bartscore", direction="reference_to_prediction")
    adapter = BARTScoreAdapter(profile, backend=backend, device="cpu", batch_size=4)
    output = adapter.score_batch(["short", "x" * 30], ["ref", "ref"])

    assert output.scores == ({"bartscore": -1.5}, {"bartscore": -2.0})
    assert output.truncated == (False, True)
    assert calls[0]["direction"] == "reference_to_prediction"
    assert calls[0]["max_length"] == 16


def test_bleurt_adapter_returns_one_score_per_pair() -> None:
    profile = learned_profile("bleurt")
    adapter = BLEURTAdapter(profile, backend=lambda predictions, references, **kwargs: [0.8])

    output = adapter.score_batch(["answer"], ["reference"])

    assert output.scores == ({"bleurt": 0.8},)
    assert output.truncated == (False,)


def test_default_bleurt_backend_uses_bleurt_pytorch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imports: list[str] = []

    def fake_optional_module(name: str) -> object:
        imports.append(name)
        return object()

    monkeypatch.setattr(
        "longcoqa.evaluation.paper_metrics._optional_module", fake_optional_module
    )

    BLEURTAdapter(learned_profile("bleurt"))

    assert imports == ["torch", "bleurt_pytorch"]


@pytest.mark.parametrize("adapter_class", [BERTScoreAdapter, BARTScoreAdapter, BLEURTAdapter])
def test_learned_adapter_rejects_mismatched_input_lengths(adapter_class: type) -> None:
    adapter = adapter_class(
        learned_profile("bertscore_f1" if adapter_class is BERTScoreAdapter else "score"),
        backend=lambda *args, **kwargs: [],
    )
    with pytest.raises(ValueError, match="same length"):
        adapter.score_batch(["a"], [])


def test_learned_adapter_rejects_backend_output_length_mismatch() -> None:
    adapter = BLEURTAdapter(
        learned_profile("bleurt"), backend=lambda predictions, references, **kwargs: []
    )
    with pytest.raises(ValueError, match="returned 0 scores for 1 inputs"):
        adapter.score_batch(["a"], ["b"])


def test_missing_optional_dependency_has_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(_: str) -> object:
        raise ImportError("missing")

    monkeypatch.setattr("longcoqa.evaluation.paper_metrics.importlib.import_module", missing)
    with pytest.raises(RuntimeError, match="paper-metrics"):
        BERTScoreAdapter(learned_profile("bertscore_f1"))


def test_default_bertscore_backend_resolves_locked_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    class FakeBertScore:
        @staticmethod
        def score(predictions: list[str], references: list[str], **kwargs: object):
            calls["model_type"] = kwargs["model_type"]
            calls["num_layers"] = kwargs["num_layers"]
            assert "model_revision" not in kwargs
            return [1.0], [1.0], [1.0]

    class FakeHub:
        @staticmethod
        def snapshot_download(*, repo_id: str, revision: str) -> str:
            calls["repo_id"] = repo_id
            calls["revision"] = revision
            return "/locked/model"

    def fake_import(name: str) -> object:
        return {"bert_score": FakeBertScore, "huggingface_hub": FakeHub}[name]

    monkeypatch.setattr("longcoqa.evaluation.paper_metrics.importlib.import_module", fake_import)
    metric_profile = learned_profile("bertscore_f1", num_layers=9)
    output = BERTScoreAdapter(metric_profile).score_batch(["a"], ["a"])

    assert output.scores[0]["bertscore_f1"] == 1.0
    assert calls == {
        "repo_id": metric_profile.model_id,
        "revision": metric_profile.model_revision,
        "model_type": "/locked/model",
        "num_layers": 9,
    }


def test_score_batch_rows_checkpoints_and_reuses_learned_scores(tmp_path) -> None:
    calls = 0

    class FakeAdapter:
        def score_batch(self, predictions, references):
            nonlocal calls
            calls += 1
            return BatchMetricOutput(
                scores=tuple({"bleurt": 0.5} for _ in predictions),
                truncated=tuple(False for _ in predictions),
            )

    rows = [
        {
            "status": "ok",
            "sample_id": "factoid:1:1",
            "question_type": "factoid",
            "dialogue_id": 1,
            "turn": 1,
            "prediction": "answer",
            "reference": "reference",
        },
        {
            "status": "ok",
            "sample_id": "non_factoid:2:1",
            "question_type": "non_factoid",
            "dialogue_id": 2,
            "turn": 5,
            "prediction": "answer 2",
            "reference": "reference 2",
        },
    ]
    metric_profile = learned_profile("bleurt")

    first = score_batch_rows(rows, metric_profile, tmp_path, FakeAdapter())
    second = score_batch_rows(rows, metric_profile, tmp_path, FakeAdapter())

    assert calls == 1
    assert first == second
    assert all(item["status"] == "ok" for item in first)
    assert first[0]["scores"] == {"bleurt": 0.5}
