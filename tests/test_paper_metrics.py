from __future__ import annotations

import dataclasses

import pytest

from longcoqa.evaluation.paper_metric_profiles import (
    MetricCompatibility,
    MetricProfile,
)
from longcoqa.evaluation.paper_metrics import paper_lexical_scores


def lexical_profile(**overrides: object) -> MetricProfile:
    values: dict[str, object] = {
        "profile_id": "longcoqa-paper-lexical-v1",
        "metrics": (
            "exact_match",
            "exact_match_contain",
            "recall",
            "bleu_2",
            "rouge_l",
        ),
        "compatibility": MetricCompatibility.PAPER_NAMED_LOCKED,
        "implementation": "longcoqa.evaluation.paper_metrics",
        "normalization": "zh_alnum_lower",
        "tokenization": "character",
        "options": {"bleu_smoothing": "add_one"},
    }
    values.update(overrides)
    return MetricProfile(**values)


def test_metric_profile_is_immutable_and_hash_is_stable() -> None:
    left = lexical_profile(options={"bleu_smoothing": "add_one", "casefold": True})
    right = lexical_profile(options={"casefold": True, "bleu_smoothing": "add_one"})

    assert left.profile_hash == right.profile_hash
    assert left.to_dict()["compatibility"] == "paper_named_locked"
    with pytest.raises(dataclasses.FrozenInstanceError):
        left.profile_id = "changed"  # type: ignore[misc]


def test_learned_metric_profile_requires_locked_revision() -> None:
    with pytest.raises(ValueError, match="revision"):
        lexical_profile(
            profile_id="bertscore-v1",
            metrics=("bertscore_f1",),
            model_id="example/bert",
            model_revision=None,
        )


@pytest.mark.parametrize("field", ["profile_id", "implementation", "normalization", "tokenization"])
def test_metric_profile_rejects_blank_required_fields(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        lexical_profile(**{field: " "})


def test_paper_lexical_scores_use_reference_containment_direction() -> None:
    result = paper_lexical_scores("答案是小天狼星。", "小天狼星", lexical_profile())

    assert result.profile_id == "longcoqa-paper-lexical-v1"
    assert result.compatibility == MetricCompatibility.PAPER_NAMED_LOCKED
    assert result.scores["exact_match"] == 0.0
    assert result.scores["exact_match_contain"] == 1.0
    assert result.scores["recall"] == 1.0
    assert 0.0 < result.scores["bleu_2"] < 1.0
    assert 0.0 < result.scores["rouge_l"] < 1.0


def test_paper_lexical_scores_normalize_punctuation_and_case() -> None:
    result = paper_lexical_scores(" Sirius！", "sirius", lexical_profile())

    assert result.scores == {
        "exact_match": 1.0,
        "exact_match_contain": 1.0,
        "recall": 1.0,
        "bleu_2": 1.0,
        "rouge_l": 1.0,
    }


@pytest.mark.parametrize(
    ("prediction", "reference", "expected"),
    [("", "", 1.0), ("", "答案", 0.0), ("答案", "", 0.0)],
)
def test_paper_lexical_scores_define_empty_inputs(
    prediction: str, reference: str, expected: float
) -> None:
    scores = paper_lexical_scores(prediction, reference, lexical_profile()).scores

    assert scores["exact_match"] == expected
    assert scores["recall"] == expected
    assert scores["bleu_2"] == expected
    assert scores["rouge_l"] == expected


def test_paper_lexical_scores_only_emit_requested_metrics() -> None:
    profile = lexical_profile(metrics=("exact_match", "recall"))

    assert set(paper_lexical_scores("a", "a", profile).scores) == {
        "exact_match",
        "recall",
    }


def test_paper_lexical_scores_reject_unknown_metric() -> None:
    with pytest.raises(ValueError, match="unknown_metric"):
        paper_lexical_scores("a", "a", lexical_profile(metrics=("unknown_metric",)))
