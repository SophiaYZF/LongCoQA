from __future__ import annotations

import pytest

from longcoqa.evaluation.rag_quality import (
    ClaimAssessment,
    adapt_ragchecker_result,
    aggregate_rag_quality,
    citation_scores,
    claim_quality_record,
    parse_citations,
)


def test_parse_citations_supports_evidence_and_retrieved_ids() -> None:
    text = (
        "答案。\n引用：[EVIDENCE 12, EVIDENCE 18, EVIDENCE 12, NEG-2] "
        "[RETRIEVED 3] malformed"
    )

    assert parse_citations(text) == ("12", "18", "NEG-2", "3")
    assert parse_citations("没有引用") == ()


def test_citation_scores_count_unknown_and_deduplicate() -> None:
    score = citation_scores(["1", "1", "9"], ["1", "2"], answerable=True)

    assert score.status == "ok"
    assert score.precision == 0.5
    assert score.recall == 0.5
    assert score.f1 == 0.5
    assert score.correct_ids == ("1",)
    assert score.unknown_ids == ("9",)


def test_citation_scores_distinguish_no_citation_from_not_applicable() -> None:
    answerable = citation_scores([], ["1"], answerable=True)
    missing_gold = citation_scores([], [], answerable=False)

    assert answerable.status == "ok"
    assert answerable.precision is None
    assert answerable.recall == 0.0
    assert missing_gold.status == "inapplicable"
    assert missing_gold.reason == "missing_gold"


def test_claim_quality_record_computes_recall_and_full_support() -> None:
    record = claim_quality_record(
        sample_id="factoid:1:1",
        question_type="factoid",
        answerable=True,
        reference_claims=("claim a", "claim b"),
        candidate_claims=(
            ClaimAssessment("claim a", material=True, supported=True, matched_reference=0),
            ClaimAssessment("extra", material=True, supported=False, matched_reference=None),
        ),
        evidence_available=True,
    )

    assert record["status"] == "ok"
    assert record["claim_recall"] == 0.5
    assert record["claim_support_rate"] == 0.5
    assert record["answer_supported"] is False


def test_claim_quality_record_separates_missing_evidence_and_extraction_failure() -> None:
    no_evidence = claim_quality_record(
        sample_id="factoid:1:1",
        question_type="factoid",
        answerable=True,
        reference_claims=("a",),
        candidate_claims=(ClaimAssessment("a", material=True, supported=None),),
        evidence_available=False,
    )
    failed = claim_quality_record(
        sample_id="factoid:1:2",
        question_type="factoid",
        answerable=True,
        reference_claims=(),
        candidate_claims=(),
        evidence_available=True,
        extraction_error="invalid JSON",
    )

    assert no_evidence["claim_recall"] == 1.0
    assert no_evidence["claim_support_rate"] is None
    assert no_evidence["answer_supported"] is None
    assert no_evidence["support_status"] == "inapplicable"
    assert failed["status"] == "error"
    assert failed["error"] == "invalid JSON"


def test_empty_material_claims_are_not_automatically_supported() -> None:
    record = claim_quality_record(
        sample_id="factoid:1:1",
        question_type="factoid",
        answerable=True,
        reference_claims=("a",),
        candidate_claims=(),
        evidence_available=True,
    )

    assert record["claim_recall"] == 0.0
    assert record["claim_support_rate"] is None
    assert record["answer_supported"] is False


def test_ragchecker_adapter_preserves_official_fields_and_rejects_missing() -> None:
    raw = {
        "claim_recall": 0.8,
        "context_precision": 0.7,
        "context_utilization": 0.6,
        "noise_sensitivity": 0.1,
        "faithfulness": 0.9,
        "extra": "preserved",
    }

    adapted = adapt_ragchecker_result(raw)

    assert adapted["schema_version"] == "longcoqa-ragchecker-adapter-v1"
    assert adapted["official"] == raw
    with pytest.raises(ValueError, match="faithfulness"):
        adapt_ragchecker_result({key: value for key, value in raw.items() if key != "faithfulness"})


def test_aggregate_rag_quality_separates_answerable_and_missing_gold() -> None:
    records = [
        {
            "status": "ok",
            "answerable": True,
            "citation": {"status": "ok", "precision": 1.0, "recall": 0.5, "f1": 2 / 3},
            "claim_recall": 0.5,
            "claim_support_rate": 1.0,
            "answer_supported": True,
        },
        {
            "status": "ok",
            "answerable": False,
            "citation": {"status": "inapplicable"},
            "claim_recall": None,
            "claim_support_rate": None,
            "answer_supported": None,
        },
    ]

    report = aggregate_rag_quality(records)

    assert report["requested_samples"] == 2
    assert report["answerable"]["samples"] == 1
    assert report["answerable"]["citation_precision"] == 1.0
    assert report["answerable"]["answer_supported_rate"] == 1.0
    assert report["missing_gold"]["samples"] == 1
