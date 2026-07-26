from longcoqa.evaluation.p1_cpt_reporting import (
    normalize_p1_reporting_rows,
    p1_protocol_metadata,
)


def test_normalize_p1_reporting_rows_uses_answer_text_and_filters_missing_gold() -> None:
    rows = [
        {
            "status": "ok",
            "sample_id": "factoid:1:1",
            "answerable": True,
            "prediction": "答案。\n\n引用：[EVIDENCE 1]",
            "answer_text": "答案。",
        },
        {
            "status": "ok",
            "sample_id": "factoid:1:2",
            "answerable": False,
            "prediction": "提供的证据不足以回答该问题。",
            "answer_text": "提供的证据不足以回答该问题。",
        },
    ]

    normalized = normalize_p1_reporting_rows(rows)

    assert len(normalized) == 1
    assert normalized[0]["sample_id"] == "factoid:1:1"
    assert normalized[0]["prediction"] == "答案。"


def test_p1_protocol_metadata_preserves_citation_and_abstention() -> None:
    summary = {
        "citation": {"precision": 1.0, "recall": 0.75},
        "abstention": {"precision": 0.4, "recall": 0.9, "f1": 0.55},
        "answerable_samples": 484,
        "successful_samples": 539,
    }

    metadata = p1_protocol_metadata(summary)

    assert metadata["citation"] == summary["citation"]
    assert metadata["abstention"] == summary["abstention"]
    assert metadata["answerable_samples"] == 484
    assert metadata["missing_gold_samples"] == 55
