from __future__ import annotations

import pandas as pd

from scripts import analyze_p1_evidence_overlap as analysis


def test_normalize_result_metrics_accepts_missing_raft_refusal_metrics() -> None:
    normalize = getattr(analysis, "normalize_result_metrics", None)

    assert normalize is not None, "analysis must normalize legal null metrics"
    assert normalize(None) == {
        "character_f1": None,
        "rouge_l_f1": None,
        "exact_match": None,
    }


def test_paired_lift_table_reports_exact_and_hierarchical_groups() -> None:
    rows = pd.DataFrame(
        [
            {
                "sample_id": sample_id,
                "model_variant": model_variant,
                "eval_variant": "grounded",
                "answerable": True,
                "evidence_count": 1,
                "dialogue_key": dialogue_key,
                "question_type": "factoid",
                "exact_binary": exact_group,
                "hierarchy_status": hierarchy_group,
                "character_f1": score,
                "judge_100": score * 100,
            }
            for sample_id, dialogue_key, exact_group, hierarchy_group in (
                ("factoid:1:1", "factoid:1", "exact_seen", "exact_seen"),
                (
                    "factoid:2:1",
                    "factoid:2",
                    "exact_unseen",
                    "strict_unseen_chapter",
                ),
            )
            for model_variant, score in (("base", 0.2), ("grounded", 0.4))
        ]
    )

    result = analysis.paired_lift_table(rows)

    assert "group_scheme" in result.columns
    assert set(result["group_scheme"]) == {"exact_binary", "hierarchy"}
    assert "question_type" in result.columns
    assert set(result["question_type"]) == {"overall", "factoid"}
