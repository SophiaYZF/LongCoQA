from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from longcoqa.evaluation.p0_cpu_priority import (
    analyze_within_question_hit_miss,
    cluster_bootstrap_mean,
    fit_ols,
    holm_adjust,
    score_generation_row,
    paired_cluster_bootstrap,
    run_g0_g3_rewrite_quality,
    run_r1,
    summarize_hit_miss,
    turn_bucket,
    validate_formal_rows,
    validate_equal_sample_ids,
)


def test_cluster_bootstrap_mean_is_deterministic_and_clustered() -> None:
    values = np.asarray([0.0, 0.0, 1.0, 1.0])
    clusters = np.asarray(["a", "a", "b", "b"])

    first = cluster_bootstrap_mean(values, clusters, iterations=200, seed=7)
    second = cluster_bootstrap_mean(values, clusters, iterations=200, seed=7)

    assert first == second
    assert first[0] == pytest.approx(0.5)
    assert first[1] <= first[0] <= first[2]


def test_paired_cluster_bootstrap_reports_right_minus_left() -> None:
    left = np.asarray([0.1, 0.2, 0.3, 0.4])
    right = left + 0.25
    clusters = np.asarray(["a", "a", "b", "b"])

    estimate, low, high, p_value = paired_cluster_bootstrap(
        left,
        right,
        clusters,
        iterations=200,
        seed=11,
    )

    assert estimate == pytest.approx(0.25)
    assert low == pytest.approx(0.25)
    assert high == pytest.approx(0.25)
    assert 0.0 <= p_value <= 1.0


def test_holm_adjust_is_monotone_in_sorted_p_values() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.03])

    assert adjusted == pytest.approx([0.03, 0.06, 0.06])
    ordered = [adjusted[index] for index in np.argsort([0.01, 0.04, 0.03])]
    assert ordered == sorted(ordered)


def test_validate_equal_sample_ids_rejects_duplicates_and_mismatch() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        validate_equal_sample_ids({"left": ["a", "a"], "right": ["a", "b"]})
    with pytest.raises(ValueError, match="sample ID mismatch"):
        validate_equal_sample_ids({"left": ["a", "b"], "right": ["a", "c"]})


def test_fit_ols_recovers_known_coefficients() -> None:
    design = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
        ]
    )
    outcome = np.asarray([2.0, 5.0, 8.0, 11.0])

    coefficients = fit_ols(design, outcome)

    assert coefficients == pytest.approx([2.0, 3.0])


def test_turn_bucket_contract() -> None:
    assert [turn_bucket(value) for value in (1, 2, 4, 5, 14)] == [
        "turn_1",
        "turn_2_4",
        "turn_2_4",
        "turn_5_plus",
        "turn_5_plus",
    ]
    with pytest.raises(ValueError, match="positive"):
        turn_bucket(0)


def test_score_generation_row_adds_locked_metrics_and_lengths() -> None:
    scored = score_generation_row(
        {
            "sample_id": "factoid:1:1",
            "question_type": "factoid",
            "dialogue_id": 1,
            "turn": 1,
            "status": "ok",
            "prediction": "答案：巫师女歌手。",
            "reference": "巫师女歌手。",
        }
    )

    assert scored["exact_match_contain"] == 1.0
    assert scored["prediction_length"] == 7
    assert scored["reference_length"] == 5
    assert scored["cluster"] == "factoid:1"


def test_validate_formal_rows_rejects_failure_duplicate_and_wrong_count() -> None:
    row = {"sample_id": "factoid:1:1", "status": "ok"}
    with pytest.raises(ValueError, match="failed"):
        validate_formal_rows([{**row, "status": "error"}], expected=1, label="cell")
    with pytest.raises(ValueError, match="duplicate"):
        validate_formal_rows([row, row], expected=2, label="cell")
    with pytest.raises(ValueError, match="expected 2"):
        validate_formal_rows([row], expected=2, label="cell")


def test_within_question_hit_miss_uses_only_discordant_samples() -> None:
    rows = [
        {"sample_id": "factoid:1:1", "method": "a", "hit": 1, "score": 0.8},
        {"sample_id": "factoid:1:1", "method": "b", "hit": 0, "score": 0.2},
        {"sample_id": "factoid:1:1", "method": "c", "hit": 1, "score": 0.6},
        {"sample_id": "factoid:2:1", "method": "a", "hit": 1, "score": 0.4},
        {"sample_id": "factoid:2:1", "method": "b", "hit": 1, "score": 0.5},
    ]

    paired = analyze_within_question_hit_miss(rows, metric="score")

    assert len(paired) == 1
    assert paired[0]["sample_id"] == "factoid:1:1"
    assert paired[0]["hit_mean"] == pytest.approx(0.7)
    assert paired[0]["miss_mean"] == pytest.approx(0.2)
    assert paired[0]["delta"] == pytest.approx(0.5)


def test_run_r1_uses_exact_pairing_and_holm_adjustment() -> None:
    sample_ids = ["factoid:1:1", "factoid:1:2", "summary:2:1", "summary:2:2"]
    base = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "cluster": ["factoid:1", "factoid:1", "summary:2", "summary:2"],
            "hit_at_20": [0.0, 0.0, 1.0, 1.0],
            "recall_at_20": [0.1, 0.2, 0.3, 0.4],
            "ndcg_at_20": [0.1, 0.2, 0.3, 0.4],
            "reciprocal_rank": [0.1, 0.2, 0.3, 0.4],
        }
    )
    better = base.copy()
    for metric in ("hit_at_20", "recall_at_20", "ndcg_at_20", "reciprocal_rank"):
        better[metric] = base[metric] + 0.1

    summary, pairs = run_r1({"base": base, "better": better}, iterations=200, seed=3)

    hit_summary = summary[summary["metric"] == "hit_at_20"].set_index("method")
    assert hit_summary.loc["better", "estimate"] - hit_summary.loc["base", "estimate"] == pytest.approx(0.1)
    hit_pair = pairs[pairs["metric"] == "hit_at_20"].iloc[0]
    assert hit_pair["delta_right_minus_left"] == pytest.approx(0.1)
    assert 0.0 <= hit_pair["p_holm"] <= 1.0


def test_summarize_hit_miss_reports_conditional_and_discordant_effects() -> None:
    frame = pd.DataFrame(
        [
            {"sample_id": "factoid:1:1", "cluster": "factoid:1", "method": "a", "hit": 1, "character_f1": 0.8},
            {"sample_id": "factoid:1:1", "cluster": "factoid:1", "method": "b", "hit": 0, "character_f1": 0.2},
            {"sample_id": "factoid:2:1", "cluster": "factoid:2", "method": "a", "hit": 1, "character_f1": 0.6},
            {"sample_id": "factoid:2:1", "cluster": "factoid:2", "method": "b", "hit": 1, "character_f1": 0.4},
        ]
    )

    conditional, discordant = summarize_hit_miss(
        frame,
        metric="character_f1",
        iterations=200,
        seed=5,
    )

    assert set(conditional["condition"]) == {"hit", "miss"}
    assert discordant.iloc[0]["discordant_samples"] == 1
    assert discordant.iloc[0]["delta_hit_minus_miss"] == pytest.approx(0.6)


def test_g0_g3_rewrite_quality_uses_g3_minus_g0_and_factoid_only_em() -> None:
    rows = []
    values = {
        ("a", "factoid:1:1"): (0.2, 0.4, 0.0, 1.0),
        ("b", "factoid:1:1"): (0.4, 0.5, 1.0, 1.0),
        ("a", "non_factoid:2:5"): (0.3, 0.2, 0.0, 0.0),
        ("b", "non_factoid:2:5"): (0.2, 0.4, 0.0, 0.0),
    }
    for (model, sample_id), (g0_f1, g3_f1, g0_em, g3_em) in values.items():
        question_type, dialogue_id, turn = sample_id.split(":")
        for protocol, character_f1, exact_match_contain in (
            ("g0_released_gold", g0_f1, g0_em),
            ("g3_aligned_raw", g3_f1, g3_em),
        ):
            rows.append(
                {
                    "model": model,
                    "protocol": protocol,
                    "sample_id": sample_id,
                    "question_type": question_type,
                    "dialogue_id": int(dialogue_id),
                    "turn": int(turn),
                    "turn_bucket": turn_bucket(int(turn)),
                    "cluster": f"{question_type}:{dialogue_id}",
                    "character_f1": character_f1,
                    "exact_match_contain": exact_match_contain,
                }
            )

    aggregate, model_level = run_g0_g3_rewrite_quality(
        pd.DataFrame(rows),
        iterations=200,
        seed=9,
    )

    overall = aggregate[
        (aggregate["metric"] == "character_f1") & (aggregate["scope"] == "overall")
    ].iloc[0]
    factoid = aggregate[
        (aggregate["metric"] == "character_f1")
        & (aggregate["scope"] == "question_type")
        & (aggregate["question_type"] == "factoid")
    ].iloc[0]
    factoid_em = aggregate[
        (aggregate["metric"] == "exact_match_contain")
        & (aggregate["scope"] == "question_type")
    ].iloc[0]

    assert overall["delta_g3_minus_g0"] == pytest.approx(0.1)
    assert factoid["delta_g3_minus_g0"] == pytest.approx(0.15)
    assert factoid_em["delta_g3_minus_g0"] == pytest.approx(0.5)
    assert set(aggregate.loc[aggregate["metric"] == "exact_match_contain", "question_type"]) == {
        "factoid"
    }
    assert len(model_level) == 8


def test_g0_g3_rewrite_quality_rejects_incomplete_pairs() -> None:
    frame = pd.DataFrame(
        [
            {
                "model": "a",
                "protocol": "g0_released_gold",
                "sample_id": "factoid:1:1",
                "question_type": "factoid",
                "dialogue_id": 1,
                "turn": 1,
                "turn_bucket": "turn_1",
                "cluster": "factoid:1",
                "character_f1": 0.2,
                "exact_match_contain": 0.0,
            }
        ]
    )

    with pytest.raises(ValueError, match="G0/G3 sample ID mismatch"):
        run_g0_g3_rewrite_quality(frame, iterations=20, seed=1)
