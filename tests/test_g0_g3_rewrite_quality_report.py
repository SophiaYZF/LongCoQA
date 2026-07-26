from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.analyze_g0_g3_rewrite_quality import build_artifact


def test_build_artifact_has_native_chart_tables_and_relative_sources() -> None:
    aggregate = pd.DataFrame(
        [
            {"metric": "character_f1", "scope": "overall", "question_type": "all", "turn_bucket": "all", "samples": 4, "model_observations": 8, "dialogue_clusters": 2, "g0_mean": 0.30, "g3_mean": 0.32, "delta_g3_minus_g0": 0.02, "ci_low": 0.01, "ci_high": 0.03, "p_bootstrap": 0.01},
            {"metric": "character_f1", "scope": "question_type", "question_type": "factoid", "turn_bucket": "all", "samples": 2, "model_observations": 4, "dialogue_clusters": 1, "g0_mean": 0.20, "g3_mean": 0.24, "delta_g3_minus_g0": 0.04, "ci_low": 0.02, "ci_high": 0.06, "p_bootstrap": 0.01},
            {"metric": "character_f1", "scope": "question_type", "question_type": "non_factoid", "turn_bucket": "all", "samples": 2, "model_observations": 4, "dialogue_clusters": 1, "g0_mean": 0.40, "g3_mean": 0.40, "delta_g3_minus_g0": 0.00, "ci_low": -0.01, "ci_high": 0.01, "p_bootstrap": 1.00},
            {"metric": "exact_match_contain", "scope": "question_type", "question_type": "factoid", "turn_bucket": "all", "samples": 2, "model_observations": 4, "dialogue_clusters": 1, "g0_mean": 0.40, "g3_mean": 0.50, "delta_g3_minus_g0": 0.10, "ci_low": 0.05, "ci_high": 0.15, "p_bootstrap": 0.01},
            {"metric": "character_f1", "scope": "question_type_turn_bucket", "question_type": "factoid", "turn_bucket": "turn_1", "samples": 1, "model_observations": 2, "dialogue_clusters": 1, "g0_mean": 0.20, "g3_mean": 0.21, "delta_g3_minus_g0": 0.01, "ci_low": -0.01, "ci_high": 0.03, "p_bootstrap": 0.20},
            {"metric": "character_f1", "scope": "question_type_turn_bucket", "question_type": "factoid", "turn_bucket": "turn_5_plus", "samples": 1, "model_observations": 2, "dialogue_clusters": 1, "g0_mean": 0.20, "g3_mean": 0.25, "delta_g3_minus_g0": 0.05, "ci_low": 0.02, "ci_high": 0.08, "p_bootstrap": 0.01},
            {"metric": "character_f1", "scope": "question_type_turn_bucket", "question_type": "non_factoid", "turn_bucket": "turn_5_plus", "samples": 1, "model_observations": 2, "dialogue_clusters": 1, "g0_mean": 0.40, "g3_mean": 0.40, "delta_g3_minus_g0": 0.00, "ci_low": -0.01, "ci_high": 0.01, "p_bootstrap": 1.00},
        ]
    )
    model_level = pd.DataFrame(
        [
            {"model": "a", "metric": "character_f1", "question_type": "factoid", "samples": 2, "model_observations": 2, "dialogue_clusters": 1, "g0_mean": 0.20, "g3_mean": 0.25, "delta_g3_minus_g0": 0.05, "ci_low": 0.02, "ci_high": 0.08, "p_bootstrap": 0.01},
            {"model": "b", "metric": "character_f1", "question_type": "factoid", "samples": 2, "model_observations": 2, "dialogue_clusters": 1, "g0_mean": 0.20, "g3_mean": 0.23, "delta_g3_minus_g0": 0.03, "ci_low": 0.01, "ci_high": 0.05, "p_bootstrap": 0.01},
        ]
    )

    artifact = build_artifact(
        aggregate=aggregate,
        model_level=model_level,
        output_dir=Path("reports/generated/analysis/g0_g3_rewrite_quality_20260724").resolve(),
        generated_at="2026-07-24T12:00:00Z",
    )

    assert artifact["surface"] == "report"
    assert artifact["manifest"]["blocks"][0]["body"].startswith("# ")
    assert len(artifact["manifest"]["charts"]) == 1
    assert len(artifact["manifest"]["tables"]) == 2
    assert all(not Path(source["path"]).is_absolute() for source in artifact["manifest"]["sources"])
