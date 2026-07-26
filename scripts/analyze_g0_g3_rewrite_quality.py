#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from longcoqa.evaluation.p0_cpu_priority import (  # noqa: E402
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_SEED,
    load_p0_generation_rows,
    run_g0_g3_rewrite_quality,
)
from longcoqa.paths import PROJECT_ROOT  # noqa: E402
from longcoqa.utils.io import atomic_write_json  # noqa: E402

DEFAULT_OUTPUT = PROJECT_ROOT / "reports/generated/analysis/g0_g3_rewrite_quality_20260724"


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def selected_row(frame: pd.DataFrame, **filters: str) -> pd.Series:
    selected = frame
    for field, value in filters.items():
        selected = selected[selected[field] == value]
    if len(selected) != 1:
        raise ValueError(f"expected one row for {filters}, found {len(selected)}")
    return selected.iloc[0]


def effect_text(row: pd.Series) -> str:
    return (
        f"{row['delta_g3_minus_g0']:+.4f} "
        f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]"
    )


def build_artifact(
    *,
    aggregate: pd.DataFrame,
    model_level: pd.DataFrame,
    output_dir: Path,
    generated_at: str,
) -> dict[str, Any]:
    relative_dir = output_dir.resolve().relative_to(PROJECT_ROOT)
    overall = selected_row(
        aggregate,
        metric="character_f1",
        scope="overall",
        question_type="all",
        turn_bucket="all",
    )
    factoid = selected_row(
        aggregate,
        metric="character_f1",
        scope="question_type",
        question_type="factoid",
        turn_bucket="all",
    )
    non_factoid = selected_row(
        aggregate,
        metric="character_f1",
        scope="question_type",
        question_type="non_factoid",
        turn_bucket="all",
    )
    factoid_em = selected_row(
        aggregate,
        metric="exact_match_contain",
        scope="question_type",
        question_type="factoid",
        turn_bucket="all",
    )
    factoid_turn_one = selected_row(
        aggregate,
        metric="character_f1",
        scope="question_type_turn_bucket",
        question_type="factoid",
        turn_bucket="turn_1",
    )
    factoid_turn_five = selected_row(
        aggregate,
        metric="character_f1",
        scope="question_type_turn_bucket",
        question_type="factoid",
        turn_bucket="turn_5_plus",
    )

    chart_data = aggregate[
        (aggregate["metric"] == "character_f1")
        & (aggregate["scope"] == "question_type_turn_bucket")
    ].copy()
    model_factoid = model_level[
        (model_level["metric"] == "character_f1")
        & (model_level["question_type"] == "factoid")
    ]
    model_non_factoid = model_level[
        (model_level["metric"] == "character_f1")
        & (model_level["question_type"] == "non_factoid")
    ]
    factoid_positive = int((model_factoid["delta_g3_minus_g0"] > 0).sum())
    non_factoid_positive = int((model_non_factoid["delta_g3_minus_g0"] > 0).sum())
    factoid_model_count = len(model_factoid)
    non_factoid_model_count = len(model_non_factoid)

    def source_path(filename: str) -> str:
        return str(relative_dir / filename)

    def csv_source(identifier: str, label: str, filename: str) -> dict[str, Any]:
        path = source_path(filename)
        return {
            "id": identifier,
            "label": label,
            "path": path,
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "description": f"Load the validated aggregate rows from {filename}.",
                "executed_at": generated_at,
                "sql": f"SELECT * FROM read_csv_auto('{path}')",
            },
        }

    summary_path = source_path("analysis_summary.json")
    sources = [
        csv_source("aggregate_source", "G3 minus G0 aggregate effects", "aggregate_effects.csv"),
        csv_source("model_source", "G3 minus G0 model effects", "model_effects.csv"),
        {
            "id": "summary_source",
            "label": "Analysis metadata and source manifest",
            "path": summary_path,
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "description": "Load the analysis quality gate and provenance metadata.",
                "executed_at": generated_at,
                "sql": f"SELECT * FROM read_json_auto('{summary_path}')",
            },
        },
    ]
    chart = {
        "id": "turn_type_effect_chart",
        "title": "G3−G0 Character-F1 by question type and turn bucket",
        "subtitle": "Seven-model paired effects; positive values indicate higher performance with aligned raw evidence.",
        "intent": "comparison",
        "question": "Where does released rewrite lose answer-relevant performance relative to aligned raw evidence?",
        "rationale": "Grouped bars compare the pre-registered turn buckets within each question type on one common delta scale.",
        "type": "bar",
        "dataset": "turn_type_effects",
        "sourceId": "aggregate_source",
        "encodings": {
            "x": {"field": "turn_bucket", "type": "nominal", "label": "Turn bucket"},
            "y": {
                "field": "delta_g3_minus_g0",
                "type": "quantitative",
                "label": "Character-F1 delta",
            },
            "color": {
                "field": "question_type",
                "type": "nominal",
                "label": "Question type",
            },
            "tooltip": [
                {"field": "samples", "type": "quantitative", "label": "Question N"},
                {"field": "ci_low", "type": "quantitative", "label": "95% CI low"},
                {"field": "ci_high", "type": "quantitative", "label": "95% CI high"},
            ],
        },
    }
    aggregate_table = {
        "id": "aggregate_table",
        "title": "Aggregate G3−G0 effects",
        "subtitle": "Exact paired estimates across overall, question-type, and turn-depth cuts.",
        "dataset": "aggregate_effects",
        "sourceId": "aggregate_source",
        "density": "spacious",
        "columns": [
            {"field": "metric", "label": "Metric", "type": "text"},
            {"field": "scope", "label": "Scope", "type": "text"},
            {"field": "question_type", "label": "Question type", "type": "text"},
            {"field": "turn_bucket", "label": "Turn bucket", "type": "text"},
            {"field": "samples", "label": "Question N", "type": "number"},
            {"field": "delta_g3_minus_g0", "label": "G3−G0", "format": "number", "movement": True},
            {"field": "ci_low", "label": "CI low", "format": "number"},
            {"field": "ci_high", "label": "CI high", "format": "number"},
        ],
    }
    model_table = {
        "id": "model_table",
        "title": "Model-level G3−G0 effects",
        "subtitle": "Seven models, with Character-F1 by question type and Factoid EM-contain.",
        "dataset": "model_effects",
        "sourceId": "model_source",
        "density": "dense",
        "columns": [
            {"field": "model", "label": "Model", "type": "text"},
            {"field": "metric", "label": "Metric", "type": "text"},
            {"field": "question_type", "label": "Question type", "type": "text"},
            {"field": "samples", "label": "Question N", "type": "number"},
            {"field": "g0_mean", "label": "G0 mean", "format": "number"},
            {"field": "g3_mean", "label": "G3 mean", "format": "number"},
            {"field": "delta_g3_minus_g0", "label": "G3−G0", "format": "number", "movement": True},
            {"field": "ci_low", "label": "CI low", "format": "number"},
            {"field": "ci_high", "label": "CI high", "format": "number"},
        ],
    }
    blocks = [
        {
            "id": "title",
            "type": "markdown",
            "body": "# LongCoQA G0/G3 Rewrite Quality Analysis\n\n2026-07-24 formal data snapshot",
        },
        {
            "id": "technical_summary",
            "type": "markdown",
            "sourceId": "aggregate_source",
            "body": (
                "## Technical summary\n\n"
                f"**Released rewrite is close to aligned raw evidence on Non-factoid questions, but not lossless overall.** "
                f"Across 3,019 questions and seven models, G3−G0 Character-F1 is {effect_text(overall)}. "
                f"The effect is concentrated in Factoid ({effect_text(factoid)}; EM-contain {effect_text(factoid_em)}), "
                f"while Non-factoid is much smaller at {effect_text(non_factoid)}.\n\n"
                "**The safe paper claim is asymmetric.** Released rewrite preserves open-ended answer performance closely, "
                "but aligned raw evidence retains measurable short-answer detail. This comparison does not directly score textual semantic equivalence."
            ),
        },
        {
            "id": "turn_finding",
            "type": "markdown",
            "sourceId": "aggregate_source",
            "body": (
                "## Factoid rewrite loss becomes clear after Turn 1\n\n"
                f"Factoid Turn 1 is {effect_text(factoid_turn_one)}, with a confidence interval crossing zero. "
                f"By Turn 5+, the effect grows to {effect_text(factoid_turn_five)}. "
                "The chart uses one shared zero baseline; bars above zero favor aligned raw evidence."
            ),
        },
        {"id": "turn_chart_block", "type": "chart", "chartId": "turn_type_effect_chart"},
        {
            "id": "aggregate_interpretation",
            "type": "markdown",
            "body": "## Exact aggregate cuts\n\nThe lookup table keeps overall, question-type, turn-bucket, and interaction cuts together. Character-F1 covers all questions; EM-contain rows are restricted to Factoid, so no artificial binary correctness label is introduced for Non-factoid answers.",
        },
        {"id": "aggregate_table_block", "type": "table", "tableId": "aggregate_table"},
        {
            "id": "model_finding",
            "type": "markdown",
            "sourceId": "model_source",
            "body": (
                "## The Factoid direction is consistent across models\n\n"
                f"G3−G0 Factoid Character-F1 is positive for {factoid_positive}/{factoid_model_count} available models. "
                f"Non-factoid is positive for {non_factoid_positive}/{non_factoid_model_count} available models and is materially smaller. "
                "Model-level intervals expose exceptions rather than allowing the aggregate result to hide them."
            ),
        },
        {"id": "model_table_block", "type": "table", "tableId": "model_table"},
        {
            "id": "methods",
            "type": "markdown",
            "body": "## Definitions and method\n\nG0 supplies released rewritten gold evidence; G3 supplies aligned raw gold evidence. Every row is paired by model and sample ID, and every effect is G3 minus G0. The analysis contains seven models × 3,019 questions = 21,133 paired model observations. Confidence intervals use 2,000 percentile bootstrap draws over `(question_type, dialogue_id)` clusters with seed 42.",
        },
        {
            "id": "limitations",
            "type": "markdown",
            "body": "## Limitations and next step\n\n- Prediction metrics measure downstream answer behavior, not semantic equivalence between the two evidence texts.\n- G0 and G3 differ in wording and evidence alignment; the analysis does not randomize evidence representation.\n- Character-F1 and EM-contain can disagree with human semantic correctness.\n- The next CPU step is to inspect high-loss Factoid pairs by entity/number omission and alignment confidence using a fixed, blinded manifest.",
        },
    ]
    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "LongCoQA G0/G3 Rewrite Quality Analysis",
            "description": "Paired seven-model analysis of released rewritten versus aligned raw gold evidence.",
            "generatedAt": generated_at,
            "filters": [],
            "cards": [],
            "charts": [chart],
            "tables": [aggregate_table, model_table],
            "sources": sources,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "turn_type_effects": records(chart_data),
                "aggregate_effects": records(aggregate),
                "model_effects": records(model_level),
            },
            "accessIssues": [],
        },
        "sources": sources,
    }


def run(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    generation, source_manifest = load_p0_generation_rows()
    aggregate, model_level = run_g0_g3_rewrite_quality(generation)
    overall = selected_row(
        aggregate,
        metric="character_f1",
        scope="overall",
        question_type="all",
        turn_bucket="all",
    )
    if int(overall["samples"]) != 3019 or int(overall["model_observations"]) != 7 * 3019:
        raise ValueError("real G0/G3 matrix failed the denominator gate")
    aggregate.to_csv(output_dir / "aggregate_effects.csv", index=False, float_format="%.12g")
    model_level.to_csv(output_dir / "model_effects.csv", index=False, float_format="%.12g")
    g0_g3_manifest = [
        row
        for row in source_manifest
        if row["cell"].endswith("/g0_released_gold") or row["cell"].endswith("/g3_aligned_raw")
    ]
    summary = {
        "schema_version": "longcoqa-g0-g3-rewrite-quality-v1",
        "generated_at": generated_at,
        "effect_direction": "g3_aligned_raw_minus_g0_released_gold",
        "bootstrap": {
            "unit": "question_type:dialogue_id",
            "iterations": BOOTSTRAP_ITERATIONS,
            "seed": BOOTSTRAP_SEED,
            "interval": "percentile_95",
        },
        "data_quality": {
            "models": 7,
            "protocols": ["g0_released_gold", "g3_aligned_raw"],
            "unique_samples": 3019,
            "factoid_samples": 1172,
            "non_factoid_samples": 1847,
            "paired_model_observations": 21133,
            "status": "passed",
        },
        "outputs": {
            "aggregate_effects": str((output_dir / "aggregate_effects.csv").relative_to(PROJECT_ROOT)),
            "model_effects": str((output_dir / "model_effects.csv").relative_to(PROJECT_ROOT)),
        },
        "source_manifest": g0_g3_manifest,
        "limitations": [
            "Downstream answer metrics do not directly measure textual semantic preservation.",
            "The comparison is observational and does not identify a causal evidence-representation effect.",
            "No human or calibrated cross-family Judge outcome is included.",
        ],
    }
    atomic_write_json(output_dir / "analysis_summary.json", summary)
    artifact = build_artifact(
        aggregate=aggregate,
        model_level=model_level,
        output_dir=output_dir,
        generated_at=generated_at,
    )
    atomic_write_json(output_dir / "artifact.json", artifact)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze paired G0/G3 rewrite quality effects.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = run(args.output_dir.resolve())
    print(json.dumps(summary["data_quality"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
