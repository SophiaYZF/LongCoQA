from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from longcoqa.evaluation.reporting import (
    compare_inference_results,
    evaluate_inference_results,
)
from longcoqa.utils.io import atomic_write_json, atomic_write_jsonl, read_jsonl

EXPECTED_SAMPLES = {"conversation": 602, "grounded": 539, "raft": 539}


def normalize_p1_reporting_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Use citation-free answers and exclude refusal-only rows from answer scores."""
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if row.get("status") != "ok" or not row.get("answerable", True):
            continue
        item = dict(row)
        item["prediction"] = str(row.get("answer_text") or row["prediction"])
        normalized.append(item)
    return normalized


def p1_protocol_metadata(summary: dict[str, Any]) -> dict[str, Any]:
    successful = int(summary["successful_samples"])
    answerable = int(summary["answerable_samples"])
    return {
        "successful_samples": successful,
        "answerable_samples": answerable,
        "missing_gold_samples": successful - answerable,
        "citation": dict(summary["citation"]),
        "abstention": dict(summary["abstention"]),
    }


def _load_validated_cell(
    root: Path,
    *,
    model_variant: str,
    eval_variant: str,
    expected_samples: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cell = root / model_variant / eval_variant / "full"
    rows = read_jsonl(cell / "results.jsonl")
    summary = json.loads((cell / "summary.json").read_text(encoding="utf-8"))
    if len(rows) != expected_samples:
        raise ValueError(
            f"{model_variant}/{eval_variant}: expected {expected_samples}, found {len(rows)}"
        )
    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"{model_variant}/{eval_variant}: duplicate sample IDs")
    if any(row.get("status") != "ok" for row in rows):
        raise ValueError(f"{model_variant}/{eval_variant}: failed rows")
    if {str(row.get("model_variant")) for row in rows} != {model_variant}:
        raise ValueError(f"{model_variant}/{eval_variant}: mismatched model variant")
    return rows, summary


def evaluate_p1_cpt(
    formal_root: Path,
    base_root: Path,
    output_root: Path,
    *,
    bootstrap_iterations: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Create CPT P1 automatic reports and paired Base-vs-CPT comparisons."""
    if not (formal_root / ".complete").is_file():
        raise FileNotFoundError(f"Missing CPT formal completion marker: {formal_root}")
    cpt_root = formal_root / "merged"
    reports: dict[str, Any] = {"automatic": {}, "comparisons": {}}

    with tempfile.TemporaryDirectory(prefix="longcoqa-p1-cpt-") as temporary:
        scratch = Path(temporary)
        for eval_variant, expected_samples in EXPECTED_SAMPLES.items():
            cpt_rows, cpt_summary = _load_validated_cell(
                cpt_root,
                model_variant="cpt",
                eval_variant=eval_variant,
                expected_samples=expected_samples,
            )
            base_rows, _ = _load_validated_cell(
                base_root,
                model_variant="base",
                eval_variant=eval_variant,
                expected_samples=expected_samples,
            )
            normalized_cpt = normalize_p1_reporting_rows(cpt_rows)
            normalized_base = normalize_p1_reporting_rows(base_rows)
            cpt_path = scratch / f"cpt_{eval_variant}.jsonl"
            base_path = scratch / f"base_{eval_variant}.jsonl"
            atomic_write_jsonl(cpt_path, normalized_cpt)
            atomic_write_jsonl(base_path, normalized_base)

            automatic_path = output_root / "automatic" / f"cpt_{eval_variant}.json"
            automatic = evaluate_inference_results(
                cpt_path,
                automatic_path,
                bootstrap_iterations=bootstrap_iterations,
                seed=seed,
            )
            automatic["source_input"] = str(
                cpt_root / "cpt" / eval_variant / "full" / "results.jsonl"
            )
            automatic["p1_protocol"] = p1_protocol_metadata(cpt_summary)
            atomic_write_json(automatic_path, automatic)
            reports["automatic"][eval_variant] = {
                "output": str(automatic_path),
                "answerable_samples": len(normalized_cpt),
            }

            comparison_path = (
                output_root
                / "comparisons"
                / f"{eval_variant}_base_vs_cpt.json"
            )
            comparison = compare_inference_results(
                base_path,
                cpt_path,
                comparison_path,
                bootstrap_iterations=bootstrap_iterations,
                seed=seed,
            )
            reports["comparisons"][eval_variant] = {
                "output": str(comparison_path),
                "samples": int(comparison["samples"]),
            }

    suite = {
        "schema_version": "longcoqa-p1-cpt-report-suite-v1",
        "formal_root": str(formal_root),
        "base_root": str(base_root),
        "bootstrap": {"iterations": bootstrap_iterations, "seed": seed},
        **reports,
    }
    atomic_write_json(output_root / "report_suite.json", suite)
    return suite
