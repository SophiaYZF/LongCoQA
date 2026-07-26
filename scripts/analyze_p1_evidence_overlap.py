from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from longcoqa.data import LongCoQADataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports/generated/analysis/p1_evidence_overlap"
BOOTSTRAP_ITERATIONS = 2_000
BOOTSTRAP_SEED = 42


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pipe(values: Iterable[Any]) -> str:
    return "|".join(map(str, sorted(set(values), key=str)))


def mean_or_none(values: pd.Series) -> float | None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.mean()) if len(clean) else None


def normalize_result_metrics(metrics: dict[str, Any] | None) -> dict[str, float | None]:
    metrics = metrics or {}
    return {
        "character_f1": metrics.get("character_f1"),
        "rouge_l_f1": metrics.get("rouge_l_f1"),
        "exact_match": metrics.get("exact_match"),
    }


def cluster_bootstrap_mean(
    rows: pd.DataFrame,
    value_column: str,
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    values = rows[["dialogue_key", value_column]].dropna()
    if values.empty:
        return float("nan"), float("nan"), float("nan")
    grouped = {
        key: group[value_column].to_numpy(dtype=float)
        for key, group in values.groupby("dialogue_key", sort=True)
    }
    keys = np.array(sorted(grouped), dtype=object)
    observed = float(values[value_column].mean())
    if len(keys) == 1:
        return observed, observed, observed
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=float)
    for index in range(iterations):
        sampled_keys = rng.choice(keys, size=len(keys), replace=True)
        sampled = np.concatenate([grouped[key] for key in sampled_keys])
        estimates[index] = float(sampled.mean())
    low, high = np.quantile(estimates, [0.025, 0.975])
    return observed, float(low), float(high)


def build_location_maps(
    alignments: dict[int, dict[str, Any]], parents: list[dict[str, Any]]
) -> tuple[dict[str, str], dict[int, dict[str, set[str]]]]:
    paragraph_to_parents: dict[str, set[str]] = defaultdict(set)
    for parent in parents:
        parent_id = str(parent["parent_id"])
        for paragraph_id in parent.get("paragraph_ids", []):
            paragraph_to_parents[str(paragraph_id)].add(parent_id)
    ambiguous = {key: value for key, value in paragraph_to_parents.items() if len(value) != 1}
    if ambiguous:
        raise ValueError(f"Paragraph-to-parent mapping is not one-to-one: {list(ambiguous.items())[:5]}")
    paragraph_to_parent = {key: next(iter(value)) for key, value in paragraph_to_parents.items()}

    locations: dict[int, dict[str, set[str]]] = {}
    for evidence_id, row in alignments.items():
        paragraph_ids = {str(value) for value in row.get("paragraph_ids", [])}
        parent_ids = {
            paragraph_to_parent[paragraph_id]
            for paragraph_id in paragraph_ids
            if paragraph_id in paragraph_to_parent
        }
        chapter_ids = {str(row["chapter_id"])} if row.get("chapter_id") else set()
        book_ids = {str(row["book_id"])} if row.get("book_id") is not None else set()
        locations[evidence_id] = {
            "paragraph_ids": paragraph_ids,
            "parent_ids": parent_ids,
            "chapter_ids": chapter_ids,
            "book_ids": book_ids,
        }
    return paragraph_to_parent, locations


def union_locations(
    evidence_ids: Iterable[int], locations: dict[int, dict[str, set[str]]]
) -> dict[str, set[str]]:
    result = {
        "paragraph_ids": set(),
        "parent_ids": set(),
        "chapter_ids": set(),
        "book_ids": set(),
    }
    for evidence_id in evidence_ids:
        location = locations.get(int(evidence_id), {})
        for key in result:
            result[key].update(location.get(key, set()))
    return result


def exposure_label(
    evidence_ids: list[int],
    test_locations: dict[str, set[str]],
    train_evidence_ids: set[int],
    train_locations: dict[str, set[str]],
) -> tuple[str, str, str]:
    if not evidence_ids:
        return "no_gold_annotation", "no_gold_annotation", "no_gold_annotation"

    seen_ids = set(evidence_ids) & train_evidence_ids
    if len(seen_ids) == len(set(evidence_ids)):
        exact_status = "all_exact_seen"
    elif seen_ids:
        exact_status = "partial_exact_seen"
    else:
        exact_status = "exact_unseen"
    exact_binary = "exact_seen" if seen_ids else "exact_unseen"

    if seen_ids:
        hierarchy = "exact_seen"
    elif test_locations["paragraph_ids"] & train_locations["paragraph_ids"]:
        hierarchy = "paragraph_only"
    elif test_locations["parent_ids"] & train_locations["parent_ids"]:
        hierarchy = "parent_only"
    elif test_locations["chapter_ids"] & train_locations["chapter_ids"]:
        hierarchy = "chapter_only"
    else:
        hierarchy = "strict_unseen_chapter"
    return exact_status, exact_binary, hierarchy


def build_exposure_frame() -> tuple[pd.DataFrame, dict[str, Any], dict[str, str]]:
    dataset = LongCoQADataset(PROJECT_ROOT / "data")
    samples = list(dataset.iter_samples())
    samples_by_id = {str(row["sample_id"]): row for row in samples}
    if len(samples_by_id) != len(samples):
        raise ValueError("Released dataset contains duplicate sample IDs")

    manifest_path = PROJECT_ROOT / "artifacts/manifests/p1_dialogue_split.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_by_id = {str(row["sample_id"]): str(row["split"]) for row in manifest["samples"]}
    if set(split_by_id) != set(samples_by_id):
        raise ValueError("Manifest and released dataset sample IDs do not match")

    alignment_path = PROJECT_ROOT / "artifacts/alignments/evidence_alignment.jsonl"
    parent_path = PROJECT_ROOT / "artifacts/corpus/parents.jsonl"
    alignments = {int(row["evidence_id"]): row for row in read_jsonl(alignment_path)}
    parents = read_jsonl(parent_path)
    paragraph_to_parent, locations = build_location_maps(alignments, parents)

    referenced_evidence = {
        int(evidence_id)
        for sample in samples
        for evidence_id in sample.get("evidence_ids", [])
    }
    missing_alignments = referenced_evidence - set(alignments)
    if missing_alignments:
        raise ValueError(f"Referenced evidence lacks alignment rows: {sorted(missing_alignments)[:20]}")

    train_samples = [row for row in samples if split_by_id[str(row["sample_id"])] == "train"]
    train_evidence_ids = {
        int(evidence_id)
        for sample in train_samples
        for evidence_id in sample.get("evidence_ids", [])
    }
    train_locations = union_locations(train_evidence_ids, locations)

    rows: list[dict[str, Any]] = []
    for sample_id, split in sorted(split_by_id.items()):
        if split != "test":
            continue
        sample = samples_by_id[sample_id]
        evidence_ids = [int(value) for value in sample.get("evidence_ids", [])]
        test_locations = union_locations(evidence_ids, locations)
        exact_status, exact_binary, hierarchy = exposure_label(
            evidence_ids,
            test_locations,
            train_evidence_ids,
            train_locations,
        )
        seen_ids = set(evidence_ids) & train_evidence_ids
        rows.append(
            {
                "sample_id": sample_id,
                "dialogue_key": f"{sample['question_type']}:{sample['dialogue_id']}",
                "dialogue_id": int(sample["dialogue_id"]),
                "question_type": str(sample["question_type"]),
                "turn": int(sample["turn"]),
                "evidence_count": len(evidence_ids),
                "evidence_ids": pipe(evidence_ids),
                "seen_evidence_ids": pipe(seen_ids),
                "exact_seen_count": len(seen_ids),
                "exact_seen_ratio": len(seen_ids) / len(set(evidence_ids)) if evidence_ids else None,
                "paragraph_count": len(test_locations["paragraph_ids"]),
                "parent_count": len(test_locations["parent_ids"]),
                "chapter_count": len(test_locations["chapter_ids"]),
                "book_count": len(test_locations["book_ids"]),
                "exact_status": exact_status,
                "exact_binary": exact_binary,
                "hierarchy_status": hierarchy,
            }
        )

    exposure = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
    if len(exposure) != 602 or exposure["sample_id"].nunique() != 602:
        raise ValueError("Expected exactly 602 unique P1 test samples")

    mapped_paragraphs = {
        paragraph_id
        for location in locations.values()
        for paragraph_id in location["paragraph_ids"]
        if paragraph_id in paragraph_to_parent
    }
    all_alignment_paragraphs = {
        paragraph_id
        for location in locations.values()
        for paragraph_id in location["paragraph_ids"]
    }
    quality = {
        "released_samples": len(samples),
        "manifest_samples": len(split_by_id),
        "test_samples": len(exposure),
        "test_dialogues": int(exposure["dialogue_key"].nunique()),
        "referenced_evidence_ids": len(referenced_evidence),
        "alignment_rows": len(alignments),
        "missing_alignment_rows": len(missing_alignments),
        "alignment_paragraph_ids": len(all_alignment_paragraphs),
        "alignment_paragraphs_mapped_to_parent": len(mapped_paragraphs),
        "unmapped_alignment_paragraph_ids": len(all_alignment_paragraphs - mapped_paragraphs),
    }
    sources = {
        str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
        for path in (manifest_path, alignment_path, parent_path)
    }
    return exposure, quality, sources


def load_prediction_and_judge_frames() -> tuple[pd.DataFrame, dict[str, Any], dict[str, str]]:
    prediction_rows: list[dict[str, Any]] = []
    sources: dict[str, str] = {}
    result_roots = [
        PROJECT_ROOT / "outputs/formal/p1",
        PROJECT_ROOT / "outputs/formal/p1_cpt",
    ]
    for root in result_roots:
        for path in sorted(root.glob("merged/*/*/full/results.jsonl")):
            relative = path.relative_to(root)
            model_variant = relative.parts[1]
            eval_variant = relative.parts[2]
            rows = read_jsonl(path)
            sample_ids = [str(row["sample_id"]) for row in rows]
            if len(sample_ids) != len(set(sample_ids)):
                raise ValueError(f"Duplicate prediction sample IDs in {path}")
            for row in rows:
                metrics = normalize_result_metrics(row.get("metrics"))
                prediction_rows.append(
                    {
                        "sample_id": str(row["sample_id"]),
                        "model_variant": model_variant,
                        "eval_variant": eval_variant,
                        "status": str(row.get("status")),
                        "answerable": bool(row.get("answerable", True)),
                        "raft_mode": row.get("raft_mode"),
                        "character_f1": metrics.get("character_f1"),
                        "rouge_l_f1": metrics.get("rouge_l_f1"),
                        "exact_match": metrics.get("exact_match"),
                    }
                )
            sources[str(path.relative_to(PROJECT_ROOT))] = sha256_file(path)

    judgment_rows: list[dict[str, Any]] = []
    judgment_roots = [
        PROJECT_ROOT / "outputs/judgments/qwen37/p1",
        PROJECT_ROOT / "outputs/judgments/qwen37/p1_cpt",
    ]
    for root in judgment_roots:
        for path in sorted(root.glob("*/*/full/judgments.jsonl")):
            relative = path.relative_to(root)
            model_variant = relative.parts[0]
            eval_variant = relative.parts[1]
            rows = read_jsonl(path)
            keys = [(str(row["sample_id"]), str(row.get("judgment_type"))) for row in rows]
            if len(keys) != len(set(keys)):
                raise ValueError(f"Duplicate judgment sample/type keys in {path}")
            for row in rows:
                judgment = row.get("judgment", {})
                correctness = (
                    judgment.get("correctness")
                    if row.get("judgment_type") == "answer_quality"
                    else None
                )
                judgment_rows.append(
                    {
                        "sample_id": str(row["sample_id"]),
                        "model_variant": model_variant,
                        "eval_variant": eval_variant,
                        "judgment_status": str(row.get("status")),
                        "judgment_type": str(row.get("judgment_type")),
                        "judge_correctness": correctness,
                        "judge_100": 25.0 * (float(correctness) - 1.0)
                        if correctness is not None
                        else None,
                    }
                )
            sources[str(path.relative_to(PROJECT_ROOT))] = sha256_file(path)

    predictions = pd.DataFrame(prediction_rows)
    judgments = pd.DataFrame(judgment_rows)
    merged = predictions.merge(
        judgments,
        on=["sample_id", "model_variant", "eval_variant"],
        how="left",
        validate="one_to_one",
    )
    if not (merged["status"] == "ok").all():
        raise ValueError("Prediction results contain non-ok rows")
    if not (merged["judgment_status"] == "ok").all():
        raise ValueError("Judgment results are missing or contain non-ok rows")

    expected = {"conversation": 602, "grounded": 539, "raft": 539}
    cell_counts = (
        merged.groupby(["model_variant", "eval_variant"], sort=True)
        .size()
        .rename("rows")
        .reset_index()
    )
    for row in cell_counts.itertuples(index=False):
        if int(row.rows) != expected[str(row.eval_variant)]:
            raise ValueError(
                f"Unexpected result count for {row.model_variant}/{row.eval_variant}: {row.rows}"
            )
    quality = {
        "prediction_cells": int(len(cell_counts)),
        "prediction_rows": int(len(merged)),
        "answer_quality_judgments": int((merged["judgment_type"] == "answer_quality").sum()),
        "refusal_quality_judgments": int((merged["judgment_type"] == "refusal_quality").sum()),
        "cell_counts": cell_counts.to_dict(orient="records"),
    }
    return merged, quality, sources


def distribution_tables(exposure: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    exact = (
        exposure.groupby("exact_status", sort=False)
        .agg(samples=("sample_id", "size"), dialogues=("dialogue_key", "nunique"))
        .reset_index()
    )
    exact["sample_share"] = exact["samples"] / len(exposure)
    hierarchy = (
        exposure.groupby("hierarchy_status", sort=False)
        .agg(samples=("sample_id", "size"), dialogues=("dialogue_key", "nunique"))
        .reset_index()
    )
    hierarchy["sample_share"] = hierarchy["samples"] / len(exposure)
    return exact, hierarchy


def performance_tables(rows: pd.DataFrame) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    scopes = [("overall", rows)] + [
        (str(question_type), group)
        for question_type, group in rows.groupby("question_type", sort=True)
    ]
    for question_type, scoped_rows in scopes:
        for scheme, column in (
            ("exact_binary", "exact_binary"),
            ("exact_status", "exact_status"),
            ("hierarchy", "hierarchy_status"),
        ):
            for keys, group in scoped_rows.groupby(
                ["model_variant", "eval_variant", column], sort=True, dropna=False
            ):
                model_variant, eval_variant, overlap_group = keys
                answer_rows = group[group["answerable"]].copy()
                output.append(
                    {
                        "question_type": question_type,
                        "group_scheme": scheme,
                        "overlap_group": str(overlap_group),
                        "model_variant": str(model_variant),
                        "eval_variant": str(eval_variant),
                        "samples": int(len(answer_rows)),
                        "dialogues": int(answer_rows["dialogue_key"].nunique()),
                        "character_f1": mean_or_none(answer_rows["character_f1"]),
                        "rouge_l_f1": mean_or_none(answer_rows["rouge_l_f1"]),
                        "exact_match": mean_or_none(answer_rows["exact_match"]),
                        "judge_100": mean_or_none(answer_rows["judge_100"]),
                        "judge_correctness": mean_or_none(answer_rows["judge_correctness"]),
                    }
                )
    return pd.DataFrame(output)


def paired_lift_table(rows: pd.DataFrame) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    eligible = rows[(rows["answerable"]) & (rows["evidence_count"] > 0)].copy()
    base = eligible[eligible["model_variant"] == "base"]
    for eval_variant in sorted(eligible["eval_variant"].unique()):
        base_eval = base[base["eval_variant"] == eval_variant]
        for model_variant in ("conversation", "grounded", "raft", "cpt"):
            candidate = eligible[
                (eligible["model_variant"] == model_variant)
                & (eligible["eval_variant"] == eval_variant)
            ]
            paired = candidate.merge(
                base_eval[
                    ["sample_id", "character_f1", "judge_100"]
                ].rename(
                    columns={
                        "character_f1": "base_character_f1",
                        "judge_100": "base_judge_100",
                    }
                ),
                on="sample_id",
                how="inner",
                validate="one_to_one",
            )
            paired["character_f1_delta"] = paired["character_f1"] - paired["base_character_f1"]
            paired["judge_100_delta"] = paired["judge_100"] - paired["base_judge_100"]
            scopes = [("overall", paired)] + [
                (str(question_type), group)
                for question_type, group in paired.groupby("question_type", sort=True)
            ]
            for question_type, scoped_rows in scopes:
                for group_scheme, group_column in (
                    ("exact_binary", "exact_binary"),
                    ("hierarchy", "hierarchy_status"),
                ):
                    for overlap_group, group in scoped_rows.groupby(group_column, sort=True):
                        char_mean, char_low, char_high = cluster_bootstrap_mean(
                            group,
                            "character_f1_delta",
                            seed=BOOTSTRAP_SEED + len(output),
                        )
                        judge_mean, judge_low, judge_high = cluster_bootstrap_mean(
                            group,
                            "judge_100_delta",
                            seed=BOOTSTRAP_SEED + 10_000 + len(output),
                        )
                        output.append(
                            {
                                "question_type": question_type,
                                "group_scheme": group_scheme,
                                "model_variant": model_variant,
                                "eval_variant": eval_variant,
                                "overlap_group": str(overlap_group),
                                "samples": int(len(group)),
                                "dialogues": int(group["dialogue_key"].nunique()),
                                "character_f1_delta": char_mean,
                                "character_f1_ci_low": char_low,
                                "character_f1_ci_high": char_high,
                                "judge_100_delta": judge_mean,
                                "judge_100_ci_low": judge_low,
                                "judge_100_ci_high": judge_high,
                                "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
                            }
                        )
    return pd.DataFrame(output)


def nested_dict_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", force_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit P1 train/test evidence overlap and grouped results")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    exposure, exposure_quality, exposure_sources = build_exposure_frame()
    results, results_quality, result_sources = load_prediction_and_judge_frames()
    joined = results.merge(exposure, on="sample_id", how="left", validate="many_to_one")
    if joined["exact_status"].isna().any():
        missing = joined.loc[joined["exact_status"].isna(), "sample_id"].unique().tolist()
        raise ValueError(f"Results include samples outside P1 test manifest: {missing[:20]}")

    exact_distribution, hierarchy_distribution = distribution_tables(exposure)
    performance = performance_tables(joined)
    paired_lifts = paired_lift_table(joined)

    exposure.to_csv(output_dir / "sample_exposure.csv", index=False)
    exact_distribution.to_csv(output_dir / "exact_overlap_distribution.csv", index=False)
    hierarchy_distribution.to_csv(output_dir / "hierarchical_overlap_distribution.csv", index=False)
    performance.to_csv(output_dir / "performance_by_overlap.csv", index=False)
    paired_lifts.to_csv(output_dir / "paired_lifts_by_exact_overlap.csv", index=False)

    source_hashes = {**exposure_sources, **result_sources}
    analysis = {
        "schema_version": "longcoqa-p1-evidence-overlap-v1",
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "definitions": {
            "all_exact_seen": "Every gold evidence ID for the test sample occurs in supervised P1 train.",
            "partial_exact_seen": "At least one but not all gold evidence IDs occur in supervised P1 train.",
            "exact_unseen": "No gold evidence ID occurs in supervised P1 train.",
            "hierarchy": (
                "For exact-unseen samples, the closest shared source unit is paragraph, "
                "parent passage, chapter, or none at chapter level."
            ),
            "no_gold_annotation": (
                "Released data provides no gold evidence IDs; not interpreted as unseen evidence."
            ),
            "paired_lift": (
                "Candidate minus Base on identical samples and eval protocol; "
                "95% intervals resample dialogue clusters."
            ),
        },
        "data_quality": {**exposure_quality, **results_quality},
        "source_hashes": source_hashes,
        "exact_overlap_distribution": nested_dict_records(exact_distribution),
        "hierarchical_overlap_distribution": nested_dict_records(hierarchy_distribution),
        "performance_by_overlap": nested_dict_records(performance),
        "paired_lifts_by_exact_overlap": nested_dict_records(paired_lifts),
    }
    (output_dir / "analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output_dir": str(output_dir),
        "test_samples": len(exposure),
        "exact_overlap_distribution": analysis["exact_overlap_distribution"],
        "hierarchical_overlap_distribution": analysis["hierarchical_overlap_distribution"],
        "prediction_cells": results_quality["prediction_cells"],
        "prediction_rows": results_quality["prediction_rows"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
