from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from longcoqa.evaluation.reporting import extended_score_answer
from longcoqa.evaluation.text_metrics import normalize_answer
from longcoqa.paths import PROJECT_ROOT

BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_SEED = 42

MODEL_ROOTS = {
    "qwen3_5_9b": PROJECT_ROOT / "outputs/formal/main/qwen3_5_9b",
    "qwen3_6_27b": PROJECT_ROOT / "outputs/formal/main/qwen3_6_27b",
    "qwen3_8b": PROJECT_ROOT / "outputs/formal/models/qwen3_8b",
    "glm4_7_flash": PROJECT_ROOT / "outputs/formal/models/glm4_7_flash",
    "qwen3_5_35b_a3b": PROJECT_ROOT / "outputs/formal/models/qwen3_5_35b_a3b",
    "gpt_oss_120b": PROJECT_ROOT / "outputs/formal/models/gpt_oss_120b",
    "qwen37": PROJECT_ROOT / "outputs/formal/models/qwen37",
}
PROTOCOLS = ("z0_current", "z1_teacher_forced", "g0_released_gold", "g3_aligned_raw")

RETRIEVAL_SOURCES = {
    "bm25_current": PROJECT_ROOT / "reports/generated/retrieval/bm25_current_question.jsonl",
    "bm25_history": PROJECT_ROOT / "reports/generated/retrieval/bm25_history_questions.jsonl",
    "dense_current": PROJECT_ROOT / "reports/generated/retrieval/qwen3_embedding_current_question.jsonl",
    "dense_history": PROJECT_ROOT / "reports/generated/retrieval/qwen3_embedding_history_questions.jsonl",
    "hybrid_current": PROJECT_ROOT / "reports/generated/retrieval/hybrid_rrf_current_question.jsonl",
    "hybrid_history": PROJECT_ROOT / "reports/generated/retrieval/hybrid_rrf_history_questions.jsonl",
    "rewritten_dense": PROJECT_ROOT / "reports/generated/rag/rewritten_dense/results.jsonl",
    "contextual_bm25_current": PROJECT_ROOT / "reports/generated/rag/contextual_retrieval/contextual_bm25_current_question.jsonl",
    "contextual_bm25_rewritten": PROJECT_ROOT / "reports/generated/rag/contextual_retrieval/contextual_bm25_rewritten_dialogue.jsonl",
    "contextual_dense_current": PROJECT_ROOT / "reports/generated/rag/contextual_retrieval/contextual_dense_current_question.jsonl",
    "contextual_dense_rewritten": PROJECT_ROOT / "reports/generated/rag/contextual_retrieval/contextual_dense_rewritten_dialogue.jsonl",
    "contextual_rrf_current": PROJECT_ROOT / "reports/generated/rag/contextual_retrieval/contextual_rrf_current_question.jsonl",
    "contextual_rrf_rewritten": PROJECT_ROOT / "reports/generated/rag/contextual_retrieval/contextual_rrf_rewritten_dialogue.jsonl",
    "graph_progressive": PROJECT_ROOT / "reports/generated/rag/graph_progressive/results.jsonl",
    "progressive": PROJECT_ROOT / "reports/generated/rag/progressive/results.jsonl",
    "reranker_hybrid_current": PROJECT_ROOT / "reports/generated/rag/reranker_current/results.jsonl",
    "reranker_dense_current": PROJECT_ROOT / "reports/generated/rag/reranker_dense_current/results.jsonl",
}

RAG_CHILD_SOURCES = {
    "rewritten_dense": (
        RETRIEVAL_SOURCES["rewritten_dense"],
        PROJECT_ROOT / "outputs/rag/qwen3_5_9b_rewritten_parent_top10/merged/results.jsonl",
    ),
    "contextual_dense_rewritten": (
        RETRIEVAL_SOURCES["contextual_dense_rewritten"],
        PROJECT_ROOT / "outputs/rag/qwen3_5_9b_contextual_parent_top10/merged/results.jsonl",
    ),
    "graph_progressive": (
        RETRIEVAL_SOURCES["graph_progressive"],
        PROJECT_ROOT / "outputs/rag/qwen3_5_9b_graph_progressive_parent_top10/merged/results.jsonl",
    ),
    "progressive": (
        RETRIEVAL_SOURCES["progressive"],
        PROJECT_ROOT / "outputs/rag/qwen3_5_9b_progressive_parent_top10/merged/results.jsonl",
    ),
}
RAG_PARENT_SOURCE = (
    PROJECT_ROOT / "reports/generated/rag/hipporag2/full/results.jsonl",
    PROJECT_ROOT / "outputs/rag/qwen3_5_9b_hipporag2_parent_top10/merged/results.jsonl",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def turn_bucket(turn: int) -> str:
    if turn < 1:
        raise ValueError("turn must be positive")
    if turn == 1:
        return "turn_1"
    if turn <= 4:
        return "turn_2_4"
    return "turn_5_plus"


def score_generation_row(row: Mapping[str, Any]) -> dict[str, Any]:
    sample_id = str(row["sample_id"])
    parts = sample_id.split(":")
    question_type = str(row.get("question_type", parts[0]))
    dialogue_id = int(row.get("dialogue_id", parts[1]))
    turn = int(row.get("turn", parts[2]))
    prediction = str(row["prediction"])
    reference = str(row["reference"])
    stored_metrics = row.get("metrics")
    if isinstance(stored_metrics, Mapping) and "character_f1" in stored_metrics:
        predicted = normalize_answer(prediction)
        expected = normalize_answer(reference)
        metrics = {
            "character_f1": float(stored_metrics["character_f1"]),
            "exact_match_contain": float(bool(expected) and expected in predicted),
        }
    else:
        metrics = extended_score_answer(prediction, reference)
    return {
        **dict(row),
        "sample_id": sample_id,
        "question_type": question_type,
        "dialogue_id": dialogue_id,
        "turn": turn,
        "turn_bucket": turn_bucket(turn),
        "cluster": f"{question_type}:{dialogue_id}",
        "prediction_length": len(normalize_answer(prediction)),
        "reference_length": len(normalize_answer(reference)),
        **metrics,
    }


def validate_formal_rows(
    rows: Sequence[Mapping[str, Any]], *, expected: int, label: str
) -> None:
    if len(rows) != expected:
        raise ValueError(f"{label}: expected {expected} rows, found {len(rows)}")
    failed = [row for row in rows if row.get("status") != "ok"]
    if failed:
        raise ValueError(f"{label}: found {len(failed)} failed rows")
    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError(f"{label}: duplicate sample IDs")


def analyze_within_question_hit_miss(
    rows: Sequence[Mapping[str, Any]], *, metric: str
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["sample_id"]), []).append(row)
    output = []
    for sample_id, items in sorted(grouped.items()):
        hit = [float(item[metric]) for item in items if int(item["hit"]) == 1]
        miss = [float(item[metric]) for item in items if int(item["hit"]) == 0]
        if not hit or not miss:
            continue
        hit_mean = float(np.mean(hit))
        miss_mean = float(np.mean(miss))
        parts = sample_id.split(":")
        output.append(
            {
                "sample_id": sample_id,
                "question_type": parts[0],
                "dialogue_id": int(parts[1]),
                "cluster": f"{parts[0]}:{parts[1]}",
                "hit_methods": len(hit),
                "miss_methods": len(miss),
                "hit_mean": hit_mean,
                "miss_mean": miss_mean,
                "delta": hit_mean - miss_mean,
            }
        )
    return output


def _validated_numeric(values: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not len(array):
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _cluster_indices(clusters: Sequence[object] | np.ndarray, expected: int) -> list[np.ndarray]:
    labels = np.asarray(clusters, dtype=object)
    if labels.ndim != 1 or len(labels) != expected:
        raise ValueError("clusters must be one-dimensional and match the value count")
    unique = sorted(set(labels.tolist()), key=str)
    if not unique:
        raise ValueError("clusters must not be empty")
    return [np.flatnonzero(labels == label) for label in unique]


def _bootstrap_means(
    values: np.ndarray,
    cluster_rows: list[np.ndarray],
    *,
    iterations: int,
    seed: int,
) -> np.ndarray:
    if iterations < 1:
        raise ValueError("iterations must be positive")
    rng = np.random.default_rng(seed)
    cluster_count = len(cluster_rows)
    cluster_sums = np.asarray([values[rows].sum() for rows in cluster_rows], dtype=float)
    cluster_sizes = np.asarray([len(rows) for rows in cluster_rows], dtype=float)
    weights = rng.multinomial(
        cluster_count,
        np.full(cluster_count, 1.0 / cluster_count),
        size=iterations,
    ).astype(float)
    numerators = np.einsum("bc,c->b", weights, cluster_sums)
    denominators = np.einsum("bc,c->b", weights, cluster_sizes)
    return numerators / denominators


def cluster_bootstrap_mean(
    values: Sequence[float] | np.ndarray,
    clusters: Sequence[object] | np.ndarray,
    *,
    iterations: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    numeric = _validated_numeric(values, name="values")
    cluster_rows = _cluster_indices(clusters, len(numeric))
    bootstrap = _bootstrap_means(
        numeric,
        cluster_rows,
        iterations=iterations,
        seed=seed,
    )
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return float(numeric.mean()), float(low), float(high)


def paired_cluster_bootstrap(
    left: Sequence[float] | np.ndarray,
    right: Sequence[float] | np.ndarray,
    clusters: Sequence[object] | np.ndarray,
    *,
    iterations: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float, float]:
    left_values = _validated_numeric(left, name="left")
    right_values = _validated_numeric(right, name="right")
    if len(left_values) != len(right_values):
        raise ValueError("paired arrays must have the same length")
    deltas = right_values - left_values
    cluster_rows = _cluster_indices(clusters, len(deltas))
    bootstrap = _bootstrap_means(
        deltas,
        cluster_rows,
        iterations=iterations,
        seed=seed,
    )
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    lower_tail = (float(np.count_nonzero(bootstrap <= 0.0)) + 1.0) / (iterations + 1.0)
    upper_tail = (float(np.count_nonzero(bootstrap >= 0.0)) + 1.0) / (iterations + 1.0)
    p_value = min(1.0, 2.0 * min(lower_tail, upper_tail))
    return float(deltas.mean()), float(low), float(high), p_value


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    values = _validated_numeric(p_values, name="p_values")
    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("p_values must be within [0, 1]")
    order = np.argsort(values, kind="stable")
    adjusted_sorted = np.empty(len(values), dtype=float)
    running = 0.0
    count = len(values)
    for rank, original_index in enumerate(order):
        candidate = float(values[original_index]) * (count - rank)
        running = max(running, candidate)
        adjusted_sorted[rank] = min(1.0, running)
    adjusted = np.empty(len(values), dtype=float)
    for rank, original_index in enumerate(order):
        adjusted[original_index] = adjusted_sorted[rank]
    return adjusted.tolist()


def validate_equal_sample_ids(sample_ids: Mapping[str, Sequence[str]]) -> tuple[str, ...]:
    if not sample_ids:
        raise ValueError("at least one sample collection is required")
    normalized: dict[str, tuple[str, ...]] = {}
    for name, values in sample_ids.items():
        current = tuple(str(value) for value in values)
        if len(set(current)) != len(current):
            raise ValueError(f"duplicate sample IDs in {name}")
        normalized[name] = tuple(sorted(current))
    reference_name, reference = next(iter(normalized.items()))
    for name, current in normalized.items():
        if current != reference:
            raise ValueError(f"sample ID mismatch between {reference_name} and {name}")
    return reference


def fit_ols(design: Sequence[Sequence[float]] | np.ndarray, outcome: Sequence[float]) -> np.ndarray:
    matrix = np.asarray(design, dtype=float)
    values = _validated_numeric(outcome, name="outcome")
    if matrix.ndim != 2 or matrix.shape[0] != len(values) or not matrix.shape[1]:
        raise ValueError("design must be a non-empty matrix with one row per outcome")
    if not np.isfinite(matrix).all():
        raise ValueError("design contains non-finite values")
    coefficients, _, _, _ = np.linalg.lstsq(matrix, values, rcond=None)
    return coefficients


def _cluster_bootstrap_statistic(
    frame: pd.DataFrame,
    statistic: Any,
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    clusters = sorted(frame["cluster"].unique())
    grouped = {cluster: frame.loc[frame["cluster"] == cluster] for cluster in clusters}
    observed = float(statistic(frame))
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations)
    for index in range(iterations):
        selected = rng.choice(clusters, size=len(clusters), replace=True)
        sampled = pd.concat([grouped[item] for item in selected], ignore_index=True)
        draws[index] = float(statistic(sampled))
    low, high = np.quantile(draws, [0.025, 0.975])
    return observed, float(low), float(high)


def _cluster_bootstrap_slope(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    labels = sorted(frame["cluster"].unique())
    sufficient = []
    for label in labels:
        subset = frame.loc[frame["cluster"] == label]
        x_values = subset[x].to_numpy(float)
        y_values = subset[y].to_numpy(float)
        sufficient.append(
            [
                len(subset),
                x_values.sum(),
                y_values.sum(),
                np.square(x_values).sum(),
                (x_values * y_values).sum(),
            ]
        )
    stats = np.asarray(sufficient, dtype=float)
    observed = np.polyfit(frame[x].to_numpy(float), frame[y].to_numpy(float), 1)[0]
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(
        len(labels),
        np.full(len(labels), 1.0 / len(labels)),
        size=iterations,
    ).astype(float)
    totals = np.einsum("bc,ck->bk", weights, stats)
    count, sum_x, sum_y, sum_xx, sum_xy = totals.T
    draws = (sum_xy - sum_x * sum_y / count) / (sum_xx - np.square(sum_x) / count)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(observed), float(low), float(high)


def _cluster_bootstrap_ols(
    design: np.ndarray,
    outcome: np.ndarray,
    clusters: np.ndarray,
    *,
    coefficient: int,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    labels = sorted(set(clusters.tolist()), key=str)
    xtx = []
    xty = []
    for label in labels:
        rows = clusters == label
        current = design[rows]
        xtx.append(np.einsum("ni,nj->ij", current, current))
        xty.append(np.einsum("ni,n->i", current, outcome[rows]))
    xtx_array = np.stack(xtx)
    xty_array = np.stack(xty)
    observed = float(fit_ols(design, outcome)[coefficient])
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations)
    for index in range(iterations):
        selected = rng.integers(0, len(labels), size=len(labels))
        beta = np.einsum(
            "ij,j->i",
            np.linalg.pinv(xtx_array[selected].sum(axis=0)),
            xty_array[selected].sum(axis=0),
        )
        draws[index] = beta[coefficient]
    low, high = np.quantile(draws, [0.025, 0.975])
    return observed, float(low), float(high)


def _fixed_effect_design(frame: pd.DataFrame, numeric: list[str], categorical: list[str]) -> np.ndarray:
    columns = [np.ones(len(frame))]
    columns.extend(frame[field].to_numpy(dtype=float) for field in numeric)
    for field in categorical:
        values = sorted(frame[field].astype(str).unique())
        columns.extend((frame[field].astype(str) == value).to_numpy(dtype=float) for value in values[1:])
    return np.column_stack(columns)


def load_p0_generation_rows() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows = []
    manifest = []
    for model, root in MODEL_ROOTS.items():
        for protocol in PROTOCOLS:
            path = root / protocol / "full/results.jsonl"
            raw = read_jsonl(path)
            label = f"{model}/{protocol}"
            validate_formal_rows(raw, expected=3019, label=label)
            for item in raw:
                rows.append({**score_generation_row(item), "model": model, "protocol": protocol})
            manifest.append(
                {
                    "kind": "p0_generation",
                    "cell": label,
                    "path": str(path.relative_to(PROJECT_ROOT)),
                    "rows": len(raw),
                    "sha256": sha256_file(path),
                }
            )
    frame = pd.DataFrame(rows)
    if len(frame) != 7 * 4 * 3019:
        raise ValueError("P0 generation matrix is incomplete")
    return frame, manifest


def load_child_retrieval_rows() -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    frames = {}
    manifests = []
    identifiers = {}
    required = ("hit_at_20", "recall_at_20", "ndcg_at_20", "reciprocal_rank")
    for method, path in RETRIEVAL_SOURCES.items():
        raw = read_jsonl(path)
        if len(raw) != 2721:
            raise ValueError(f"{method}: expected 2721 rows, found {len(raw)}")
        ids = [str(row["sample_id"]) for row in raw]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{method}: duplicate sample IDs")
        for row in raw:
            if any(metric not in row for metric in required):
                raise ValueError(f"{method}: missing retrieval metrics")
        frame = pd.DataFrame(raw).sort_values("sample_id").reset_index(drop=True)
        parts = frame["sample_id"].str.split(":", expand=True)
        frame["question_type"] = parts[0]
        frame["dialogue_id"] = parts[1].astype(int)
        frame["cluster"] = parts[0] + ":" + parts[1]
        frame["method_key"] = method
        frames[method] = frame
        identifiers[method] = ids
        manifests.append(
            {
                "kind": "child_retrieval",
                "cell": method,
                "path": str(path.relative_to(PROJECT_ROOT)),
                "rows": len(raw),
                "sha256": sha256_file(path),
            }
        )
    validate_equal_sample_ids(identifiers)
    return frames, manifests


def run_m3(generation: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    buckets = []
    slopes = []
    contrasts = []
    macro = (
        generation.groupby(
            ["protocol", "sample_id", "question_type", "dialogue_id", "turn", "turn_bucket", "cluster"],
            as_index=False,
        )["character_f1"]
        .mean()
    )
    for (protocol, question_type, bucket), subset in macro.groupby(
        ["protocol", "question_type", "turn_bucket"], sort=True
    ):
        estimate, low, high = cluster_bootstrap_mean(
            subset["character_f1"].to_numpy(),
            subset["cluster"].to_numpy(),
        )
        buckets.append(
            {
                "protocol": protocol,
                "question_type": question_type,
                "turn_bucket": bucket,
                "samples": len(subset),
                "model_observations": len(subset) * 7,
                "character_f1": estimate,
                "ci_low": low,
                "ci_high": high,
            }
        )
    for (protocol, question_type), subset in macro.groupby(["protocol", "question_type"], sort=True):
        estimate, low, high = _cluster_bootstrap_slope(
            subset,
            x="turn",
            y="character_f1",
        )
        slopes.append(
            {
                "protocol": protocol,
                "question_type": question_type,
                "samples": len(subset),
                "slope_per_turn": estimate,
                "ci_low": low,
                "ci_high": high,
            }
        )
    for left, right, contrast in (
        ("z0_current", "z1_teacher_forced", "z1_minus_z0"),
        ("z1_teacher_forced", "g3_aligned_raw", "g3_minus_z1"),
    ):
        paired = generation[generation["protocol"].isin([left, right])].pivot(
            index=["model", "sample_id", "question_type", "dialogue_id", "turn_bucket", "cluster"],
            columns="protocol",
            values="character_f1",
        ).reset_index()
        paired["delta"] = paired[right] - paired[left]
        for (question_type, bucket), subset in paired.groupby(["question_type", "turn_bucket"]):
            estimate, low, high = cluster_bootstrap_mean(subset["delta"], subset["cluster"])
            contrasts.append(
                {
                    "contrast": contrast,
                    "question_type": question_type,
                    "turn_bucket": bucket,
                    "samples": subset["sample_id"].nunique(),
                    "model_observations": len(subset),
                    "delta_character_f1": estimate,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return pd.DataFrame(buckets), pd.DataFrame(slopes), pd.DataFrame(contrasts)


def run_m4(generation: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = generation[
        (generation["question_type"] == "factoid")
        & generation["protocol"].isin(["z1_teacher_forced", "g3_aligned_raw"])
    ].copy()
    summaries = []
    for (model, protocol), subset in frame.groupby(["model", "protocol"]):
        summaries.append(
            {
                "model": model,
                "protocol": protocol,
                "samples": len(subset),
                "prediction_length": float(subset["prediction_length"].mean()),
                "reference_length": float(subset["reference_length"].mean()),
                "character_f1": float(subset["character_f1"].mean()),
                "exact_match_contain": float(subset["exact_match_contain"].mean()),
            }
        )
    paired = frame.pivot(
        index=["model", "sample_id", "dialogue_id", "cluster"],
        columns="protocol",
        values=["prediction_length", "character_f1", "exact_match_contain"],
    ).reset_index()
    paired.columns = ["__".join(value).strip("_") if isinstance(value, tuple) else value for value in paired.columns]
    paired["delta_log_length"] = np.log1p(paired["prediction_length__g3_aligned_raw"]) - np.log1p(
        paired["prediction_length__z1_teacher_forced"]
    )
    paired["delta_character_f1"] = paired["character_f1__g3_aligned_raw"] - paired[
        "character_f1__z1_teacher_forced"
    ]
    paired["delta_exact_match_contain"] = paired["exact_match_contain__g3_aligned_raw"] - paired[
        "exact_match_contain__z1_teacher_forced"
    ]
    changes = []
    for model, subset in paired.groupby("model"):
        changes.append(
            {
                "model": model,
                "samples": len(subset),
                "delta_prediction_length": float(
                    (subset["prediction_length__g3_aligned_raw"] - subset["prediction_length__z1_teacher_forced"]).mean()
                ),
                "delta_character_f1": float(subset["delta_character_f1"].mean()),
                "delta_exact_match_contain": float(subset["delta_exact_match_contain"].mean()),
                "corr_length_character_f1": float(
                    subset[["delta_log_length", "delta_character_f1"]].corr().iloc[0, 1]
                ),
            }
        )
    regressions = []
    frame["log_prediction_length"] = np.log1p(frame["prediction_length"])
    frame["log_reference_length"] = np.log1p(frame["reference_length"])
    frame["z_log_prediction_length"] = (
        frame["log_prediction_length"] - frame["log_prediction_length"].mean()
    ) / frame["log_prediction_length"].std(ddof=0)
    for outcome in ("character_f1", "exact_match_contain"):
        design = _fixed_effect_design(
            frame,
            ["z_log_prediction_length", "log_reference_length"],
            ["model", "protocol"],
        )
        estimate, low, high = _cluster_bootstrap_ols(
            design,
            frame[outcome].to_numpy(float),
            frame["cluster"].to_numpy(object),
            coefficient=1,
        )
        regressions.append(
            {
                "model_spec": "pooled_model_protocol_reference_length_fe",
                "outcome": outcome,
                "samples": len(frame),
                "coefficient_per_sd_log_prediction_length": estimate,
                "ci_low": low,
                "ci_high": high,
            }
        )
    paired["z_delta_log_length"] = (
        paired["delta_log_length"] - paired["delta_log_length"].mean()
    ) / paired["delta_log_length"].std(ddof=0)
    for outcome in ("delta_character_f1", "delta_exact_match_contain"):
        design = _fixed_effect_design(paired, ["z_delta_log_length"], ["model"])
        estimate, low, high = _cluster_bootstrap_ols(
            design,
            paired[outcome].to_numpy(float),
            paired["cluster"].to_numpy(object),
            coefficient=1,
        )
        regressions.append(
            {
                "model_spec": "within_question_g3_minus_z1_model_fe",
                "outcome": outcome,
                "samples": len(paired),
                "coefficient_per_sd_log_prediction_length": estimate,
                "ci_low": low,
                "ci_high": high,
            }
        )
    return pd.DataFrame(summaries), pd.DataFrame(changes), pd.DataFrame(regressions)


def run_r1(
    retrieval_frames: Mapping[str, pd.DataFrame],
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare retrieval methods on the exact common sample set.

    Dialogue clusters are resampled once per bootstrap draw and reused for every
    method/metric so that all differences remain paired.
    """
    metrics = ("hit_at_20", "recall_at_20", "ndcg_at_20", "reciprocal_rank")
    methods = sorted(retrieval_frames)
    if len(methods) < 2:
        raise ValueError("R1 requires at least two retrieval methods")
    validate_equal_sample_ids(
        {method: frame["sample_id"].astype(str).tolist() for method, frame in retrieval_frames.items()}
    )
    aligned = {
        method: frame.sort_values("sample_id").reset_index(drop=True) for method, frame in retrieval_frames.items()
    }
    reference = aligned[methods[0]]
    clusters = reference["cluster"].astype(str).to_numpy()
    for method, frame in aligned.items():
        if not np.array_equal(frame["cluster"].astype(str).to_numpy(), clusters):
            raise ValueError(f"cluster mismatch in {method}")
        missing = [metric for metric in metrics if metric not in frame]
        if missing:
            raise ValueError(f"{method}: missing metrics {missing}")

    values = np.stack(
        [aligned[method].loc[:, metrics].to_numpy(dtype=float) for method in methods],
        axis=1,
    )
    if not np.isfinite(values).all():
        raise ValueError("retrieval metrics contain non-finite values")
    labels = sorted(set(clusters.tolist()))
    cluster_sums = np.stack([values[clusters == label].sum(axis=0) for label in labels])
    cluster_counts = np.asarray([(clusters == label).sum() for label in labels], dtype=float)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(
        len(labels),
        np.full(len(labels), 1.0 / len(labels)),
        size=iterations,
    ).astype(float)
    denominators = np.einsum("bc,c->b", weights, cluster_counts)
    bootstrap = np.einsum("bc,cmk->bmk", weights, cluster_sums) / denominators[:, None, None]
    observed = values.mean(axis=0)

    summaries = []
    for method_index, method in enumerate(methods):
        for metric_index, metric in enumerate(metrics):
            low, high = np.quantile(bootstrap[:, method_index, metric_index], [0.025, 0.975])
            summaries.append(
                {
                    "method": method,
                    "metric": metric,
                    "samples": len(reference),
                    "dialogue_clusters": len(labels),
                    "estimate": float(observed[method_index, metric_index]),
                    "ci_low": float(low),
                    "ci_high": float(high),
                }
            )

    comparisons = []
    for left_index, right_index in itertools.combinations(range(len(methods)), 2):
        for metric_index, metric in enumerate(metrics):
            draws = bootstrap[:, right_index, metric_index] - bootstrap[:, left_index, metric_index]
            low, high = np.quantile(draws, [0.025, 0.975])
            p_empirical = min(
                1.0,
                2.0
                * min(
                    (float(np.count_nonzero(draws <= 0.0)) + 1.0) / (iterations + 1.0),
                    (float(np.count_nonzero(draws >= 0.0)) + 1.0) / (iterations + 1.0),
                ),
            )
            observed_delta = float(observed[right_index, metric_index] - observed[left_index, metric_index])
            bootstrap_se = float(draws.std(ddof=1))
            p_value = (
                math.erfc(abs(observed_delta) / (math.sqrt(2.0) * bootstrap_se))
                if bootstrap_se > 0.0
                else float(observed_delta == 0.0)
            )
            comparisons.append(
                {
                    "left_method": methods[left_index],
                    "right_method": methods[right_index],
                    "metric": metric,
                    "samples": len(reference),
                    "delta_right_minus_left": observed_delta,
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "p_bootstrap": p_value,
                    "p_empirical_mc": p_empirical,
                }
            )
    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame["p_holm"] = np.nan
    for metric, indices in comparison_frame.groupby("metric").groups.items():
        adjusted = holm_adjust(comparison_frame.loc[indices, "p_bootstrap"].to_numpy())
        comparison_frame.loc[indices, "p_holm"] = adjusted
    return pd.DataFrame(summaries), comparison_frame


def summarize_hit_miss(
    frame: pd.DataFrame,
    *,
    metric: str,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"sample_id", "cluster", "method", "hit", metric}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"hit/miss analysis missing columns: {sorted(missing)}")
    conditional = []
    for (method, hit), subset in frame.groupby(["method", "hit"], sort=True):
        estimate, low, high = cluster_bootstrap_mean(
            subset[metric].to_numpy(float),
            subset["cluster"].to_numpy(object),
            iterations=iterations,
            seed=seed,
        )
        conditional.append(
            {
                "method": method,
                "metric": metric,
                "condition": "hit" if int(hit) else "miss",
                "samples": len(subset),
                "estimate": estimate,
                "ci_low": low,
                "ci_high": high,
            }
        )

    discordant_rows = analyze_within_question_hit_miss(frame.to_dict("records"), metric=metric)
    if not discordant_rows:
        discordant = pd.DataFrame(
            columns=[
                "metric",
                "discordant_samples",
                "dialogue_clusters",
                "delta_hit_minus_miss",
                "ci_low",
                "ci_high",
            ]
        )
    else:
        discordant_frame = pd.DataFrame(discordant_rows)
        estimate, low, high = cluster_bootstrap_mean(
            discordant_frame["delta"].to_numpy(float),
            discordant_frame["cluster"].to_numpy(object),
            iterations=iterations,
            seed=seed,
        )
        discordant = pd.DataFrame(
            [
                {
                    "metric": metric,
                    "discordant_samples": len(discordant_frame),
                    "dialogue_clusters": discordant_frame["cluster"].nunique(),
                    "delta_hit_minus_miss": estimate,
                    "ci_low": low,
                    "ci_high": high,
                }
            ]
        )
    return pd.DataFrame(conditional), discordant


def load_rag_generation_rows(
    retrieval_path: Path,
    generation_path: Path,
    *,
    method: str,
    retrieval_id_field: str = "sample_id",
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    retrieval = read_jsonl(retrieval_path)
    generation = read_jsonl(generation_path)
    if len(retrieval) != 2721:
        raise ValueError(f"{method}: expected 2721 retrieval rows, found {len(retrieval)}")
    validate_formal_rows(generation, expected=3019, label=f"{method}/generation")
    retrieval_ids = [str(row[retrieval_id_field]) for row in retrieval]
    if len(retrieval_ids) != len(set(retrieval_ids)):
        raise ValueError(f"{method}: duplicate retrieval sample IDs")
    retrieval_frame = pd.DataFrame(
        {
            "sample_id": retrieval_ids,
            "hit": [int(float(row["hit_at_10"]) > 0.0) for row in retrieval],
        }
    )
    generation_frame = pd.DataFrame([score_generation_row(row) for row in generation])
    joined = retrieval_frame.merge(
        generation_frame,
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    if len(joined) != 2721 or joined["prediction"].isna().any():
        raise ValueError(f"{method}: retrieval/generation join is incomplete")
    joined["method"] = method
    manifest = []
    for kind, path, rows in (
        ("rag_retrieval", retrieval_path, retrieval),
        ("rag_generation", generation_path, generation),
    ):
        manifest.append(
            {
                "kind": kind,
                "cell": method,
                "path": str(path.relative_to(PROJECT_ROOT)),
                "rows": len(rows),
                "sha256": sha256_file(path),
            }
        )
    return joined, manifest


def _hit_miss_effects(
    frame: pd.DataFrame,
    *,
    metric: str,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    output = []
    for method, subset in frame.groupby("method", sort=True):
        if set(subset["hit"].astype(int)) != {0, 1}:
            raise ValueError(f"{method}: both hit and miss observations are required")
        means = subset.groupby("hit")[metric].mean()
        estimate = float(means.loc[1] - means.loc[0])
        labels = sorted(subset["cluster"].unique())
        cluster_stats = []
        for label in labels:
            current = subset.loc[subset["cluster"] == label]
            hit = current["hit"].to_numpy(int) == 1
            values = current[metric].to_numpy(float)
            cluster_stats.append(
                [
                    values[hit].sum(),
                    hit.sum(),
                    values[~hit].sum(),
                    (~hit).sum(),
                ]
            )
        stats = np.asarray(cluster_stats, dtype=float)
        rng = np.random.default_rng(seed)
        weights = rng.multinomial(
            len(labels),
            np.full(len(labels), 1.0 / len(labels)),
            size=iterations,
        ).astype(float)
        totals = np.einsum("bc,ck->bk", weights, stats)
        valid = (totals[:, 1] > 0) & (totals[:, 3] > 0)
        if not valid.all():
            totals = totals[valid]
        draws = totals[:, 0] / totals[:, 1] - totals[:, 2] / totals[:, 3]
        low, high = np.quantile(draws, [0.025, 0.975])
        output.append(
            {
                "method": method,
                "metric": metric,
                "samples": len(subset),
                "hit_samples": int(subset["hit"].sum()),
                "miss_samples": int((1 - subset["hit"]).sum()),
                "delta_hit_minus_miss": estimate,
                "ci_low": float(low),
                "ci_high": float(high),
            }
        )
    return pd.DataFrame(output)


def run_a1(
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    child_frames = []
    manifest = []
    for method, (retrieval_path, generation_path) in RAG_CHILD_SOURCES.items():
        frame, sources = load_rag_generation_rows(
            retrieval_path,
            generation_path,
            method=method,
        )
        child_frames.append(frame)
        manifest.extend(sources)
    child = pd.concat(child_frames, ignore_index=True)

    conditional_outputs = []
    within_outputs = []
    effects_outputs = []
    scopes = {
        "all": child,
        "factoid": child[child["question_type"] == "factoid"],
        "non_factoid": child[child["question_type"] == "non_factoid"],
    }
    for scope, scoped in scopes.items():
        for metric in ("character_f1",):
            conditional, discordant = summarize_hit_miss(
                scoped,
                metric=metric,
                iterations=iterations,
                seed=seed,
            )
            conditional.insert(0, "question_scope", scope)
            discordant.insert(0, "question_scope", scope)
            effects = _hit_miss_effects(
                scoped,
                metric=metric,
                iterations=iterations,
                seed=seed,
            )
            effects.insert(0, "question_scope", scope)
            conditional_outputs.append(conditional)
            within_outputs.append(discordant)
            effects_outputs.append(effects)
    factoid = scopes["factoid"]
    conditional, discordant = summarize_hit_miss(
        factoid,
        metric="exact_match_contain",
        iterations=iterations,
        seed=seed,
    )
    conditional.insert(0, "question_scope", "factoid")
    discordant.insert(0, "question_scope", "factoid")
    effects = _hit_miss_effects(
        factoid,
        metric="exact_match_contain",
        iterations=iterations,
        seed=seed,
    )
    effects.insert(0, "question_scope", "factoid")
    conditional_outputs.append(conditional)
    within_outputs.append(discordant)
    effects_outputs.append(effects)

    parent, sources = load_rag_generation_rows(
        RAG_PARENT_SOURCE[0],
        RAG_PARENT_SOURCE[1],
        method="hipporag2_parent",
        retrieval_id_field="id",
    )
    manifest.extend(sources)
    parent_outputs = []
    for scope, scoped in {
        "all": parent,
        "factoid": parent[parent["question_type"] == "factoid"],
        "non_factoid": parent[parent["question_type"] == "non_factoid"],
    }.items():
        metrics = ["character_f1"]
        if scope == "factoid":
            metrics.append("exact_match_contain")
        for metric in metrics:
            conditional, _ = summarize_hit_miss(
                scoped,
                metric=metric,
                iterations=iterations,
                seed=seed,
            )
            effects = _hit_miss_effects(
                scoped,
                metric=metric,
                iterations=iterations,
                seed=seed,
            )
            pivoted = conditional.pivot(
                index="method",
                columns="condition",
                values=["samples", "estimate"],
            )
            pivoted.columns = [f"{value}_{condition}" for value, condition in pivoted.columns]
            pivoted = pivoted.reset_index()
            merged = effects.merge(
                pivoted,
                on="method",
            )
            merged.insert(0, "question_scope", scope)
            parent_outputs.append(merged)

    conditionals = pd.concat(conditional_outputs, ignore_index=True)
    effects_frame = pd.concat(effects_outputs, ignore_index=True)
    within = pd.concat(within_outputs, ignore_index=True)
    conditionals = conditionals.merge(
        effects_frame[
            ["question_scope", "method", "metric", "delta_hit_minus_miss", "ci_low", "ci_high"]
        ].rename(columns={"ci_low": "delta_ci_low", "ci_high": "delta_ci_high"}),
        on=["question_scope", "method", "metric"],
        how="left",
        validate="many_to_one",
    )
    return conditionals, within, pd.concat(parent_outputs, ignore_index=True), manifest


def run_g0_g3_rewrite_quality(
    generation: pd.DataFrame,
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate paired G3-minus-G0 effects at aggregate and model level."""
    protocols = ("g0_released_gold", "g3_aligned_raw")
    metadata = (
        "model",
        "sample_id",
        "question_type",
        "dialogue_id",
        "turn",
        "turn_bucket",
        "cluster",
    )
    metrics = ("character_f1", "exact_match_contain")
    required = set(metadata + metrics + ("protocol",))
    missing = required - set(generation.columns)
    if missing:
        raise ValueError(f"G0/G3 analysis missing columns: {sorted(missing)}")
    selected = generation[generation["protocol"].isin(protocols)].copy()
    key_columns = ["model", "sample_id"]
    identifiers = {}
    for protocol in protocols:
        protocol_keys = selected.loc[selected["protocol"] == protocol, key_columns]
        identifiers[protocol] = [
            f"{model}::{sample_id}"
            for model, sample_id in protocol_keys.itertuples(index=False, name=None)
        ]
    try:
        validate_equal_sample_ids(identifiers)
    except ValueError as error:
        raise ValueError("G0/G3 sample ID mismatch") from error

    left = selected[selected["protocol"] == protocols[0]].loc[:, metadata + metrics]
    right = selected[selected["protocol"] == protocols[1]].loc[:, metadata + metrics]
    right = right.rename(
        columns={
            **{field: f"{field}__g3" for field in metadata if field not in key_columns},
            **{metric: f"{metric}__g3" for metric in metrics},
        }
    )
    paired = left.merge(right, on=key_columns, how="inner", validate="one_to_one")
    if len(paired) != len(left) or len(paired) != len(right):
        raise ValueError("G0/G3 sample ID mismatch")
    for field in metadata:
        if field in key_columns:
            continue
        right_field = f"{field}__g3"
        if not paired[field].equals(paired[right_field]):
            raise ValueError(f"G0/G3 metadata mismatch for {field}")

    def summarize(subset: pd.DataFrame, *, metric: str) -> dict[str, Any]:
        estimate, low, high, p_value = paired_cluster_bootstrap(
            subset[metric].to_numpy(float),
            subset[f"{metric}__g3"].to_numpy(float),
            subset["cluster"].to_numpy(object),
            iterations=iterations,
            seed=seed,
        )
        return {
            "samples": subset["sample_id"].nunique(),
            "model_observations": len(subset),
            "dialogue_clusters": subset["cluster"].nunique(),
            "g0_mean": float(subset[metric].mean()),
            "g3_mean": float(subset[f"{metric}__g3"].mean()),
            "delta_g3_minus_g0": estimate,
            "ci_low": low,
            "ci_high": high,
            "p_bootstrap": p_value,
        }

    aggregate_rows = []

    def append_aggregate(
        subset: pd.DataFrame,
        *,
        metric: str,
        scope: str,
        question_type: str,
        bucket: str,
    ) -> None:
        aggregate_rows.append(
            {
                "metric": metric,
                "scope": scope,
                "question_type": question_type,
                "turn_bucket": bucket,
                **summarize(subset, metric=metric),
            }
        )

    append_aggregate(
        paired,
        metric="character_f1",
        scope="overall",
        question_type="all",
        bucket="all",
    )
    for question_type, subset in paired.groupby("question_type", sort=True):
        append_aggregate(
            subset,
            metric="character_f1",
            scope="question_type",
            question_type=str(question_type),
            bucket="all",
        )
    for bucket, subset in paired.groupby("turn_bucket", sort=True):
        append_aggregate(
            subset,
            metric="character_f1",
            scope="turn_bucket",
            question_type="all",
            bucket=str(bucket),
        )
    for (question_type, bucket), subset in paired.groupby(
        ["question_type", "turn_bucket"], sort=True
    ):
        append_aggregate(
            subset,
            metric="character_f1",
            scope="question_type_turn_bucket",
            question_type=str(question_type),
            bucket=str(bucket),
        )

    factoid = paired[paired["question_type"] == "factoid"]
    append_aggregate(
        factoid,
        metric="exact_match_contain",
        scope="question_type",
        question_type="factoid",
        bucket="all",
    )
    for bucket, subset in factoid.groupby("turn_bucket", sort=True):
        append_aggregate(
            subset,
            metric="exact_match_contain",
            scope="question_type_turn_bucket",
            question_type="factoid",
            bucket=str(bucket),
        )

    model_rows = []
    for model, model_frame in paired.groupby("model", sort=True):
        model_rows.append(
            {
                "model": model,
                "metric": "character_f1",
                "question_type": "all",
                **summarize(model_frame, metric="character_f1"),
            }
        )
        for question_type, subset in model_frame.groupby("question_type", sort=True):
            model_rows.append(
                {
                    "model": model,
                    "metric": "character_f1",
                    "question_type": str(question_type),
                    **summarize(subset, metric="character_f1"),
                }
            )
        model_factoid = model_frame[model_frame["question_type"] == "factoid"]
        model_rows.append(
            {
                "model": model,
                "metric": "exact_match_contain",
                "question_type": "factoid",
                **summarize(model_factoid, metric="exact_match_contain"),
            }
        )
    return pd.DataFrame(aggregate_rows), pd.DataFrame(model_rows)
