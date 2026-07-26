from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from longcoqa.evaluation.text_metrics import normalize_answer, score_answer
from longcoqa.utils.hashing import stable_json_hash
from longcoqa.utils.io import atomic_write_json, read_jsonl

METRICS = (
    "exact_match",
    "exact_match_contain",
    "character_precision",
    "character_recall",
    "character_f1",
    "bleu_2",
    "rouge_l_f1",
)


def _ngrams(text: str, size: int) -> Counter[str]:
    return Counter(text[index : index + size] for index in range(len(text) - size + 1))


def bleu_2(prediction: str, reference: str) -> float:
    prediction = normalize_answer(prediction)
    reference = normalize_answer(reference)
    if not prediction or not reference:
        return float(prediction == reference)
    precisions = []
    for size in (1, 2):
        predicted = _ngrams(prediction, size)
        expected = _ngrams(reference, size)
        overlap = sum((predicted & expected).values())
        total = sum(predicted.values())
        # Add-one smoothing keeps short Chinese answers defined.
        precisions.append((overlap + 1) / (total + 1))
    brevity_penalty = (
        1.0 if len(prediction) > len(reference) else math.exp(1 - len(reference) / len(prediction))
    )
    return brevity_penalty * math.sqrt(precisions[0] * precisions[1])


def extended_score_answer(prediction: str, reference: str) -> dict[str, float]:
    basic = score_answer(prediction, reference)
    predicted = normalize_answer(prediction)
    expected = normalize_answer(reference)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    precision = overlap / len(predicted) if predicted else float(not expected)
    recall = overlap / len(expected) if expected else float(not predicted)
    return {
        "exact_match": float(basic["exact_match"]),
        "exact_match_contain": float(bool(expected) and expected in predicted),
        "character_precision": precision,
        "character_recall": recall,
        "character_f1": float(basic["character_f1"]),
        "bleu_2": bleu_2(prediction, reference),
        "rouge_l_f1": float(basic["rouge_l_f1"]),
    }


def _scored_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        scored.append(
            {
                "sample_id": str(row["sample_id"]),
                "question_type": str(row["question_type"]),
                "dialogue_id": int(row["dialogue_id"]),
                "turn": int(row["turn"]),
                "metrics": extended_score_answer(
                    str(row["prediction"]), str(row["reference"])
                ),
            }
        )
    return scored


def _average(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    return {
        metric: (
            sum(float(row["metrics"][metric]) for row in rows) / len(rows)
            if rows
            else None
        )
        for metric in METRICS
    }


def _turn_bucket(turn: int) -> str:
    if turn == 1:
        return "turn_1"
    if turn <= 4:
        return "turn_2_4"
    return "turn_5_plus"


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * probability)]


def _cluster_bootstrap(
    rows: list[dict[str, Any]], *, iterations: int, seed: int
) -> dict[str, dict[str, float]]:
    clusters: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        clusters[(row["question_type"], row["dialogue_id"])].append(row)
    keys = sorted(clusters)
    rng = random.Random(seed)
    draws = {metric: [] for metric in METRICS}
    for _ in range(iterations):
        sampled = [clusters[rng.choice(keys)] for _ in keys]
        count = sum(len(cluster) for cluster in sampled)
        for metric in METRICS:
            draws[metric].append(
                sum(float(row["metrics"][metric]) for cluster in sampled for row in cluster)
                / count
            )
    return {
        metric: {
            "low": _percentile(values, 0.025),
            "high": _percentile(values, 0.975),
        }
        for metric, values in draws.items()
    }


def evaluate_inference_results(
    input_path: Path,
    output_path: Path,
    *,
    bootstrap_iterations: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    raw = read_jsonl(input_path)
    rows = _scored_rows(raw)
    by_type = {
        question_type: {
            "samples": len(subset := [r for r in rows if r["question_type"] == question_type]),
            "metrics": _average(subset),
            "bootstrap_95_ci": _cluster_bootstrap(
                subset, iterations=bootstrap_iterations, seed=seed
            ),
        }
        for question_type in ("factoid", "non_factoid")
    }
    by_turn = {}
    for bucket in ("turn_1", "turn_2_4", "turn_5_plus"):
        subset = [row for row in rows if _turn_bucket(row["turn"]) == bucket]
        by_turn[bucket] = {"samples": len(subset), "metrics": _average(subset)}
    report = {
        "schema_version": "longcoqa-automatic-evaluation-v1",
        "input": str(input_path),
        "input_hash": stable_json_hash(
            [{"sample_id": row["sample_id"], "prediction": row["prediction"]} for row in raw]
        ),
        "requested_samples": len(raw),
        "successful_samples": len(rows),
        "failed_samples": len(raw) - len(rows),
        "bootstrap": {
            "unit": "question_type:dialogue_id",
            "iterations": bootstrap_iterations,
            "seed": seed,
        },
        "overall": {
            "samples": len(rows),
            "metrics": _average(rows),
            "bootstrap_95_ci": _cluster_bootstrap(
                rows, iterations=bootstrap_iterations, seed=seed
            ),
        },
        "by_question_type": by_type,
        "by_turn": by_turn,
    }
    atomic_write_json(output_path, report)
    return report


def _paired_bootstrap(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    clusters: dict[tuple[str, int], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for pair in pairs:
        row = pair[0]
        clusters[(row["question_type"], row["dialogue_id"])].append(pair)
    keys = sorted(clusters)
    rng = random.Random(seed)
    draws = {metric: [] for metric in METRICS}
    observed = {}
    for metric in METRICS:
        observed[metric] = sum(
            right["metrics"][metric] - left["metrics"][metric] for left, right in pairs
        ) / len(pairs)
    for _ in range(iterations):
        sampled = [clusters[rng.choice(keys)] for _ in keys]
        count = sum(len(cluster) for cluster in sampled)
        for metric in METRICS:
            draws[metric].append(
                sum(
                    right["metrics"][metric] - left["metrics"][metric]
                    for cluster in sampled
                    for left, right in cluster
                )
                / count
            )
    return {
        metric: {
            "difference_right_minus_left": observed[metric],
            "low": _percentile(draws[metric], 0.025),
            "high": _percentile(draws[metric], 0.975),
            "two_sided_p": min(
                1.0,
                2
                * min(
                    sum(value <= 0 for value in draws[metric]) / iterations,
                    sum(value >= 0 for value in draws[metric]) / iterations,
                ),
            ),
        }
        for metric in METRICS
    }


def compare_inference_results(
    left_path: Path,
    right_path: Path,
    output_path: Path,
    *,
    bootstrap_iterations: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    left = {row["sample_id"]: row for row in _scored_rows(read_jsonl(left_path))}
    right = {row["sample_id"]: row for row in _scored_rows(read_jsonl(right_path))}
    if set(left) != set(right):
        raise ValueError(
            f"Paired comparison requires identical IDs: left={len(left)} right={len(right)}"
        )
    pairs = [(left[sample_id], right[sample_id]) for sample_id in sorted(left)]
    report = {
        "schema_version": "longcoqa-paired-bootstrap-v1",
        "left": str(left_path),
        "right": str(right_path),
        "samples": len(pairs),
        "bootstrap": {
            "unit": "question_type:dialogue_id",
            "iterations": bootstrap_iterations,
            "seed": seed,
        },
        "overall": _paired_bootstrap(pairs, iterations=bootstrap_iterations, seed=seed),
        "by_question_type": {
            question_type: _paired_bootstrap(
                [pair for pair in pairs if pair[0]["question_type"] == question_type],
                iterations=bootstrap_iterations,
                seed=seed,
            )
            for question_type in ("factoid", "non_factoid")
        },
    }
    atomic_write_json(output_path, report)
    return report
