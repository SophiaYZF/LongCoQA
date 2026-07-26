from __future__ import annotations

import csv
import io
import json
import os
import random
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from longcoqa.evaluation.paper_metric_profiles import MetricProfile
from longcoqa.evaluation.paper_metrics import (
    BatchMetricOutput,
    MetricScore,
    paper_lexical_scores,
)
from longcoqa.utils.hashing import stable_json_hash
from longcoqa.utils.io import atomic_write_json

MetricScorer = Callable[[str, str, MetricProfile], MetricScore]


def _turn_bucket(turn: int) -> str:
    if turn == 1:
        return "turn_1"
    if turn <= 4:
        return "turn_2_4"
    return "turn_5_plus"


def _checkpoint_name(sample_id: str) -> str:
    return f"{sample_id.replace(':', '__')}.json"


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _checkpoint_current(
    value: Mapping[str, Any] | None,
    *,
    input_hash: str,
    reference_hash: str,
    profile_hash: str,
) -> bool:
    return bool(
        value
        and value.get("status") == "ok"
        and value.get("input_hash") == input_hash
        and value.get("reference_hash") == reference_hash
        and value.get("profile_hash") == profile_hash
    )


def score_rows(
    rows: Sequence[Mapping[str, Any]],
    profile: MetricProfile,
    output_root: Path,
    *,
    scorer: MetricScorer = paper_lexical_scores,
) -> list[dict[str, Any]]:
    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Duplicate sample ID in paper metric inputs")
    failures = [row for row in rows if row.get("status") != "ok"]
    if failures:
        raise ValueError(f"Paper metrics received {len(failures)} non-ok inference rows")

    results: list[dict[str, Any]] = []
    checkpoint_root = output_root / "checkpoints"
    for row in rows:
        sample_id = str(row["sample_id"])
        prediction = str(row["prediction"])
        reference = str(row["reference"])
        input_hash = stable_json_hash(prediction)
        reference_hash = stable_json_hash(reference)
        checkpoint_path = checkpoint_root / _checkpoint_name(sample_id)
        existing = _load_json(checkpoint_path)
        if _checkpoint_current(
            existing,
            input_hash=input_hash,
            reference_hash=reference_hash,
            profile_hash=profile.profile_hash,
        ):
            results.append(dict(existing or {}))
            continue

        base: dict[str, Any] = {
            "schema_version": "longcoqa-paper-metric-score-v1",
            "sample_id": sample_id,
            "question_type": str(row["question_type"]),
            "dialogue_id": int(row["dialogue_id"]),
            "turn": int(row["turn"]),
            "input_hash": input_hash,
            "reference_hash": reference_hash,
            "profile_id": profile.profile_id,
            "profile_hash": profile.profile_hash,
            "compatibility": profile.compatibility.value,
            "scorer": profile.to_dict(),
            "truncated": False,
        }
        started = time.perf_counter()
        try:
            metric_score = scorer(prediction, reference, profile)
            result = {
                **base,
                "status": "ok",
                "scores": dict(metric_score.scores),
                "elapsed_seconds": time.perf_counter() - started,
            }
        except Exception as error:
            result = {
                **base,
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error)[:1000],
                "elapsed_seconds": time.perf_counter() - started,
            }
        atomic_write_json(checkpoint_path, result)
        results.append(result)
    return results


def score_batch_rows(
    rows: Sequence[Mapping[str, Any]],
    profile: MetricProfile,
    output_root: Path,
    adapter: Any,
) -> list[dict[str, Any]]:
    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Duplicate sample ID in paper metric inputs")
    failures = [row for row in rows if row.get("status") != "ok"]
    if failures:
        raise ValueError(f"Paper metrics received {len(failures)} non-ok inference rows")

    checkpoint_root = output_root / "checkpoints"
    by_sample: dict[str, dict[str, Any]] = {}
    pending: list[tuple[Mapping[str, Any], dict[str, Any], Path]] = []
    for row in rows:
        sample_id = str(row["sample_id"])
        prediction = str(row["prediction"])
        reference = str(row["reference"])
        input_hash = stable_json_hash(prediction)
        reference_hash = stable_json_hash(reference)
        checkpoint_path = checkpoint_root / _checkpoint_name(sample_id)
        existing = _load_json(checkpoint_path)
        if _checkpoint_current(
            existing,
            input_hash=input_hash,
            reference_hash=reference_hash,
            profile_hash=profile.profile_hash,
        ):
            by_sample[sample_id] = dict(existing or {})
            continue
        base = {
            "schema_version": "longcoqa-paper-metric-score-v1",
            "sample_id": sample_id,
            "question_type": str(row["question_type"]),
            "dialogue_id": int(row["dialogue_id"]),
            "turn": int(row["turn"]),
            "input_hash": input_hash,
            "reference_hash": reference_hash,
            "profile_id": profile.profile_id,
            "profile_hash": profile.profile_hash,
            "compatibility": profile.compatibility.value,
            "scorer": profile.to_dict(),
        }
        pending.append((row, base, checkpoint_path))

    if pending:
        started = time.perf_counter()
        try:
            output: BatchMetricOutput = adapter.score_batch(
                [str(row["prediction"]) for row, _, _ in pending],
                [str(row["reference"]) for row, _, _ in pending],
            )
            if len(output.scores) != len(pending) or len(output.truncated) != len(pending):
                raise ValueError("Learned adapter output length does not match pending rows")
            elapsed = (time.perf_counter() - started) / len(pending)
            for (_, base, checkpoint_path), scores, truncated in zip(
                pending, output.scores, output.truncated, strict=True
            ):
                result = {
                    **base,
                    "status": "ok",
                    "scores": dict(scores),
                    "truncated": bool(truncated),
                    "elapsed_seconds": elapsed,
                }
                atomic_write_json(checkpoint_path, result)
                by_sample[str(base["sample_id"])] = result
        except Exception as error:
            elapsed = (time.perf_counter() - started) / len(pending)
            for _, base, checkpoint_path in pending:
                result = {
                    **base,
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error)[:1000],
                    "truncated": False,
                    "elapsed_seconds": elapsed,
                }
                atomic_write_json(checkpoint_path, result)
                by_sample[str(base["sample_id"])] = result
    return [by_sample[sample_id] for sample_id in sample_ids]


def _metric_names(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    names = {tuple(sorted(dict(row["scores"]))) for row in rows}
    if len(names) != 1:
        raise ValueError("Successful metric rows must contain identical metric keys")
    return next(iter(names), ())


def _average(
    rows: Sequence[Mapping[str, Any]], metrics: Sequence[str]
) -> dict[str, float | None]:
    if not rows:
        return {metric: None for metric in metrics}
    return {
        metric: sum(float(dict(row["scores"])[metric]) for row in rows) / len(rows)
        for metric in metrics
    }


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * probability)]


def _cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    metrics: Sequence[str],
    *,
    iterations: int,
    seed: int,
) -> dict[str, dict[str, float] | None]:
    if not rows:
        return {metric: None for metric in metrics}
    if iterations < 1:
        raise ValueError("bootstrap_iterations must be positive")
    clusters: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        clusters[(str(row["question_type"]), int(row["dialogue_id"]))].append(row)
    keys = sorted(clusters)
    rng = random.Random(seed)
    draws = {metric: [] for metric in metrics}
    for _ in range(iterations):
        selected = [clusters[rng.choice(keys)] for _ in keys]
        count = sum(len(cluster) for cluster in selected)
        for metric in metrics:
            draws[metric].append(
                sum(
                    float(dict(row["scores"])[metric])
                    for cluster in selected
                    for row in cluster
                )
                / count
            )
    return {
        metric: {
            "low": _percentile(values, 0.025),
            "high": _percentile(values, 0.975),
        }
        for metric, values in draws.items()
    }


def _group_report(
    rows: Sequence[Mapping[str, Any]],
    metrics: Sequence[str],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "samples": len(rows),
        "metrics": _average(rows, metrics),
        "bootstrap_95_ci": _cluster_bootstrap(
            rows, metrics, iterations=iterations, seed=seed
        ),
    }


def aggregate_paper_scores(
    records: Sequence[Mapping[str, Any]],
    *,
    bootstrap_iterations: int = 2000,
    seed: int = 42,
    allow_failures: bool = False,
) -> dict[str, Any]:
    sample_ids = [str(row["sample_id"]) for row in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Duplicate sample ID in paper metric score records")
    successful = [row for row in records if row.get("status") == "ok"]
    failed = [row for row in records if row.get("status") == "error"]
    inapplicable = [row for row in records if row.get("status") == "inapplicable"]
    if failed and not allow_failures:
        raise ValueError(f"Paper metrics contain {len(failed)} failed metric rows")
    metrics = _metric_names(successful)
    profile_hashes = {str(row["profile_hash"]) for row in successful}
    if len(profile_hashes) > 1:
        raise ValueError("Paper metric aggregation cannot mix profile hashes")
    compatibilities = {str(row["compatibility"]) for row in successful}
    if len(compatibilities) > 1:
        raise ValueError("Paper metric aggregation cannot mix compatibility labels")
    compatibility = next(iter(compatibilities), None)

    report = {
        "schema_version": "longcoqa-paper-metric-report-v1",
        "requested_samples": len(records),
        "successful_samples": len(successful),
        "failed_samples": len(failed),
        "inapplicable_samples": len(inapplicable),
        "score_records_hash": stable_json_hash(list(records)),
        "profile_hash": next(iter(profile_hashes), None),
        "metric_metadata": {
            metric: {"compatibility": compatibility} for metric in metrics
        },
        "bootstrap": {
            "unit": "question_type:dialogue_id",
            "iterations": bootstrap_iterations,
            "seed": seed,
        },
        "overall": _group_report(
            successful,
            metrics,
            iterations=bootstrap_iterations,
            seed=seed,
        ),
        "by_question_type": {},
        "by_turn": {},
    }
    for question_type in ("factoid", "non_factoid"):
        subset = [row for row in successful if row["question_type"] == question_type]
        report["by_question_type"][question_type] = _group_report(
            subset, metrics, iterations=bootstrap_iterations, seed=seed
        )
    for bucket in ("turn_1", "turn_2_4", "turn_5_plus"):
        subset = [row for row in successful if _turn_bucket(int(row["turn"])) == bucket]
        report["by_turn"][bucket] = _group_report(
            subset, metrics, iterations=bootstrap_iterations, seed=seed
        )
    return report


def _paired_bootstrap(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    metrics: Sequence[str],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    clusters: dict[
        tuple[str, int], list[tuple[Mapping[str, Any], Mapping[str, Any]]]
    ] = defaultdict(list)
    for pair in pairs:
        left = pair[0]
        clusters[(str(left["question_type"]), int(left["dialogue_id"]))].append(pair)
    keys = sorted(clusters)
    rng = random.Random(seed)
    draws = {metric: [] for metric in metrics}
    observed = {
        metric: sum(
            float(dict(right["scores"])[metric]) - float(dict(left["scores"])[metric])
            for left, right in pairs
        )
        / len(pairs)
        for metric in metrics
    }
    for _ in range(iterations):
        selected = [clusters[rng.choice(keys)] for _ in keys]
        count = sum(len(cluster) for cluster in selected)
        for metric in metrics:
            draws[metric].append(
                sum(
                    float(dict(right["scores"])[metric])
                    - float(dict(left["scores"])[metric])
                    for cluster in selected
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
        for metric in metrics
    }


def compare_paper_scores(
    left_records: Sequence[Mapping[str, Any]],
    right_records: Sequence[Mapping[str, Any]],
    *,
    bootstrap_iterations: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    left = {
        str(row["sample_id"]): row for row in left_records if row.get("status") == "ok"
    }
    right = {
        str(row["sample_id"]): row for row in right_records if row.get("status") == "ok"
    }
    if set(left) != set(right):
        raise ValueError("Paired paper metric comparison requires identical IDs")
    if not left:
        raise ValueError("Paired paper metric comparison requires at least one row")
    pairs = [(left[sample_id], right[sample_id]) for sample_id in sorted(left)]
    metrics = _metric_names(list(left.values()))
    if metrics != _metric_names(list(right.values())):
        raise ValueError("Paired paper metric comparison requires identical metric keys")
    return {
        "schema_version": "longcoqa-paper-metric-comparison-v1",
        "samples": len(pairs),
        "bootstrap": {
            "unit": "question_type:dialogue_id",
            "iterations": bootstrap_iterations,
            "seed": seed,
        },
        "overall": _paired_bootstrap(
            pairs, metrics, iterations=bootstrap_iterations, seed=seed
        ),
        "by_question_type": {
            question_type: _paired_bootstrap(
                [pair for pair in pairs if pair[0]["question_type"] == question_type],
                metrics,
                iterations=bootstrap_iterations,
                seed=seed,
            )
            for question_type in ("factoid", "non_factoid")
            if any(pair[0]["question_type"] == question_type for pair in pairs)
        },
    }


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def export_paper_table(
    entries: Iterable[Mapping[str, Any]],
    *,
    metric_order: Sequence[str],
    csv_path: Path,
    markdown_path: Path,
    metadata_path: Path,
    precision: int = 4,
) -> None:
    columns = ["model", "protocol", *metric_order]
    table_rows: list[dict[str, str]] = []
    compatibility: dict[str, str | None] = {metric: None for metric in metric_order}
    for entry in entries:
        report = dict(entry["report"])
        metrics = dict(dict(report["overall"])["metrics"])
        metadata = dict(report.get("metric_metadata", {}))
        output = {"model": str(entry["model"]), "protocol": str(entry["protocol"])}
        for metric in metric_order:
            value = metrics.get(metric)
            output[metric] = "" if value is None else f"{float(value):.{precision}f}"
            detail = metadata.get(metric)
            if isinstance(detail, Mapping):
                compatibility[metric] = str(detail.get("compatibility"))
        table_rows.append(output)

    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=columns)
    writer.writeheader()
    writer.writerows(table_rows)
    _atomic_write_text(csv_path, csv_buffer.getvalue())

    markdown_lines = [
        f"| {' | '.join(columns)} |",
        f"| {' | '.join(['---'] * len(columns))} |",
    ]
    markdown_lines.extend(
        f"| {' | '.join(row[column] for column in columns)} |" for row in table_rows
    )
    _atomic_write_text(markdown_path, "\n".join(markdown_lines) + "\n")
    atomic_write_json(
        metadata_path,
        {
            "schema_version": "longcoqa-paper-table-metadata-v1",
            "metrics": list(metric_order),
            "compatibility": compatibility,
        },
    )
