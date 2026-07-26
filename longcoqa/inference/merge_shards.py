from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from longcoqa.inference.local_runner import summarize_local_results
from longcoqa.utils.io import atomic_write_json, atomic_write_jsonl, read_jsonl


def merge_local_shards(
    shard_root: Path,
    output_root: Path,
    *,
    protocols: tuple[str, ...],
    shard_count: int,
    expected_samples: int,
) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for protocol in protocols:
        results: list[dict[str, Any]] = []
        shard_summaries: list[dict[str, Any]] = []
        for shard_index in range(shard_count):
            run_dir = shard_root / f"shard_{shard_index:02d}" / protocol / "full"
            summary_path = run_dir / "summary.json"
            results_path = run_dir / "results.jsonl"
            if not summary_path.is_file() or not results_path.is_file():
                raise FileNotFoundError(f"Incomplete shard output: {run_dir}")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("failed_samples") != 0:
                raise ValueError(f"Shard has failed samples: {summary_path}")
            shard_summaries.append(summary)
            results.extend(read_jsonl(results_path))

        sample_ids = [str(item["sample_id"]) for item in results]
        if len(results) != expected_samples:
            raise ValueError(
                f"{protocol}: expected {expected_samples} rows, found {len(results)}"
            )
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError(f"{protocol}: duplicate sample IDs across shards")
        if any(item.get("status") != "ok" for item in results):
            raise ValueError(f"{protocol}: merge input contains failed rows")
        results.sort(
            key=lambda item: (
                str(item["question_type"]),
                int(item["dialogue_id"]),
                int(item["turn"]),
            )
        )
        target = output_root / protocol / "full"
        atomic_write_jsonl(target / "results.jsonl", results)
        report = summarize_local_results(
            results,
            model=dict(shard_summaries[0]["model"]),
            protocol=protocol,
            seed=int(shard_summaries[0]["seed"]),
        )
        report["merged_shards"] = shard_count
        report["expected_samples"] = expected_samples
        atomic_write_json(target / "summary.json", report)
        reports[protocol] = report

    suite = {
        "schema_version": "longcoqa-local-merged-suite-v1",
        "shard_root": str(shard_root),
        "shard_count": shard_count,
        "expected_samples_per_protocol": expected_samples,
        "protocols": list(protocols),
        "reports": reports,
    }
    atomic_write_json(output_root / "suite_full.json", suite)
    return suite
