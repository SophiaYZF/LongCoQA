#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from longcoqa.config import load_provider_settings
from longcoqa.paths import PROJECT_ROOT, resolve_project_path
from longcoqa.providers import ProviderClient
from longcoqa.providers.cache import ResponseCache
from longcoqa.training.evidence_reconstruction import (
    build_aligned_datasets,
    build_human_audit_packet,
    build_quality_report,
    load_checkpoint_labels,
    load_reconstruction_rows,
    run_reconstruction,
)
from longcoqa.utils.io import atomic_write_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare aligned evidence-aware SFT data without starting training."
    )
    parser.add_argument("--execute", action="store_true", help="Run paid teacher calls.")
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Render datasets only from compatible successful checkpoints.",
    )
    parser.add_argument("--provider", default="qwen37")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=1400)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/training/evidence_sft",
    )
    parser.add_argument(
        "--train-input",
        type=Path,
        default=PROJECT_ROOT / "artifacts/training/raft_train.jsonl",
    )
    parser.add_argument(
        "--dev-input",
        type=Path,
        default=PROJECT_ROOT / "artifacts/training/raft_dev.jsonl",
    )
    return parser


def _print(value: dict[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _load_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    return load_reconstruction_rows(
        {
            "train": resolve_project_path(args.train_input),
            "dev": resolve_project_path(args.dev_input),
        }
    )


def _finalize(
    rows: list[dict[str, Any]],
    labels: dict[str, dict[str, Any]],
    *,
    output_dir: Path,
    input_paths: dict[str, Path],
    source_counts: dict[str, int],
) -> dict[str, Any]:
    data_report = build_aligned_datasets(rows, labels, output_dir=output_dir)
    quality = build_quality_report(
        rows,
        labels,
        data_report=data_report,
        input_paths=input_paths,
    )
    audit = build_human_audit_packet(
        rows,
        labels,
        output_dir=output_dir,
        count=250,
        seed=42,
    )
    atomic_write_json(output_dir / "quality_report.json", quality)
    status = {
        "schema_version": "longcoqa-evidence-sft-status-v1",
        "status": "AWAITING_HUMAN_AUDIT",
        "training_allowed": False,
        "reason": "The 250-row human audit has not been imported and validated.",
        "accepted_examples": data_report["accepted"],
        "rejected_examples": data_report["rejected"],
        "human_audit": audit,
    }
    atomic_write_json(output_dir / "pretraining_status.json", status)
    manifest = {
        "schema_version": "longcoqa-evidence-sft-preparation-v1",
        "mode": "finalized",
        "status": status["status"],
        "test_rows_read": 0,
        "source_counts": source_counts,
        "data": data_report,
        "quality_report": str(output_dir / "quality_report.json"),
        "pretraining_status": str(output_dir / "pretraining_status.json"),
        "human_audit": audit,
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.execute and args.finalize_only:
        raise SystemExit("--execute and --finalize-only are mutually exclusive")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    rows = _load_rows(args)
    split_counts = Counter(str(row["metadata"]["split"]) for row in rows)
    output_dir = resolve_project_path(args.output_dir)
    checkpoint_dir = output_dir / "reconstruction/checkpoints"
    input_paths = {
        "train": resolve_project_path(args.train_input),
        "dev": resolve_project_path(args.dev_input),
    }

    if not args.execute and not args.finalize_only:
        settings = load_provider_settings(args.provider)
        _print(
            {
                "mode": "dry_run",
                "provider": settings.safe_summary(),
                "splits": dict(sorted(split_counts.items())),
                "requested": len(rows),
                "test_rows_read": 0,
                "checkpoint_dir": str(checkpoint_dir),
                "output_dir": str(output_dir),
            }
        )
        return 0

    if args.finalize_only:
        labels = load_checkpoint_labels(rows, checkpoint_dir)
        manifest = _finalize(
            rows,
            labels,
            output_dir=output_dir,
            input_paths=input_paths,
            source_counts=dict(sorted(split_counts.items())),
        )
        _print(manifest)
        return 0

    selected_rows = rows[: args.limit] if args.limit is not None else rows
    settings = load_provider_settings(args.provider)
    workers = args.max_workers or settings.concurrency
    cache_path = output_dir / "reconstruction/teacher_cache.sqlite3"
    with ResponseCache(cache_path) as cache, ProviderClient(settings, cache=cache) as client:
        generation = run_reconstruction(
            selected_rows,
            client=client,
            checkpoint_dir=checkpoint_dir,
            max_workers=workers,
            max_attempts=args.max_attempts,
            max_tokens=args.max_tokens,
        )
    response: dict[str, Any] = {
        key: value for key, value in generation.items() if key != "labels"
    }
    response.update(
        {
            "mode": "execute",
            "provider": settings.safe_summary(),
            "limited": args.limit is not None,
            "test_rows_read": 0,
            "checkpoint_dir": str(checkpoint_dir),
        }
    )
    if args.limit is None:
        labels = load_checkpoint_labels(rows, checkpoint_dir)
        response["finalization"] = _finalize(
            rows,
            labels,
            output_dir=output_dir,
            input_paths=input_paths,
            source_counts=dict(sorted(split_counts.items())),
        )
    _print(response)
    return 1 if generation["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
