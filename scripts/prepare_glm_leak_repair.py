#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

PROTOCOLS = (
    "z0_current",
    "z1_teacher_forced",
    "g0_released_gold",
    "g3_aligned_raw",
)
EXPECTED = {
    "z0_current": 5,
    "z1_teacher_forced": 8,
    "g0_released_gold": 1,
    "g3_aligned_raw": 1,
}
REASONING_TAG_RE = re.compile(r"</?(?:think|analysis)>", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def has_reasoning_tag(prediction: str) -> bool:
    return bool(REASONING_TAG_RE.search(prediction))


def checkpoint_name(sample_id: str) -> str:
    return f"{sample_id.replace(':', '__')}.json"


def read_leaks(model_root: Path) -> list[tuple[str, str]]:
    leaks: list[tuple[str, str]] = []
    for protocol in PROTOCOLS:
        results_path = model_root / protocol / "full/results.jsonl"
        if not results_path.is_file():
            raise FileNotFoundError(f"Missing merged results: {results_path}")
        for line_number, line in enumerate(
            results_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            row = json.loads(line)
            sample_id = str(row.get("sample_id", ""))
            if not sample_id:
                raise ValueError(f"Missing sample_id in {results_path}:{line_number}")
            if has_reasoning_tag(str(row.get("prediction", ""))):
                leaks.append((protocol, sample_id))
    return leaks


def build_move_plan(model_root: Path, run_dir: Path) -> list[dict[str, object]]:
    leaks = read_leaks(model_root)
    counts = Counter(protocol for protocol, _ in leaks)
    actual = {protocol: counts.get(protocol, 0) for protocol in PROTOCOLS}
    if actual != EXPECTED:
        raise ValueError(f"Unexpected leak counts: expected {EXPECTED}, found {actual}")

    rows: list[dict[str, object]] = []
    for protocol, sample_id in leaks:
        matches = sorted(
            (model_root / "shards").glob(
                f"shard_*/{protocol}/full/checkpoints/{checkpoint_name(sample_id)}"
            )
        )
        if len(matches) != 1:
            raise ValueError(
                f"Expected one checkpoint for {protocol}/{sample_id}, found {matches}"
            )
        source = matches[0]
        shard_name = source.parents[3].name
        shard_index = int(shard_name.removeprefix("shard_"))
        backup = (
            run_dir
            / "backup"
            / shard_name
            / protocol
            / "full/checkpoints"
            / source.name
        )
        if backup.exists():
            raise FileExistsError(f"Backup already exists: {backup}")
        rows.append(
            {
                "protocol": protocol,
                "sample_id": sample_id,
                "shard_index": shard_index,
                "source_checkpoint": str(source.resolve()),
                "backup_checkpoint": str(backup.resolve()),
            }
        )
    if len({(row["protocol"], row["sample_id"]) for row in rows}) != len(rows):
        raise ValueError("Duplicate protocol/sample pairs in repair plan")
    return rows


def execute_move_plan(rows: list[dict[str, object]], run_dir: Path) -> None:
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    moved: list[tuple[Path, Path]] = []
    try:
        for row in rows:
            source = Path(str(row["source_checkpoint"]))
            backup = Path(str(row["backup_checkpoint"]))
            backup.parent.mkdir(parents=True, exist_ok=True)
            source.replace(backup)
            moved.append((source, backup))

        counts = Counter(str(row["protocol"]) for row in rows)
        manifest = {
            "schema_version": "longcoqa-glm-leak-repair-v1",
            "counts": {protocol: counts.get(protocol, 0) for protocol in PROTOCOLS},
            "rows": rows,
        }
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        affected_shards = sorted({int(row["shard_index"]) for row in rows})
        (run_dir / "affected_shards.txt").write_text(
            "".join(f"{shard}\n" for shard in affected_shards), encoding="utf-8"
        )
    except Exception:
        for source, backup in reversed(moved):
            if backup.exists() and not source.exists():
                source.parent.mkdir(parents=True, exist_ok=True)
                backup.replace(source)
        raise


def main() -> None:
    args = parse_args()
    rows = build_move_plan(args.model_root.resolve(), args.run_dir.resolve())
    execute_move_plan(rows, args.run_dir.resolve())
    print(
        json.dumps(
            {
                "status": "prepared",
                "checkpoints": len(rows),
                "run_dir": str(args.run_dir.resolve()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
