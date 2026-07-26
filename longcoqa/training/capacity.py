"""Deterministic trial planning and selection for CPT GPU-capacity scans."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrialSpec:
    batch_size: int
    gradient_accumulation_steps: int


def classify_status(*, exit_code: int, adapter_ok: bool, log_text: str) -> str:
    """Classify a trial without depending on a particular CUDA OOM spelling."""
    if exit_code == 0 and adapter_ok:
        return "success"
    if re.search(r"out of memory|OutOfMemoryError", log_text, flags=re.IGNORECASE):
        return "oom"
    return "failed"


def build_trial_specs() -> list[TrialSpec]:
    """Return the approved batch-2-first scan sequence."""
    return [TrialSpec(2, 2), TrialSpec(3, 1), TrialSpec(4, 1)]


def _peak_ratio(trial: dict[str, Any], total_memory_mib: int) -> float:
    peaks = trial.get("peak_memory_mib") or []
    return max(peaks, default=0) / total_memory_mib


def _is_valid(trial: dict[str, Any], total_memory_mib: int, high_ratio: float) -> bool:
    return (
        trial.get("status") in {"success", "baseline_success"}
        and trial.get("adapter_ok") is True
        and trial.get("both_gpus_observed") is True
        and 0 < _peak_ratio(trial, total_memory_mib) <= high_ratio
    )


def choose_trial(
    trials: list[dict[str, Any]],
    total_memory_mib: int,
    *,
    low_ratio: float = 0.85,
    high_ratio: float = 0.92,
    target_ratio: float = 0.90,
    fallback: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Choose the stable scan trial closest to target without crossing the ceiling."""
    del low_ratio  # The lower bound controls stopping, not final validity.
    valid = [trial for trial in trials if _is_valid(trial, total_memory_mib, high_ratio)]
    if valid:
        chosen = min(valid, key=lambda trial: abs(_peak_ratio(trial, total_memory_mib) - target_ratio))
        return {
            **chosen,
            "peak_ratio": _peak_ratio(chosen, total_memory_mib),
            "selection_source": "scan",
        }
    if fallback is not None and _is_valid(fallback, total_memory_mib, high_ratio):
        return {
            **fallback,
            "peak_ratio": _peak_ratio(fallback, total_memory_mib),
            "selection_source": "fallback",
        }
    return None


def should_stop(
    trial: dict[str, Any],
    total_memory_mib: int,
    *,
    low_ratio: float = 0.85,
    high_ratio: float = 0.92,
) -> bool:
    """Stop on a hard failure or once the measured peak reaches the target band/ceiling."""
    if trial.get("status") != "success":
        return True
    if trial.get("adapter_ok") is not True or trial.get("both_gpus_observed") is not True:
        return True
    ratio = _peak_ratio(trial, total_memory_mib)
    return ratio >= low_ratio or ratio > high_ratio


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("specs")

    choose = subparsers.add_parser("choose")
    choose.add_argument("--trials", type=Path, required=True)
    choose.add_argument("--output", type=Path, required=True)
    choose.add_argument("--total-memory-mib", type=int, required=True)
    choose.add_argument("--fallback-batch-one", action="store_true")

    stop = subparsers.add_parser("should-stop")
    stop.add_argument("--trial", type=Path, required=True)
    stop.add_argument("--total-memory-mib", type=int, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "specs":
        print(json.dumps([asdict(spec) for spec in build_trial_specs()]))
        return
    if args.command == "should-stop":
        trial = json.loads(args.trial.read_text(encoding="utf-8"))
        raise SystemExit(0 if should_stop(trial, args.total_memory_mib) else 1)

    trials = _load_jsonl(args.trials)
    fallback = None
    if args.fallback_batch_one:
        fallback = {
            "batch_size": 1,
            "gradient_accumulation_steps": 4,
            "status": "baseline_success",
            "peak_memory_mib": [55_921, 55_921],
            "adapter_ok": True,
            "both_gpus_observed": True,
        }
    selected = choose_trial(trials, args.total_memory_mib, fallback=fallback)
    summary = {
        "schema_version": "longcoqa-cpt-capacity-scan-v1",
        "total_memory_mib": args.total_memory_mib,
        "target_ratio": 0.90,
        "allowed_ratio": [0.85, 0.92],
        "trials": trials,
        "selected": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
