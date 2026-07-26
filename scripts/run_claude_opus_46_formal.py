from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from longcoqa.config import load_provider_settings, read_env_file
from longcoqa.data import LongCoQADataset
from longcoqa.inference.claude_formal import (
    FORMAL_PROTOCOLS,
    finalize_formal_suite,
    preflight_key_pool,
    run_formal_protocol,
)
from longcoqa.paths import PROJECT_ROOT, resolve_project_path
from longcoqa.utils.io import atomic_write_json, read_yaml

ROOT = PROJECT_ROOT
DEFAULT_CONFIG = PROJECT_ROOT / "configs/experiments/claude_opus_4_6_formal.yaml"
DEFAULT_ENV_FILE = ROOT / ".env"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the explicitly authorized Claude Opus 4.6 formal suite."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.add_argument(
        "--stage",
        choices=("canary", "full", "finalize"),
        required=True,
    )
    parser.add_argument("--protocol", choices=FORMAL_PROTOCOLS)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--requests-per-minute", type=int)
    return parser.parse_args()


def _formal_settings(
    config: dict[str, Any],
    *,
    env_file: Path,
    concurrency: int,
    requests_per_minute: int,
):
    file_environment = read_env_file(env_file)
    environment = {**file_environment, **os.environ}
    base = load_provider_settings(str(config["provider"]), environment=environment)
    if not base.base_url:
        raise ValueError("Anthropic base URL is not configured")
    if not base.api_keys:
        raise ValueError("No Anthropic API keys are configured")
    return replace(
        base,
        name="anthropic_formal_claude_opus_4_6",
        model=str(config["model"]),
        concurrency=concurrency,
        requests_per_minute=requests_per_minute,
    )


def main() -> None:
    args = parse_args()
    config = read_yaml(args.config)
    primary = config["tiers"]["primary"]
    concurrency = int(args.concurrency or primary["concurrency"])
    requests_per_minute = int(
        args.requests_per_minute or primary["requests_per_minute"]
    )
    if concurrency < 1 or requests_per_minute < 1:
        raise ValueError("Concurrency and requests per minute must be positive")

    dataset = LongCoQADataset()
    output_root = resolve_project_path(config["output_root"])
    runtime_root = resolve_project_path(config["runtime_root"])
    cache_path = resolve_project_path(config["cache"])
    runtime_root.mkdir(parents=True, exist_ok=True)

    if args.stage == "finalize":
        summary = finalize_formal_suite(
            dataset,
            output_root=output_root,
            model=str(config["model"]),
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return

    settings, key_preflight = preflight_key_pool(
        _formal_settings(
            config,
            env_file=args.env_file,
            concurrency=concurrency,
            requests_per_minute=requests_per_minute,
        )
    )
    generation = config["generation"]
    common = {
        "alignment_path": resolve_project_path(config["alignment"]),
        "cache_path": cache_path,
        "seed": int(config["seed"]),
        "temperature": float(generation["temperature"]),
        "max_tokens_factoid": int(generation["max_tokens_factoid"]),
        "max_tokens_non_factoid": int(generation["max_tokens_non_factoid"]),
    }
    if args.stage == "canary":
        summary = run_formal_protocol(
            dataset,
            settings,
            protocol="z2_free_running",
            output_root=runtime_root / "canary",
            dialogues_per_type=int(config["canary"]["dialogues_per_type"]),
            **common,
        )
        canary = {
            "schema_version": "longcoqa-claude-opus-46-canary-v1",
            "status": "ok",
            "selected_tier": {
                "concurrency": concurrency,
                "requests_per_minute": requests_per_minute,
            },
            "provider": settings.safe_summary(),
            "key_preflight": key_preflight,
            "report": summary,
        }
        atomic_write_json(runtime_root / "canary_summary.json", canary)
        print(json.dumps(canary, ensure_ascii=False, indent=2, sort_keys=True))
        return

    if args.protocol is None:
        raise ValueError("--protocol is required for --stage full")
    if args.protocol not in tuple(str(value) for value in config["protocols"]):
        raise ValueError(f"Protocol is outside the authorized scope: {args.protocol}")
    summary = run_formal_protocol(
        dataset,
        settings,
        protocol=args.protocol,
        output_root=output_root,
        **common,
    )
    if int(summary["formal_validation"]["successful_samples"]) != int(
        config["expected_samples"]
    ):
        raise ValueError("Formal sample count does not match the locked configuration")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
