#!/usr/bin/env python3
"""Download and validate the pinned root gpt-oss-120b runtime checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ID = "openai/gpt-oss-120b"
REVISION = "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
DEFAULT_TARGET = Path("models/openai/gpt-oss-120b")
EXPECTED_SHARDS = 15
EXPECTED_WEIGHT_BYTES = 65_248_893_184
ALLOW_PATTERNS = (
    "*.json",
    "*.jinja",
    "*.txt",
    "*.model",
    "LICENSE",
    "README.md",
    "model-*.safetensors",
    "model.safetensors.index.json",
)
IGNORE_PATTERNS = ("original/*", "metal/*")


def validate_snapshot(target: Path) -> dict[str, Any]:
    """Fail closed unless the root snapshot has the exact expected weight set."""
    shards = sorted(target.glob("model-*.safetensors"))
    weight_bytes = sum(path.stat().st_size for path in shards)
    nested_weights = sorted(
        path for path in target.rglob("*.safetensors") if path.parent != target
    )
    if (
        len(shards) != EXPECTED_SHARDS
        or weight_bytes != EXPECTED_WEIGHT_BYTES
        or nested_weights
    ):
        raise RuntimeError(
            "Invalid checkpoint: "
            f"shards={len(shards)} bytes={weight_bytes} "
            f"nested_weights={[str(path) for path in nested_weights]}"
        )
    required = ("config.json", "tokenizer.json", "model.safetensors.index.json")
    missing = [name for name in required if not (target / name).is_file()]
    if missing:
        raise RuntimeError(f"Invalid checkpoint: missing runtime files {missing}")
    return {
        "repo_id": REPO_ID,
        "revision": REVISION,
        "target": str(target),
        "shards": len(shards),
        "weight_bytes": weight_bytes,
    }


def download_snapshot(target: Path) -> dict[str, Any]:
    """Download only the root Transformers/vLLM files at the pinned revision."""
    from huggingface_hub import snapshot_download

    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=REPO_ID,
        revision=REVISION,
        local_dir=target,
        allow_patterns=list(ALLOW_PATTERNS),
        ignore_patterns=list(IGNORE_PATTERNS),
    )
    manifest = validate_snapshot(target)
    manifest_path = target / "longcoqa_snapshot.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate an existing local snapshot without contacting Hugging Face.",
    )
    args = parser.parse_args()
    result = (
        validate_snapshot(args.target)
        if args.validate_only
        else download_snapshot(args.target)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
