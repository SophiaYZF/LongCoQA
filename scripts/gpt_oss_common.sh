#!/usr/bin/env bash

LONGCOQA_GPT_OSS_CONFIG=${LONGCOQA_GPT_OSS_CONFIG:-$LONGCOQA_PROJECT/configs/models/gpt_oss_120b.yaml}
LONGCOQA_MODEL_ROOT=${LONGCOQA_MODEL_ROOT:-$LONGCOQA_PROJECT/models}
LONGCOQA_GPT_OSS_MODEL=${LONGCOQA_GPT_OSS_MODEL:-$LONGCOQA_MODEL_ROOT/openai/gpt-oss-120b}

longcoqa_check_gpt_oss_assets() {
  command -v "$LONGCOQA_VLLM_PYTHON" >/dev/null 2>&1 || [[ -x "$LONGCOQA_VLLM_PYTHON" ]] || {
    echo "Missing required gpt-oss Python: $LONGCOQA_VLLM_PYTHON" >&2
    return 1
  }
  local required=(
    "$LONGCOQA_CONFIG"
    "$LONGCOQA_GPT_OSS_CONFIG"
    "$LONGCOQA_GPT_OSS_MODEL/config.json"
    "$LONGCOQA_GPT_OSS_MODEL/tokenizer.json"
    "$LONGCOQA_GPT_OSS_MODEL/model.safetensors.index.json"
    "$LONGCOQA_PROJECT/data/LongCoQA_factoid.csv"
    "$LONGCOQA_PROJECT/data/LongCoQA_non_factoid.csv"
    "$LONGCOQA_PROJECT/artifacts/alignments/evidence_alignment.jsonl"
  )
  local path
  for path in "${required[@]}"; do
    [[ -s "$path" ]] || {
      echo "Missing required gpt-oss asset: $path" >&2
      return 1
    }
  done
  "$LONGCOQA_VLLM_PYTHON" "$LONGCOQA_PROJECT/scripts/download_gpt_oss_120b.py" \
    --target "$LONGCOQA_GPT_OSS_MODEL" --validate-only
}

longcoqa_gpt_oss_shard_complete() {
  local suite_path=$1
  [[ -s "$suite_path" ]] || return 1
  "$LONGCOQA_VLLM_PYTHON" - "$suite_path" <<'PY'
import json
import sys
from pathlib import Path

expected = {
    "z0_current",
    "z1_teacher_forced",
    "g0_released_gold",
    "g3_aligned_raw",
}
try:
    suite = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    reports = suite["reports"]
    if suite.get("schema_version") != "longcoqa-gpt-oss-suite-v1":
        raise ValueError("wrong schema")
    if set(suite.get("protocols", [])) != expected or set(reports) != expected:
        raise ValueError("wrong protocols")
    for protocol in expected:
        report = reports[protocol]
        requested = int(report["requested_samples"])
        successful = int(report["successful_samples"])
        failed = int(report["failed_samples"])
        if report.get("protocol") != protocol:
            raise ValueError("protocol mismatch")
        if requested < 1 or failed != 0 or successful != requested:
            raise ValueError("incomplete protocol")
except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
    raise SystemExit(1)
PY
}

longcoqa_gpt_oss_cleanup() {
  local code=$?
  [[ -z "${monitor_pid:-}" ]] || kill "$monitor_pid" 2>/dev/null || true
  if [[ "$code" -ne 0 ]]; then
    local pid
    for pid in "${lane_pids[@]:-}"; do
      kill "$pid" 2>/dev/null || true
    done
    echo "[$(date -Is)] GPT_OSS_JOB_FAILED exit=$code log=${log:-unset}" >&2
  fi
}
