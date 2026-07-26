#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export LONGCOQA_EXPECTED_GPUS=2
export LONGCOQA_SHARD_COUNT=2
source "$script_dir/common.sh"
longcoqa_setup_environment

limit=${LONGCOQA_P1_CPT_PREFLIGHT_LIMIT:-16}
config=${LONGCOQA_P1_CPT_CONFIG:-$LONGCOQA_PROJECT/configs/experiments/phase_f_cpt.yaml}
adapter_dir=$LONGCOQA_PROJECT/outputs/training/qwen3_5_9b_cpt_conservative_lora
output_root=${LONGCOQA_P1_CPT_PREFLIGHT_OUTPUT:-$LONGCOQA_PROJECT/runtime/p1_cpt_preflight/cpt_raw_conservative_smoke${limit}}
shard_root=$output_root/shards
log_dir=$LONGCOQA_PROJECT/runtime/p1_cpt_preflight/logs
run_id=$(date -u +%Y%m%dT%H%M%SZ)_$$
mkdir -p "$log_dir"
log=$log_dir/p1_cpt_preflight_$run_id.log
exec > >(tee -a "$log") 2>&1

if [[ -e "$output_root/.complete" ]]; then
  echo "[$(date -Is)] LONGCOQA_P1_CPT_PREFLIGHT_ALREADY_COMPLETE output=$output_root"
  exit 0
fi

monitor_pid=
lane_pids=()
lane_labels=()
cleanup() {
  local code=$?
  [[ -z "$monitor_pid" ]] || kill "$monitor_pid" 2>/dev/null || true
  for pid in "${lane_pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  if [[ "$code" -ne 0 ]]; then
    echo "[$(date -Is)] LONGCOQA_P1_CPT_PREFLIGHT_FAILED exit=$code log=$log"
  fi
}
trap cleanup EXIT INT TERM

longcoqa_check_gpus
for path in \
  "$config" \
  "$LONGCOQA_PROJECT/artifacts/training/conversation_test.jsonl" \
  "$LONGCOQA_PROJECT/artifacts/training/grounded_test.jsonl" \
  "$LONGCOQA_PROJECT/artifacts/training/raft_test.jsonl" \
  "$adapter_dir/.complete" \
  "$adapter_dir/adapter_config.json" \
  "$adapter_dir/adapter_model.safetensors"; do
  [[ -e "$path" ]] || {
    echo "Missing P1 CPT asset: $path" >&2
    exit 1
  }
done

if [[ "${LONGCOQA_WARM_ADAPTERS:-1}" == "1" ]]; then
  dd \
    if="$adapter_dir/adapter_model.safetensors" \
    of=/dev/null bs=16M status=none
fi
monitor_pid=$(longcoqa_start_monitor "$log_dir/p1_cpt_preflight_${run_id}_gpu.csv")

for gpu in 0 1; do
  shard=$shard_root/shard_$(printf '%02d' "$gpu")
  CUDA_VISIBLE_DEVICES=$gpu "$LONGCOQA_VLLM_PYTHON" -m longcoqa.cli run-p1-local \
    --config "$config" \
    --output-root "$shard" \
    --limit "$limit" \
    --shard-index "$gpu" \
    --shard-count 2 \
    >"$log_dir/p1_cpt_preflight_${run_id}_gpu${gpu}.log" 2>&1 &
  lane_pids+=("$!")
  lane_labels+=("p1-cpt-preflight/gpu$gpu")
done
longcoqa_wait_lanes

"$LONGCOQA_VLLM_PYTHON" -m longcoqa.cli merge-p1-shards \
  --config "$config" \
  --shard-root "$shard_root" \
  --output-root "$output_root/merged" \
  --shard-count 2 \
  --limit "$limit"

PYTHONPATH="$LONGCOQA_PROJECT" "$LONGCOQA_VLLM_PYTHON" - \
  "$output_root/merged" "$output_root/quality_summary.json" "$limit" <<'PY'
import json
import sys
from pathlib import Path

from longcoqa.training.cpt_factorial import summarize_dev_rows
from longcoqa.utils.io import atomic_write_json, read_jsonl

root = Path(sys.argv[1])
destination = Path(sys.argv[2])
expected = int(sys.argv[3])
cells = {}
for evaluation in ("conversation", "grounded", "raft"):
    path = root / "cpt" / evaluation / "full" / "results.jsonl"
    cells[evaluation] = summarize_dev_rows(read_jsonl(path), expected=expected)
report = {
    "schema_version": "longcoqa-p1-cpt-preflight-quality-v1",
    "candidate": "cpt_raw_conservative",
    "adapter_path": "outputs/training/qwen3_5_9b_cpt_conservative_lora",
    "expected_per_cell": expected,
    "cells": cells,
    "gate_passed": all(cell["gate_passed"] for cell in cells.values()),
}
atomic_write_json(destination, report)
print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
if not report["gate_passed"]:
    raise SystemExit("P1 CPT candidate failed quality gates")
PY

touch "$output_root/.complete"
echo "[$(date -Is)] LONGCOQA_P1_CPT_PREFLIGHT_OK output=$output_root/merged log=$log"
