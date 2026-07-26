#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export LONGCOQA_EXPECTED_GPUS=2
export LONGCOQA_SHARD_COUNT=2
source "$script_dir/common.sh"
longcoqa_setup_environment

limit=${LONGCOQA_CPT_FACTORIAL_DEV_LIMIT:-16}
config=${LONGCOQA_CPT_FACTORIAL_DEV_CONFIG:-$LONGCOQA_PROJECT/configs/experiments/phase_f_cpt_factorial_dev.yaml}
run_id=$(date -u +%Y%m%dT%H%M%SZ)_$$
output_root=${LONGCOQA_CPT_FACTORIAL_DEV_OUTPUT:-$LONGCOQA_PROJECT/runtime/cpt_factorial_dev/$run_id}
shard_root=$output_root/shards
merged_root=$output_root/merged
log_dir=$output_root/logs
mkdir -p "$log_dir"

monitor_pid=
lane_pids=()
lane_labels=()
cleanup() {
  local code=$?
  [[ -z "$monitor_pid" ]] || kill "$monitor_pid" 2>/dev/null || true
  for pid in "${lane_pids[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  return "$code"
}
trap cleanup EXIT INT TERM

longcoqa_check_gpus
for path in \
  "$config" \
  "$LONGCOQA_PROJECT/artifacts/training/conversation_dev.jsonl" \
  "$LONGCOQA_PROJECT/artifacts/training/grounded_dev.jsonl" \
  "$LONGCOQA_PROJECT/artifacts/training/raft_dev.jsonl"; do
  [[ -s "$path" ]] || { echo "Missing CPT factorial dev asset: $path" >&2; exit 2; }
done
for marker in \
  "$LONGCOQA_PROJECT/outputs/training/qwen3_5_9b_cpt_lora/.complete" \
  "$LONGCOQA_PROJECT/outputs/training/qwen3_5_9b_cpt_clean_lora/.complete" \
  "$LONGCOQA_PROJECT/outputs/training/qwen3_5_9b_cpt_conservative_lora/.complete" \
  "$LONGCOQA_PROJECT/outputs/training/qwen3_5_9b_cpt_clean_conservative_lora/.complete"; do
  [[ -e "$marker" ]] || { echo "Missing CPT factorial dev marker: $marker" >&2; exit 2; }
done

monitor_pid=$(longcoqa_start_monitor "$log_dir/gpu.csv")
for gpu in 0 1; do
  shard=$shard_root/shard_$(printf '%02d' "$gpu")
  CUDA_VISIBLE_DEVICES=$gpu "$LONGCOQA_VLLM_PYTHON" -m longcoqa.cli run-p1-local \
    --config "$config" --output-root "$shard" --limit "$limit" \
    --shard-index "$gpu" --shard-count 2 \
    >"$log_dir/gpu${gpu}.log" 2>&1 &
  lane_pids+=("$!")
  lane_labels+=("cpt-factorial-dev/gpu$gpu")
done
longcoqa_wait_lanes

"$LONGCOQA_VLLM_PYTHON" -m longcoqa.cli merge-p1-shards \
  --config "$config" --shard-root "$shard_root" --output-root "$merged_root" \
  --shard-count 2 --limit "$limit"
"$LONGCOQA_VLLM_PYTHON" -m longcoqa.training.cpt_factorial summarize-dev \
  --root "$merged_root" --output "$output_root/summary.json" \
  --expected-per-cell "$limit"

"$LONGCOQA_VLLM_PYTHON" - "$output_root/summary.json" <<'PY'
import json
import sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
if not report["primary_gate_passed"]:
    raise SystemExit("Primary D adapter failed preregistered dev gates")
PY
touch "$output_root/.complete"
echo "CPT_FACTORIAL_DEV_OK output=$output_root"
