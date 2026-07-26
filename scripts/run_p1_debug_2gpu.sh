#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export LONGCOQA_EXPECTED_GPUS=2
export LONGCOQA_SHARD_COUNT=2
source "$script_dir/common.sh"
longcoqa_setup_environment

limit=${LONGCOQA_P1_DEBUG_LIMIT:-16}
config=${LONGCOQA_P1_CONFIG:-$LONGCOQA_PROJECT/configs/experiments/phase_f.yaml}
output_root=${LONGCOQA_P1_DEBUG_OUTPUT:-$LONGCOQA_PROJECT/runtime/p1_debug/smoke${limit}}
shard_root=$output_root/shards
log_dir=$LONGCOQA_PROJECT/runtime/p1_debug/logs
run_id=$(date -u +%Y%m%dT%H%M%SZ)_$$
mkdir -p "$log_dir"
log=$log_dir/p1_debug_$run_id.log
exec > >(tee -a "$log") 2>&1

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
    echo "[$(date -Is)] LONGCOQA_P1_DEBUG_FAILED exit=$code log=$log"
  fi
}
trap cleanup EXIT INT TERM

longcoqa_check_gpus
[[ -s "$config" ]] || {
  echo "Missing P1 config: $config" >&2
  exit 1
}
if [[ "${LONGCOQA_WARM_ADAPTERS:-1}" == "1" ]]; then
  for adapter in conversation grounded raft; do
    adapter_file="$LONGCOQA_PROJECT/outputs/training/qwen3_5_9b_${adapter}_lora/adapter_model.safetensors"
    [[ -s "$adapter_file" ]] || {
      echo "Missing P1 adapter: $adapter_file" >&2
      exit 1
    }
    dd if="$adapter_file" of=/dev/null bs=16M status=none
  done
fi
monitor_pid=$(longcoqa_start_monitor "$log_dir/p1_debug_${run_id}_gpu.csv")

for gpu in 0 1; do
  shard=$shard_root/shard_$(printf '%02d' "$gpu")
  CUDA_VISIBLE_DEVICES=$gpu "$LONGCOQA_VLLM_PYTHON" -m longcoqa.cli run-p1-local \
    --config "$config" \
    --output-root "$shard" \
    --limit "$limit" \
    --shard-index "$gpu" \
    --shard-count 2 \
    >"$log_dir/p1_debug_${run_id}_gpu${gpu}.log" 2>&1 &
  lane_pids+=("$!")
  lane_labels+=("p1-debug/gpu$gpu")
done
longcoqa_wait_lanes

"$LONGCOQA_VLLM_PYTHON" -m longcoqa.cli merge-p1-shards \
  --config "$config" \
  --shard-root "$shard_root" \
  --output-root "$output_root/merged" \
  --shard-count 2 \
  --limit "$limit"
touch "$output_root/.complete"
echo "[$(date -Is)] LONGCOQA_P1_DEBUG_OK output=$output_root/merged log=$log"
