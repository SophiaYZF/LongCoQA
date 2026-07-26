#!/usr/bin/env bash
set -Eeuo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LONGCOQA_PROJECT=${LONGCOQA_PROJECT:-$(cd "$script_dir/.." && pwd)}
LONGCOQA_PYTHON=${LONGCOQA_PYTHON:-python}
LONGCOQA_VLLM_PYTHON=${LONGCOQA_VLLM_PYTHON:-$LONGCOQA_PYTHON}
LONGCOQA_EXPECTED_GPUS=${LONGCOQA_EXPECTED_GPUS:-1}
LONGCOQA_SHARD_COUNT=${LONGCOQA_SHARD_COUNT:-$LONGCOQA_EXPECTED_GPUS}
LONGCOQA_CONFIG=${LONGCOQA_CONFIG:-$LONGCOQA_PROJECT/configs/experiments/phase_c.yaml}

longcoqa_require_executable() {
  local executable=$1
  local error_label=${2:-Missing executable}
  command -v "$executable" >/dev/null 2>&1 || [[ -x "$executable" ]] || {
    echo "$error_label: $executable" >&2
    return 1
  }
}

longcoqa_setup_environment() {
  export PYTHONPATH="$LONGCOQA_PROJECT${PYTHONPATH:+:$PYTHONPATH}"
  export TOKENIZERS_PARALLELISM=false
  export PYTHONUNBUFFERED=1
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
  ulimit -n 65536 || true
}

longcoqa_check_gpus() {
  command -v nvidia-smi >/dev/null || {
    echo "nvidia-smi is required for this GPU workflow" >&2
    return 1
  }
  local gpu_count
  gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
  [[ "$gpu_count" -eq "$LONGCOQA_EXPECTED_GPUS" ]] || {
    echo "Expected exactly $LONGCOQA_EXPECTED_GPUS visible GPUs, found $gpu_count" >&2
    return 1
  }
  local gpu
  for gpu in $(seq 0 $((LONGCOQA_EXPECTED_GPUS - 1))); do
    [[ "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^$/d' | wc -l)" -eq 0 ]] || {
      echo "GPU $gpu is already occupied" >&2
      return 1
    }
  done
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
}

longcoqa_start_monitor() {
  local output=$1
  mkdir -p "$(dirname "$output")"
  nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader,nounits --loop=5 >"$output" &
  echo "$!"
}

longcoqa_wait_lanes() {
  local failed=0
  local index
  for index in "${!lane_pids[@]}"; do
    if wait "${lane_pids[$index]}"; then
      echo "[$(date -Is)] lane complete ${lane_labels[$index]}"
    else
      echo "[$(date -Is)] lane FAILED ${lane_labels[$index]}" >&2
      failed=1
    fi
  done
  [[ "$failed" -eq 0 ]]
}
