#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

export LONGCOQA_EXPECTED_GPUS=2
export LONGCOQA_SHARD_COUNT=2

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/common.sh"
longcoqa_setup_environment

: "${LONGCOQA_SINGLE_MODEL_CONFIG:?Set LONGCOQA_SINGLE_MODEL_CONFIG to a model YAML file}"
: "${LONGCOQA_SINGLE_MODEL_SLUG:?Set LONGCOQA_SINGLE_MODEL_SLUG to a filesystem-safe name}"
[[ "$LONGCOQA_SINGLE_MODEL_SLUG" =~ ^[a-z0-9][a-z0-9_]*$ ]] || {
  echo "Invalid LONGCOQA_SINGLE_MODEL_SLUG: $LONGCOQA_SINGLE_MODEL_SLUG" >&2
  exit 1
}

model_config=$LONGCOQA_SINGLE_MODEL_CONFIG
[[ "$model_config" = /* ]] || model_config=$LONGCOQA_PROJECT/$model_config
slug=$LONGCOQA_SINGLE_MODEL_SLUG
limit=${LONGCOQA_SINGLE_MODEL_PREFLIGHT_SAMPLES:-16}
[[ "$limit" =~ ^[1-9][0-9]*$ && $((limit % 2)) -eq 0 ]] || {
  echo "LONGCOQA_SINGLE_MODEL_PREFLIGHT_SAMPLES must be a positive even integer" >&2
  exit 1
}

preflight_root=${LONGCOQA_SINGLE_MODEL_PREFLIGHT_ROOT:-$LONGCOQA_PROJECT/runtime/single_model_preflight/$slug}
log_dir=$LONGCOQA_PROJECT/runtime/dlc/single_model/$slug/preflight
run_id=$(date -u +%Y%m%dT%H%M%SZ)_$$
mkdir -p "$log_dir" "$preflight_root"
log=$log_dir/main_$run_id.log
exec > >(tee -a "$log") 2>&1

monitor_pid=
lane_pids=()
lane_labels=()
cleanup() {
  local code=$?
  [[ -z "$monitor_pid" ]] || kill "$monitor_pid" 2>/dev/null || true
  if [[ "$code" -ne 0 ]]; then
    for pid in "${lane_pids[@]:-}"; do
      kill "$pid" 2>/dev/null || true
    done
    echo "[$(date -Is)] SINGLE_MODEL_PREFLIGHT_FAILED model=$slug exit=$code log=$log"
  fi
}
trap cleanup EXIT INT TERM

model_path=$("$LONGCOQA_VLLM_PYTHON" - "$model_config" <<'PY'
import sys
from pathlib import Path

import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(config["path"])
PY
)
required=(
  "$LONGCOQA_VLLM_PYTHON"
  "$LONGCOQA_CONFIG"
  "$model_config"
  "$model_path/config.json"
  "$LONGCOQA_PROJECT/data/LongCoQA_factoid.csv"
  "$LONGCOQA_PROJECT/data/LongCoQA_non_factoid.csv"
  "$LONGCOQA_PROJECT/artifacts/alignments/evidence_alignment.jsonl"
)
for path in "${required[@]}"; do
  [[ -s "$path" ]] || {
    echo "Missing required asset: $path" >&2
    exit 1
  }
done

if [[ -f "$preflight_root/.complete" ]]; then
  echo "[$(date -Is)] SINGLE_MODEL_PREFLIGHT_ALREADY_COMPLETE model=$slug output=$preflight_root"
  exit 0
fi

longcoqa_check_gpus
monitor_pid=$(longcoqa_start_monitor "$log_dir/gpu_$run_id.csv")
rm -f "$preflight_root/.complete"

for gpu in 0 1; do
  shard_root=$preflight_root/shards/shard_$(printf '%02d' "$gpu")
  CUDA_VISIBLE_DEVICES=$gpu "$LONGCOQA_VLLM_PYTHON" -m longcoqa.cli run-local-suite \
    --config "$LONGCOQA_CONFIG" \
    --model-config "$model_config" \
    --samples "$limit" \
    --output-root "$shard_root" \
    --cuda-visible-devices "$gpu" \
    --shard-index "$gpu" \
    --shard-count 2 \
    >"$log_dir/gpu${gpu}_$run_id.log" 2>&1 &
  lane_pids+=("$!")
  lane_labels+=("$slug/gpu$gpu")
done
longcoqa_wait_lanes
lane_pids=()
lane_labels=()

"$LONGCOQA_VLLM_PYTHON" - "$preflight_root" "$limit" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]) / "shards"
limit = int(sys.argv[2])
protocols = {"z0_current", "z1_teacher_forced", "g0_released_gold", "g3_aligned_raw"}
for shard in range(2):
    path = root / f"shard_{shard:02d}" / f"suite_smoke{limit}.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    assert set(value["reports"]) == protocols, path
    for report in value["reports"].values():
        assert report["successful_samples"] == limit // 2, path
        assert report["failed_samples"] == 0, path
print("SINGLE_MODEL_PREFLIGHT_OUTPUTS_OK")
PY

touch "$preflight_root/.complete"
echo "[$(date -Is)] SINGLE_MODEL_PREFLIGHT_OK model=$slug output=$preflight_root log=$log"
