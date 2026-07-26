#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project=${LONGCOQA_PROJECT:-$(cd "$script_dir/.." && pwd)}
workspace=${LONGCOQA_WORKSPACE:-$project}
factory=${LLAMAFACTORY_ROOT:-$workspace/LlamaFactory}
python=${LONGCOQA_PYTHON:-python}
nvidia_smi=${NVIDIA_SMI_BIN:-nvidia-smi}
config=$project/configs/training/qwen3_5_9b_cpt_lora.yaml
run_id=$(date -u +%Y%m%dT%H%M%SZ)_$$
scan_root=${LONGCOQA_CPT_SCAN_ROOT:-$project/runtime/cpt_capacity_scan/$run_id}
max_steps=${LONGCOQA_CPT_SCAN_MAX_STEPS:-2}
max_samples=${LONGCOQA_CPT_SCAN_MAX_SAMPLES:-32}
trials_file=$scan_root/trials.jsonl
summary_file=$scan_root/summary.json
monitor_pid=

cleanup() {
  local code=$?
  [[ -z "$monitor_pid" ]] || kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
  exit "$code"
}
trap cleanup EXIT INT TERM

for asset in \
  "$factory/.venv/bin/llamafactory-cli" \
  "$config" \
  "$project/artifacts/training/dataset_info.json"; do
  [[ -s "$asset" ]] || {
    echo "Missing CPT scan asset: $asset" >&2
    exit 2
  }
done

mkdir -p "$scan_root"
[[ ! -e "$trials_file" && ! -e "$summary_file" ]] || {
  echo "Refusing to overwrite an existing scan: $scan_root" >&2
  exit 2
}
touch "$trials_file"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export NPROC_PER_NODE=2
export FORCE_TORCHRUN=1
export PATH="$factory/.venv/bin:$PATH"
export PYTHONPATH="$factory/src:$project${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

mapfile -t gpu_info < <(
  "$nvidia_smi" --query-gpu=index,memory.total --format=csv,noheader,nounits
)
[[ "${#gpu_info[@]}" -eq 2 ]] || {
  echo "CPT capacity scan requires exactly two visible GPUs, found ${#gpu_info[@]}" >&2
  exit 2
}
total_memory_mib=$(awk -F',' 'NR==1 {gsub(/ /, "", $2); print $2}' <<<"${gpu_info[0]}")

wait_for_zero_memory() {
  local attempt used
  for attempt in $(seq 1 30); do
    used=$("$nvidia_smi" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ' | paste -sd, -)
    [[ "$used" == "0,0" ]] && return 0
    sleep 1
  done
  echo "GPU memory did not return to zero: $used" >&2
  return 1
}

start_monitor() {
  local csv=$1
  printf 'timestamp,index,memory_used_mib,utilization_gpu,power_draw_w\n' >"$csv"
  (
    while true; do
      "$nvidia_smi" \
        --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw \
        --format=csv,noheader,nounits >>"$csv"
      sleep 1
    done
  ) &
  monitor_pid=$!
}

stop_monitor() {
  [[ -z "$monitor_pid" ]] || kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
  monitor_pid=
}

parse_gpu_csv() {
  "$python" - "$1" <<'PY'
import csv
import sys

peaks = {0: 0, 1: 0}
utils = {0: 0, 1: 0}
powers = {0: 0.0, 1: 0.0}
with open(sys.argv[1], encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        try:
            gpu = int(row["index"].strip())
            if gpu not in peaks:
                continue
            peaks[gpu] = max(peaks[gpu], int(float(row["memory_used_mib"].strip())))
            utils[gpu] = max(utils[gpu], int(float(row["utilization_gpu"].strip())))
            powers[gpu] = max(powers[gpu], float(row["power_draw_w"].strip()))
        except (KeyError, TypeError, ValueError):
            continue
both = int(all(peaks[gpu] > 0 for gpu in (0, 1)))
print(peaks[0], peaks[1], utils[0], utils[1], powers[0], powers[1], both)
PY
}

append_trial() {
  local trial_json=$1
  "$python" - "$trial_json" "$trials_file" <<'PY'
import json
import sys
from pathlib import Path

trial = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
with Path(sys.argv[2]).open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(trial, ensure_ascii=False, separators=(",", ":")) + "\n")
PY
}

mapfile -t specs < <(
  "$python" -m longcoqa.training.capacity specs | "$python" -c \
    'import json,sys; print("\n".join("{}:{}".format(x["batch_size"], x["gradient_accumulation_steps"]) for x in json.load(sys.stdin)))'
)

for spec in "${specs[@]}"; do
  batch=${spec%%:*}
  accumulation=${spec##*:}
  trial_dir=$scan_root/batch_$batch
  output=$trial_dir/output
  log=$trial_dir/train.log
  gpu_csv=$trial_dir/gpu.csv
  trial_json=$trial_dir/trial.json
  mkdir -p "$trial_dir"
  printf '%s\n' "$batch" >"$scan_root/current_batch"
  wait_for_zero_memory
  start_monitor "$gpu_csv"
  start_epoch=$(date +%s)
  set +e
  "$factory/.venv/bin/llamafactory-cli" train "$config" \
    max_samples="$max_samples" \
    max_steps="$max_steps" \
    num_train_epochs=1 \
    per_device_train_batch_size="$batch" \
    gradient_accumulation_steps="$accumulation" \
    preprocessing_num_workers=1 \
    dataloader_num_workers=1 \
    'eval_strategy="no"' \
    packing=true \
    save_strategy=steps \
    save_steps=1000 \
    logging_steps=1 \
    overwrite_output_dir=true \
    output_dir="$output" \
    >"$log" 2>&1
  code=$?
  set -e
  runtime_seconds=$(($(date +%s) - start_epoch))
  stop_monitor
  read -r peak0 peak1 util0 util1 power0 power1 both < <(parse_gpu_csv "$gpu_csv")

  adapter_ok=false
  if [[ -s "$output/adapter_config.json" && -s "$output/adapter_model.safetensors" ]]; then
    adapter_ok=true
  fi
  status=$(
    "$python" - "$code" "$adapter_ok" "$log" <<'PY'
import sys
from pathlib import Path
from longcoqa.training.capacity import classify_status

print(
    classify_status(
        exit_code=int(sys.argv[1]),
        adapter_ok=sys.argv[2] == "true",
        log_text=Path(sys.argv[3]).read_text(encoding="utf-8", errors="replace"),
    )
)
PY
  )
  adapter_python=False
  [[ "$adapter_ok" == true ]] && adapter_python=True
  train_loss=null
  trainer_runtime=null
  if [[ -s "$output/train_results.json" ]]; then
    read -r train_loss trainer_runtime < <(
      "$python" - "$output/train_results.json" <<'PY'
import json
import sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
print(data.get("train_loss", "null"), data.get("train_runtime", "null"))
PY
    )
  fi
  train_loss_python=$train_loss
  trainer_runtime_python=$trainer_runtime
  [[ "$train_loss" == null ]] && train_loss_python=None
  [[ "$trainer_runtime" == null ]] && trainer_runtime_python=None
  "$python" - "$trial_json" <<PY
import json
from pathlib import Path
trial = {
    "batch_size": $batch,
    "gradient_accumulation_steps": $accumulation,
    "status": "$status",
    "exit_code": $code,
    "peak_memory_mib": [$peak0, $peak1],
    "peak_utilization_percent": [$util0, $util1],
    "peak_power_w": [$power0, $power1],
    "both_gpus_observed": bool($both),
    "adapter_ok": $adapter_python,
    "wall_time_seconds": $runtime_seconds,
    "trainer_runtime_seconds": $trainer_runtime_python,
    "train_loss": $train_loss_python,
    "output_dir": "$output",
    "log": "$log",
    "gpu_csv": "$gpu_csv",
}
Path("$trial_json").write_text(json.dumps(trial, indent=2) + "\n", encoding="utf-8")
PY
  append_trial "$trial_json"
  echo "CPT_SCAN_TRIAL batch=$batch accumulation=$accumulation status=$status peaks=$peak0,$peak1"
  wait_for_zero_memory
  if "$python" -m longcoqa.training.capacity should-stop \
    --trial "$trial_json" --total-memory-mib "$total_memory_mib"; then
    break
  fi
done

"$python" -m longcoqa.training.capacity choose \
  --trials "$trials_file" \
  --output "$summary_file" \
  --total-memory-mib "$total_memory_mib" \
  --fallback-batch-one
rm -f "$scan_root/current_batch"
echo "LONGCOQA_CPT_CAPACITY_SCAN_OK summary=$summary_file"
