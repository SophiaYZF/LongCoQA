#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project=${LONGCOQA_PROJECT:-$(cd "$script_dir/.." && pwd)}
workspace=${LONGCOQA_WORKSPACE:-$project}
factory=${LLAMAFACTORY_ROOT:-$workspace/LlamaFactory}
python=${LONGCOQA_PYTHON:-python}
nvidia_smi=${NVIDIA_SMI_BIN:-nvidia-smi}
runtime=${LONGCOQA_CPT_FACTORIAL_RUNTIME:-$project/runtime/cpt_factorial_training}
dry_run=${LONGCOQA_CPT_FACTORIAL_DRY_RUN:-0}
source "$script_dir/common.sh"
run_id=$(date -u +%Y%m%dT%H%M%SZ)_$$
run_dir=$runtime/$run_id
mkdir -p "$run_dir"

base_output=$project/outputs/training/qwen3_5_9b_cpt_lora
base_adapter=$base_output/adapter_model.safetensors
base_marker=$base_output/.complete
raw_data=$project/artifacts/training/novels_cpt.jsonl
clean_data=$project/artifacts/training/novels_cpt_noindent.jsonl

longcoqa_require_executable "$python" "Missing Python executable" || exit 1
for asset in "$factory/.venv/bin/llamafactory-cli" "$base_adapter" "$raw_data"; do
  [[ -s "$asset" ]] || { echo "Missing CPT factorial asset: $asset" >&2; exit 2; }
done
[[ -e "$base_marker" ]] || { echo "Missing CPT factorial marker: $base_marker" >&2; exit 2; }

mapfile -t gpu_rows < <("$nvidia_smi" --query-gpu=index,name,memory.total --format=csv,noheader,nounits)
[[ "${#gpu_rows[@]}" -eq 2 ]] || {
  echo "CPT factorial retraining requires exactly two visible GPUs, found ${#gpu_rows[@]}" >&2
  exit 2
}
for gpu in 0 1; do
  occupied=$("$nvidia_smi" -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^$/d' | wc -l)
  [[ "$occupied" -eq 0 ]] || { echo "GPU $gpu is occupied" >&2; exit 2; }
done

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export NPROC_PER_NODE=2
export FORCE_TORCHRUN=1
export PATH="$factory/.venv/bin:$PATH"
export PYTHONPATH=${PYTHONPATH:-$project}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

"$python" -m longcoqa.training.cpt_factorial prepare-data \
  --input "$raw_data" --output "$clean_data" --expected-rows 293 \
  >"$run_dir/data_audit.json"

base_adapter_hash=$(sha256sum "$base_adapter" | awk '{print $1}')
base_marker_hash=$(sha256sum "$base_marker" | awk '{print $1}')

cells=(B C D)
names=(clean conservative clean_conservative)
datasets=(clean raw clean)
updates=(aggressive conservative conservative)
data_paths=("$clean_data" "$raw_data" "$clean_data")

for index in 0 1 2; do
  cell=${cells[$index]}
  name=${names[$index]}
  dataset_level=${datasets[$index]}
  update_level=${updates[$index]}
  config=$project/configs/training/qwen3_5_9b_cpt_${name}_lora.yaml
  output=$project/outputs/training/qwen3_5_9b_cpt_${name}_lora
  log=$run_dir/cell_${cell}_${name}.log
  gpu_csv=$run_dir/cell_${cell}_${name}_gpu.csv
  [[ -s "$config" ]] || { echo "Missing factorial config: $config" >&2; exit 2; }
  "$python" -m longcoqa.training.cpt_factorial validate-config \
    --config "$config" --dataset-level "$dataset_level" \
    --update-level "$update_level" --expected-output "$output" \
    >"$run_dir/cell_${cell}_${name}_config_audit.json"
  echo "CPT_FACTORIAL_CELL cell=$cell dataset=$dataset_level update=$update_level config=$config output=$output"

  if [[ "$dry_run" == 1 ]]; then
    continue
  fi
  if [[ -e "$output/.complete" ]]; then
    "$python" -m longcoqa.training.cpt_factorial validate-output \
      --output-dir "$output" --expected-steps 7
    echo "CPT_FACTORIAL_SKIP cell=$cell output=$output"
    continue
  fi
  if [[ -d "$output" && -n "$(find "$output" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing ambiguous incomplete output: $output" >&2
    exit 2
  fi

  "$nvidia_smi" \
    --query-gpu=timestamp,index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits --loop=5 >"$gpu_csv" &
  monitor_pid=$!
  cleanup_monitor() {
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
  }
  trap cleanup_monitor EXIT INT TERM
  set +e
  "$factory/.venv/bin/llamafactory-cli" train "$config" \
    per_device_train_batch_size=3 \
    gradient_accumulation_steps=4 \
    2>&1 | tee "$log"
  train_code=${PIPESTATUS[0]}
  set -e
  cleanup_monitor
  trap - EXIT INT TERM
  [[ "$train_code" -eq 0 ]] || {
    echo "CPT_FACTORIAL_TRAIN_FAILED cell=$cell exit=$train_code log=$log" >&2
    exit "$train_code"
  }

  "$python" -m longcoqa.training.cpt_factorial write-manifest \
    --output-dir "$output" --cell "$cell" \
    --dataset-level "$dataset_level" --update-level "$update_level" \
    --config "$config" --data "${data_paths[$index]}" \
    --base-model Qwen3.5-9B --global-batch 24 --log "$log" --expected-steps 7
  touch "$output/.complete"
  echo "CPT_FACTORIAL_TRAIN_OK cell=$cell output=$output"
done

[[ "$(sha256sum "$base_adapter" | awk '{print $1}')" == "$base_adapter_hash" ]] || {
  echo "Original A adapter changed during factorial retraining" >&2; exit 3;
}
[[ "$(sha256sum "$base_marker" | awk '{print $1}')" == "$base_marker_hash" ]] || {
  echo "Original A completion marker changed during factorial retraining" >&2; exit 3;
}

if [[ "$dry_run" == 1 ]]; then
  echo "CPT_FACTORIAL_DRY_RUN_OK run_dir=$run_dir"
else
  touch "$run_dir/.complete"
  echo "CPT_FACTORIAL_RETRAINING_OK run_dir=$run_dir"
fi
