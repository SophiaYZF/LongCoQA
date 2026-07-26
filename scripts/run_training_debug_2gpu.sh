#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project=${LONGCOQA_PROJECT:-$(cd "$script_dir/.." && pwd)}
workspace=${LONGCOQA_WORKSPACE:-$project}
factory=${LLAMAFACTORY_ROOT:-$workspace/LlamaFactory}
experiment=${1:-conversation}
gpu_devices=${CUDA_VISIBLE_DEVICES:-0,1}
gpu_count=$(awk -F',' '{print NF}' <<<"$gpu_devices")
config=$project/configs/training/qwen3_5_9b_${experiment}_lora.yaml
output=$project/runtime/training_debug/$experiment

[[ -s "$config" ]] || {
  echo "Unknown experiment '$experiment' or missing config: $config" >&2
  exit 2
}
cd "$project"
export CUDA_VISIBLE_DEVICES=$gpu_devices
export PATH="$factory/.venv/bin:$PATH"
export NPROC_PER_NODE=$gpu_count
export FORCE_TORCHRUN=1
export PYTHONPATH="$factory/src:$project${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ "$experiment" == "cpt" ]]; then
  eval_args=('eval_strategy="no"')
else
  eval_args=(
    eval_strategy=steps
    eval_steps=1000
    load_best_model_at_end=false
  )
fi

"$factory/.venv/bin/llamafactory-cli" train "$config" \
  max_samples=16 \
  max_steps=2 \
  num_train_epochs=1 \
  per_device_train_batch_size=1 \
  gradient_accumulation_steps=1 \
  per_device_eval_batch_size=1 \
  preprocessing_num_workers=1 \
  dataloader_num_workers=1 \
  "${eval_args[@]}" \
  save_strategy=steps \
  save_steps=1000 \
  logging_steps=1 \
  overwrite_output_dir=true \
  output_dir="$output"

[[ -s "$output/adapter_config.json" ]] || {
  echo "Debug training did not produce an adapter: $output" >&2
  exit 1
}
echo "LONGCOQA_TRAINING_DEBUG_OK experiment=$experiment output=$output"
