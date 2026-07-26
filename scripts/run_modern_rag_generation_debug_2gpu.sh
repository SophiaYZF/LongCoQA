#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/common.sh"

runs_raw=${LONGCOQA_MODERN_RAG_DEBUG_RUNS-contextual,hipporag2}
if [[ -z "$runs_raw" ]]; then
  echo "LONGCOQA_MODERN_RAG_DEBUG_RUNS must not be empty" >&2
  exit 2
fi
declare -A selected=()
IFS=',' read -r -a requested <<<"$runs_raw"
for method in "${requested[@]}"; do
  case "$method" in
    contextual|hipporag2) ;;
    *)
      echo "Invalid LONGCOQA_MODERN_RAG_DEBUG_RUNS method: $method" >&2
      exit 2
      ;;
  esac
  if [[ -n "${selected[$method]+x}" ]]; then
    echo "Duplicate LONGCOQA_MODERN_RAG_DEBUG_RUNS method: $method" >&2
    exit 2
  fi
  selected[$method]=1
done

longcoqa_setup_environment
LONGCOQA_EXPECTED_GPUS=2
longcoqa_check_gpus
cd "$LONGCOQA_PROJECT"

model_config=${LONGCOQA_MODERN_RAG_DEBUG_MODEL_CONFIG:-$LONGCOQA_PROJECT/configs/models/qwen3_5_9b.yaml}
run_id=$(date -u +%Y%m%dT%H%M%SZ)_$$
debug_root=${LONGCOQA_MODERN_RAG_DEBUG_ROOT:-$LONGCOQA_PROJECT/runtime/modern_rag_debug/$run_id}
mkdir -p "$debug_root/logs" "$debug_root/rankings" "$debug_root/contexts" "$debug_root/outputs"
exec > >(tee -a "$debug_root/logs/main.log") 2>&1

monitor_pid=$(longcoqa_start_monitor "$debug_root/gpu.csv")
lane_pids=()
lane_labels=()
cleanup() {
  local code=$?
  kill "$monitor_pid" 2>/dev/null || true
  for pid in "${lane_pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  if [[ "$code" -ne 0 ]]; then
    echo "[$(date -Is)] MODERN_RAG_DEBUG_FAILED exit=$code root=$debug_root"
  fi
}
trap cleanup EXIT INT TERM

source_ranking() {
  case "$1" in
    contextual) echo "$LONGCOQA_PROJECT/reports/generated/rag/contextual_retrieval/contextual_dense_rewritten_dialogue.jsonl" ;;
    hipporag2)
      if [[ -s "$LONGCOQA_PROJECT/reports/generated/rag/hipporag2/smoke1/all_results.jsonl" ]]; then
        echo "$LONGCOQA_PROJECT/reports/generated/rag/hipporag2/smoke1/all_results.jsonl"
      else
        echo "$LONGCOQA_PROJECT/reports/generated/rag/hipporag2/smoke1/results.jsonl"
      fi
      ;;
  esac
}

ranking_level() {
  [[ "$1" == hipporag2 ]] && echo parent || echo child
}

sample_count() {
  [[ "$1" == hipporag2 ]] && echo 1 || echo 16
}

run_method() {
  local method=$1
  local source level expected ranking contexts output shard_root
  source=$(source_ranking "$method")
  level=$(ranking_level "$method")
  expected=$(sample_count "$method")
  ranking=$debug_root/rankings/$method.jsonl
  contexts=$debug_root/contexts/$method.jsonl
  output=$debug_root/outputs/$method
  shard_root=$output/shards
  [[ -s "$source" ]] || {
    echo "Missing debug ranking source: $source" >&2
    return 1
  }

  "$LONGCOQA_VLLM_PYTHON" - "$source" "$ranking" "$expected" <<'PY'
import json
import sys
from pathlib import Path

source, output, limit = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
rows = []
with source.open(encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            rows.append(json.loads(line))
            if len(rows) == limit:
                break
if len(rows) != limit:
    raise ValueError(f"Expected {limit} debug rankings, found {len(rows)}")
output.write_text(
    "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
    encoding="utf-8",
)
PY
  "$LONGCOQA_VLLM_PYTHON" -m longcoqa.cli validate-rag-ranking \
    --input "$ranking" --ranking-level "$level" --expected-samples "$expected"
  "$LONGCOQA_VLLM_PYTHON" -m longcoqa.cli build-rag-contexts \
    --retrieval "$ranking" --output "$contexts" --ranking-level "$level" \
    --top-k 10 --expected-samples "$expected"

  lane_pids=()
  lane_labels=()
  for gpu in 0 1; do
    shard=$shard_root/shard_$(printf '%02d' "$gpu")
    CUDA_VISIBLE_DEVICES=$gpu "$LONGCOQA_VLLM_PYTHON" \
      -m longcoqa.cli run-rag-local --model-config "$model_config" \
      --contexts "$contexts" --output-dir "$shard" \
      --shard-index "$gpu" --shard-count 2 \
      >"$debug_root/logs/${method}_gpu${gpu}.log" 2>&1 &
    lane_pids+=("$!")
    lane_labels+=("$method/gpu$gpu")
  done
  longcoqa_wait_lanes
  lane_pids=()
  lane_labels=()
  "$LONGCOQA_VLLM_PYTHON" -m longcoqa.cli merge-rag-shards \
    --shard-root "$shard_root" --output-dir "$output/merged" \
    --shard-count 2 --expected-samples "$expected"
  "$LONGCOQA_VLLM_PYTHON" -m longcoqa.cli validate-rag-generation \
    --input "$output/merged/results.jsonl" --expected-samples "$expected"
  echo "[$(date -Is)] MODERN_RAG_DEBUG_METHOD_OK method=$method examples=$expected"
}

for method in "${requested[@]}"; do
  run_method "$method"
done
echo "[$(date -Is)] MODERN_RAG_DEBUG_OK root=$debug_root"
