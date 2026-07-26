#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

export LONGCOQA_EXPECTED_GPUS=2
export LONGCOQA_SHARD_COUNT=8

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/common.sh"

longcoqa_setup_environment
longcoqa_check_gpus
cd "$LONGCOQA_PROJECT"

model_root=$LONGCOQA_PROJECT/outputs/formal/models/glm4_7_flash
model_config=$LONGCOQA_PROJECT/configs/models/glm4_7_flash.yaml
report_root=$LONGCOQA_PROJECT/reports/generated/evaluation/automatic/glm4_7_flash
complete_marker=$LONGCOQA_PROJECT/outputs/formal/models/glm4_7_flash/.complete
run_id=$(date -u +%Y%m%dT%H%M%SZ)_$$
run_dir=$LONGCOQA_PROJECT/runtime/glm_leak_repair/$run_id
mkdir -p "$(dirname "$run_dir")"

required=(
  "$LONGCOQA_VLLM_PYTHON"
  "$model_config"
  "$LONGCOQA_CONFIG"
  "$LONGCOQA_PROJECT/artifacts/alignments/evidence_alignment.jsonl"
)
for path in "${required[@]}"; do
  [[ -s "$path" ]] || {
    echo "Missing required asset: $path" >&2
    exit 1
  }
done

"$LONGCOQA_VLLM_PYTHON" scripts/prepare_glm_leak_repair.py \
  --model-root "$model_root" --run-dir "$run_dir"
rm -f "$complete_marker"

monitor_pid=$(longcoqa_start_monitor "$run_dir/gpu.csv")
lane_pids=()
lane_labels=()
cleanup() {
  local code=$?
  kill "$monitor_pid" 2>/dev/null || true
  if [[ "$code" -ne 0 ]]; then
    for pid in "${lane_pids[@]:-}"; do
      kill "$pid" 2>/dev/null || true
    done
    echo "[$(date -Is)] GLM_LEAK_REPAIR_FAILED exit=$code run_dir=$run_dir" >&2
  fi
}
trap cleanup EXIT INT TERM

for gpu in 0 1; do
  (
    lane=0
    while read -r shard; do
      if (( lane % 2 == gpu )); then
        shard_label=$(printf '%02d' "$shard")
        echo "[$(date -Is)] RUN gpu=$gpu shard=$shard_label"
        CUDA_VISIBLE_DEVICES=$gpu "$LONGCOQA_VLLM_PYTHON" -m longcoqa.cli run-local-suite \
          --config "$LONGCOQA_CONFIG" \
          --model-config "$model_config" \
          --full \
          --output-root "$model_root/shards/shard_$shard_label" \
          --cuda-visible-devices "$gpu" \
          --shard-index "$shard" \
          --shard-count 8
      fi
      lane=$((lane + 1))
    done <"$run_dir/affected_shards.txt"
  ) >"$run_dir/gpu${gpu}.log" 2>&1 &
  lane_pids+=("$!")
  lane_labels+=("gpu$gpu")
done
longcoqa_wait_lanes
lane_pids=()
lane_labels=()

"$LONGCOQA_VLLM_PYTHON" -m longcoqa.cli merge-local-shards \
  --config "$LONGCOQA_CONFIG" \
  --shard-root "$model_root/shards" \
  --output-root "$model_root" \
  --shard-count 8 \
  --expected-samples 3019 >"$run_dir/merge.log" 2>&1

mkdir -p "$report_root"
for protocol in z0_current z1_teacher_forced g0_released_gold g3_aligned_raw; do
  "$LONGCOQA_VLLM_PYTHON" -m longcoqa.cli evaluate-inference \
    --input "$model_root/$protocol/full/results.jsonl" \
    --output "$report_root/$protocol.json" \
    --bootstrap-iterations 2000 >"$run_dir/evaluate_$protocol.log" 2>&1
done

"$LONGCOQA_VLLM_PYTHON" - "$model_root" "$report_root" "$run_dir" <<'PY'
import json
import re
import sys
from collections import Counter
from pathlib import Path

model_root, report_root, run_dir = map(Path, sys.argv[1:])
protocols = ("z0_current", "z1_teacher_forced", "g0_released_gold", "g3_aligned_raw")
expected_counts = {"z0_current": 5, "z1_teacher_forced": 8, "g0_released_gold": 1, "g3_aligned_raw": 1}
tag = re.compile(r"</?(?:think|analysis)>", re.IGNORECASE)
manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
rows = manifest["rows"]
assert len(rows) == 15, len(rows)
assert dict(Counter(row["protocol"] for row in rows)) == expected_counts
assert all(Path(row["backup_checkpoint"]).is_file() for row in rows)

validation = {"schema_version": "longcoqa-glm-leak-repair-validation-v1", "protocols": {}}
for protocol in protocols:
    results = [
        json.loads(line)
        for line in (model_root / protocol / "full/results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    ids = [str(row["sample_id"]) for row in results]
    assert len(results) == 3019, (protocol, len(results))
    assert len(set(ids)) == 3019, protocol
    assert all(row.get("status") == "ok" for row in results), protocol
    assert all(str(row.get("prediction", "")).strip() for row in results), protocol
    assert not any(tag.search(str(row.get("prediction", ""))) for row in results), protocol
    report = json.loads((report_root / f"{protocol}.json").read_text(encoding="utf-8"))
    assert report.get("requested_samples") == 3019, protocol
    assert report.get("successful_samples") == 3019, protocol
    assert report.get("failed_samples") == 0, protocol
    validation["protocols"][protocol] = {
        "rows": len(results),
        "unique_sample_ids": len(set(ids)),
        "failed_samples": 0,
        "reasoning_tag_leaks": 0,
        "metrics": report["overall"]["metrics"],
    }

(run_dir / "validation.json").write_text(
    json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

touch "$complete_marker"
echo "[$(date -Is)] GLM_LEAK_REPAIR_OK run_dir=$run_dir"
