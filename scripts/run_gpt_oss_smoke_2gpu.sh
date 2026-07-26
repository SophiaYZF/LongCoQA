#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

export LONGCOQA_EXPECTED_GPUS=2
export LONGCOQA_SHARD_COUNT=2

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/common.sh"
longcoqa_setup_environment
source "$script_dir/gpt_oss_common.sh"

preflight_root=$LONGCOQA_PROJECT/runtime/gpt_oss_preflight/smoke16
log_dir=$LONGCOQA_PROJECT/runtime/dlc/gpt_oss_120b/smoke16
run_id=$(date -u +%Y%m%dT%H%M%SZ)_$$
mkdir -p "$log_dir" "$preflight_root"
log=$log_dir/main_$run_id.log
exec > >(tee -a "$log") 2>&1

monitor_pid=
lane_pids=()
lane_labels=()
trap longcoqa_gpt_oss_cleanup EXIT INT TERM

longcoqa_check_gpt_oss_assets
if [[ -f "$preflight_root/.complete" ]]; then
  echo "[$(date -Is)] GPT_OSS_SMOKE16_ALREADY_COMPLETE output=$preflight_root"
  exit 0
fi
longcoqa_check_gpus
monitor_pid=$(longcoqa_start_monitor "$log_dir/gpu_$run_id.csv")
rm -f "$preflight_root/.complete"

for gpu in 0 1; do
  lane_root=$preflight_root/shards/shard_$(printf '%02d' "$gpu")
  CUDA_VISIBLE_DEVICES=$gpu "$LONGCOQA_VLLM_PYTHON" -m longcoqa.cli run-gpt-oss-suite \
    --config "$LONGCOQA_CONFIG" \
    --model-config "$LONGCOQA_GPT_OSS_CONFIG" \
    --samples 16 \
    --output-root "$lane_root" \
    --cuda-visible-devices "$gpu" \
    --shard-index "$gpu" \
    --shard-count 2 \
    >"$log_dir/gpu${gpu}_$run_id.log" 2>&1 &
  lane_pids+=("$!")
  lane_labels+=("gpt-oss-120b/smoke16/gpu$gpu")
done
longcoqa_wait_lanes
lane_pids=()
lane_labels=()

"$LONGCOQA_VLLM_PYTHON" - "$preflight_root" <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
protocols = ("z0_current", "z1_teacher_forced", "g0_released_gold", "g3_aligned_raw")
harmony = re.compile(r"<\|(?:start|end|channel|message|return|call)[^|]*\|>")
cjk = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")

for protocol in protocols:
    rows = []
    for shard in range(2):
        lane = root / "shards" / f"shard_{shard:02d}"
        suite_path = lane / "suite_smoke16.json"
        suite = json.loads(suite_path.read_text(encoding="utf-8"))
        assert set(suite["reports"]) == set(protocols), suite_path
        report = suite["reports"][protocol]
        assert report["successful_samples"] == 8, suite_path
        assert report["failed_samples"] == 0, suite_path
        result_path = lane / protocol / "smoke16" / "results.jsonl"
        rows.extend(json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines())
    assert len(rows) == 16, (protocol, len(rows))
    assert len({str(row["sample_id"]) for row in rows}) == 16, protocol
    for row in rows:
        prediction = str(row.get("prediction", "")).strip()
        assert row.get("status") == "ok", row.get("sample_id")
        assert row.get("response_format") == "harmony", row.get("sample_id")
        assert row.get("reasoning_effort") == "low", row.get("sample_id")
        assert prediction, f"empty final: {row.get('sample_id')}"
        assert not harmony.search(prediction), f"Harmony control leaked: {row.get('sample_id')}"
        assert cjk.search(prediction), f"CJK gate failed: {row.get('sample_id')}"
print("GPT_OSS_SMOKE16_OUTPUTS_OK")
PY

touch "$preflight_root/.complete"
echo "[$(date -Is)] GPT_OSS_SMOKE16_OK output=$preflight_root log=$log"
