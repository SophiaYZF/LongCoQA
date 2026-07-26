#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

export LONGCOQA_EXPECTED_GPUS=2
export LONGCOQA_SHARD_COUNT=2

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/common.sh"
longcoqa_setup_environment
source "$script_dir/gpt_oss_common.sh"

smoke_marker=$LONGCOQA_PROJECT/runtime/gpt_oss_preflight/smoke16/.complete
preflight_root=$LONGCOQA_PROJECT/runtime/gpt_oss_preflight/z0_z1_smoke200
log_dir=$LONGCOQA_PROJECT/runtime/dlc/gpt_oss_120b/smoke200
run_id=$(date -u +%Y%m%dT%H%M%SZ)_$$
mkdir -p "$log_dir" "$preflight_root"
log=$log_dir/main_$run_id.log
exec > >(tee -a "$log") 2>&1

monitor_pid=
lane_pids=()
lane_labels=()
trap longcoqa_gpt_oss_cleanup EXIT INT TERM

[[ -f "$smoke_marker" ]] || {
  echo "Run scripts/run_gpt_oss_smoke_2gpu.sh successfully first: $smoke_marker" >&2
  exit 1
}
longcoqa_check_gpt_oss_assets
if [[ -f "$preflight_root/.complete" ]]; then
  echo "[$(date -Is)] GPT_OSS_SMOKE200_ALREADY_COMPLETE output=$preflight_root"
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
    --samples 200 \
    --protocol z0_current \
    --protocol z1_teacher_forced \
    --output-root "$lane_root" \
    --cuda-visible-devices "$gpu" \
    --shard-index "$gpu" \
    --shard-count 2 \
    >"$log_dir/gpu${gpu}_$run_id.log" 2>&1 &
  lane_pids+=("$!")
  lane_labels+=("gpt-oss-120b/smoke200/gpu$gpu")
done
longcoqa_wait_lanes
lane_pids=()
lane_labels=()

"$LONGCOQA_VLLM_PYTHON" - "$preflight_root" <<'PY'
import json
import math
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
protocols = ("z0_current", "z1_teacher_forced")
harmony = re.compile(r"<\|(?:start|end|channel|message|return|call)[^|]*\|>")
cjk = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
diagnostics = {}

for protocol in protocols:
    rows = []
    for shard in range(2):
        lane = root / "shards" / f"shard_{shard:02d}"
        suite_path = lane / "suite_smoke200.json"
        suite = json.loads(suite_path.read_text(encoding="utf-8"))
        assert set(suite["reports"]) == set(protocols), suite_path
        report = suite["reports"][protocol]
        assert report["successful_samples"] == 100, suite_path
        assert report["failed_samples"] == 0, suite_path
        result_path = lane / protocol / "smoke200" / "results.jsonl"
        rows.extend(json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines())
    assert len(rows) == 200, (protocol, len(rows))
    assert len({str(row["sample_id"]) for row in rows}) == 200, protocol
    predictions = [str(row.get("prediction", "")).strip() for row in rows]
    assert all(row.get("status") == "ok" for row in rows), protocol
    assert all(row.get("response_format") == "harmony" for row in rows), protocol
    assert all(row.get("reasoning_effort") == "low" for row in rows), protocol
    assert all(predictions), f"{protocol}: empty final"
    assert not any(harmony.search(value) for value in predictions), f"{protocol}: Harmony control leaked"
    cjk_count = sum(bool(cjk.search(value)) for value in predictions)
    assert cjk_count >= math.ceil(0.95 * len(rows)), (protocol, cjk_count)
    lengths = sorted(len(value) for value in predictions)
    completion_tokens = sorted(int(row["usage"]["completion_tokens"]) for row in rows)
    diagnostics[protocol] = {
        "samples": len(rows),
        "cjk_predictions": cjk_count,
        "cjk_rate": cjk_count / len(rows),
        "prediction_chars": {
            "min": min(lengths),
            "median": statistics.median(lengths),
            "mean": statistics.fmean(lengths),
            "max": max(lengths),
        },
        "completion_tokens": {
            "min": min(completion_tokens),
            "median": statistics.median(completion_tokens),
            "mean": statistics.fmean(completion_tokens),
            "max": max(completion_tokens),
        },
        "finish_reasons": dict(sorted(Counter(str(row.get("finish_reason")) for row in rows).items())),
    }
(root / "diagnostics.json").write_text(
    json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print("GPT_OSS_SMOKE200_OUTPUTS_OK")
PY

touch "$preflight_root/.complete"
echo "[$(date -Is)] GPT_OSS_SMOKE200_OK output=$preflight_root log=$log"
