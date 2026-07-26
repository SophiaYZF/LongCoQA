#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
matrix_path="$project_root/configs/experiments/phase_g_evidence_sft.yaml"
data_root="$project_root/artifacts/training/evidence_sft"
status_path="$data_root/pretraining_status.json"
manifest_path="$data_root/manifest.json"
dataset_info_path="$data_root/dataset_info.json"
quality_path="$data_root/quality_report.json"
cpt_marker="$project_root/outputs/training/qwen3_5_9b_cpt_conservative_lora/.complete"

execute=0
if [[ ${1:-} == "--execute" ]]; then
  execute=1
  shift
fi
if (( $# != 0 )); then
  echo "usage: $0 [--execute]" >&2
  exit 2
fi

if (( execute == 1 )) && [[ ${LONGCOQA_EVIDENCE_SFT_HUMAN_APPROVED:-0} != 1 ]]; then
  echo "Refusing training: set LONGCOQA_EVIDENCE_SFT_HUMAN_APPROVED=1 after human audit." >&2
  exit 2
fi

python3 - "$project_root" "$matrix_path" "$data_root" "$status_path" "$manifest_path" "$dataset_info_path" "$quality_path" "$cpt_marker" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import yaml

(
    project_root_raw,
    matrix_raw,
    data_root_raw,
    status_raw,
    manifest_raw,
    dataset_info_raw,
    quality_raw,
    cpt_marker_raw,
) = sys.argv[1:]
project_root = Path(project_root_raw)
matrix_path = Path(matrix_raw)
data_root = Path(data_root_raw)
required = [
    matrix_path,
    Path(status_raw),
    Path(manifest_raw),
    Path(dataset_info_raw),
    Path(quality_raw),
    Path(cpt_marker_raw),
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit("preflight missing required files:\n" + "\n".join(missing))

status = json.loads(Path(status_raw).read_text(encoding="utf-8"))
if status.get("status") != "READY_FOR_TRAINING" or status.get("training_allowed") is not True:
    raise SystemExit(
        "training blocked by pretraining_status.json: "
        f"status={status.get('status')!r}, training_allowed={status.get('training_allowed')!r}"
    )

manifest = json.loads(Path(manifest_raw).read_text(encoding="utf-8"))
checksums = manifest["data"]["file_checksums"]
for variant, splits in checksums.items():
    for split, expected in splits.items():
        path = data_root / f"{variant}_{split}.jsonl"
        if not path.is_file():
            raise SystemExit(f"missing aligned dataset: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise SystemExit(f"sha256 mismatch for {path}: {digest} != {expected}")

matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
cells = matrix["training"]["cells"]
if len(cells) != 8 or len({cell["id"] for cell in cells}) != 8:
    raise SystemExit("matrix must contain eight unique cells")
outputs = [project_root / cell["output_dir"] for cell in cells]
if len(set(outputs)) != 8:
    raise SystemExit("matrix output directories are not unique")
for cell, output in zip(cells, outputs):
    config = project_root / cell["config"]
    if not config.is_file():
        raise SystemExit(f"missing config for {cell['id']}: {config}")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing non-empty output directory for {cell['id']}: {output}")
print("PREFLIGHT_OK: 8 cells; data sha256 checks passed; human audit approved")
for cell in cells:
    print(f"PLAN {cell['id']}: llamafactory-cli train {cell['config']}")
PY

if (( execute == 0 )); then
  echo "Dry preflight only. No training was started."
  exit 0
fi

if ! command -v llamafactory-cli >/dev/null 2>&1; then
  echo "llamafactory-cli is not available" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is not available" >&2
  exit 2
fi
gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
if (( gpu_count < 8 )); then
  echo "Eight GPUs are required; found $gpu_count" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export FORCE_TORCHRUN=1
while IFS= read -r relative_config; do
  echo "Starting: $relative_config"
  llamafactory-cli train "$project_root/$relative_config"
done < <(
  python3 - "$matrix_path" <<'PY'
import sys
from pathlib import Path
import yaml

matrix = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
for cell in matrix["training"]["cells"]:
    print(cell["config"])
PY
)
