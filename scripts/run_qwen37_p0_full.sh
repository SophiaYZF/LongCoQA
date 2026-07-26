#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${LONGCOQA_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
CONFIG="${LONGCOQA_QWEN37_CONFIG:-${PROJECT_ROOT}/configs/experiments/phase_c.yaml}"
ENV_FILE="${LONGCOQA_ENV_FILE:-${PROJECT_ROOT}/.env}"
FORMAL_ROOT="${PROJECT_ROOT}/outputs/formal/models/qwen37"
REPORT_ROOT="${PROJECT_ROOT}/reports/generated/evaluation/automatic/qwen37"
EXPECTED_SAMPLES=3019
protocols=(z0_current z1_teacher_forced g0_released_gold g3_aligned_raw)
source "${PROJECT_ROOT}/scripts/common.sh"

longcoqa_require_executable "${PYTHON}" "Missing Python executable" || exit 1
[[ -e "${PROJECT_ROOT}/runtime/qwen37_p0_preflight/.complete" ]] || {
  echo "Qwen3.7-max P0 preflight is incomplete" >&2
  exit 1
}
[[ -f "${ENV_FILE}" ]] || {
  echo "Missing environment file: ${ENV_FILE}" >&2
  exit 1
}
for name in DASHSCOPE_API_KEYS DASHSCOPE_BASE_URL DASHSCOPE_MODEL_ID; do
  grep -Eq "^[[:space:]]*${name}[[:space:]]*=" "${ENV_FILE}" || {
    echo "Missing required environment variable name in .env: ${name}" >&2
    exit 1
  }
done

mkdir -p "${FORMAL_ROOT}" "${REPORT_ROOT}"
if [[ -e "${FORMAL_ROOT}/.complete" ]]; then
  echo "Qwen3.7-max P0 full already complete: ${FORMAL_ROOT}"
  exit 0
fi
rm -f "${FORMAL_ROOT}/.complete"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
args=(
  run-api-suite
  --config "${CONFIG}"
  --full
  --output-root "${FORMAL_ROOT}"
)
for protocol in "${protocols[@]}"; do
  args+=(--protocol "${protocol}")
done
"${PYTHON}" -m longcoqa.cli "${args[@]}"

suite_path="${FORMAL_ROOT}/suite_full.json"
"${PYTHON}" - "${suite_path}" "${EXPECTED_SAMPLES}" <<'PY'
import json
import sys
from pathlib import Path

suite = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected_samples = int(sys.argv[2])
protocols = [
    "z0_current",
    "z1_teacher_forced",
    "g0_released_gold",
    "g3_aligned_raw",
]
if suite.get("status") != "ok" or suite.get("protocols") != protocols:
    raise SystemExit("invalid qwen37 P0 formal suite")
if suite.get("samples_per_protocol") != expected_samples:
    raise SystemExit("unexpected qwen37 P0 formal sample count")
for protocol in protocols:
    validation = suite["validation"][protocol]
    if validation.get("result_rows") != expected_samples:
        raise SystemExit(f"{protocol}: expected 3019 formal rows")
    if validation.get("unique_sample_ids") != expected_samples:
        raise SystemExit(f"{protocol}: duplicate or missing formal sample IDs")
    if validation.get("successful_samples") != expected_samples:
        raise SystemExit(f"{protocol}: incomplete formal results")
    if validation.get("failed_samples") != 0:
        raise SystemExit(f"{protocol}: failed formal rows")
PY

for protocol in "${protocols[@]}"; do
  "${PYTHON}" -m longcoqa.cli evaluate-inference \
    --input "${FORMAL_ROOT}/${protocol}/full/results.jsonl" \
    --output "${REPORT_ROOT}/${protocol}.json" \
    --bootstrap-iterations 2000
done

touch "${FORMAL_ROOT}/.complete"
echo "Qwen3.7-max P0 full complete: ${FORMAL_ROOT}"
