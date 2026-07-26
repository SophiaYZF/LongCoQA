#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${LONGCOQA_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
CONFIG="${LONGCOQA_QWEN37_CONFIG:-${PROJECT_ROOT}/configs/experiments/phase_c.yaml}"
ENV_FILE="${LONGCOQA_ENV_FILE:-${PROJECT_ROOT}/.env}"
PREFLIGHT_ROOT="${PROJECT_ROOT}/runtime/qwen37_p0_preflight"
SAMPLES="${LONGCOQA_QWEN37_PREFLIGHT_SAMPLES:-2}"
protocols=(z0_current z1_teacher_forced g0_released_gold g3_aligned_raw)
suite_path="${PREFLIGHT_ROOT}/suite_smoke${SAMPLES}.json"
source "${PROJECT_ROOT}/scripts/common.sh"

longcoqa_require_executable "${PYTHON}" "Missing Python executable" || exit 1
[[ "${SAMPLES}" =~ ^[1-9][0-9]*$ ]] || {
  echo "LONGCOQA_QWEN37_PREFLIGHT_SAMPLES must be a positive integer" >&2
  exit 1
}
[[ "${SAMPLES}" -ge 2 ]] || {
  echo "LONGCOQA_QWEN37_PREFLIGHT_SAMPLES must be at least 2 for balanced coverage" >&2
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

mkdir -p "${PREFLIGHT_ROOT}"
if [[ -e "${PREFLIGHT_ROOT}/.complete" && -s "${suite_path}" ]]; then
  echo "Qwen3.7-max P0 preflight already complete: ${suite_path}"
  exit 0
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
args=(
  run-api-suite
  --config "${CONFIG}"
  --samples "${SAMPLES}"
  --output-root "${PREFLIGHT_ROOT}"
)
for protocol in "${protocols[@]}"; do
  args+=(--protocol "${protocol}")
done
"${PYTHON}" -m longcoqa.cli "${args[@]}"

"${PYTHON}" - "${suite_path}" "${SAMPLES}" <<'PY'
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
    raise SystemExit("invalid qwen37 P0 preflight suite")
if suite.get("samples_per_protocol") != expected_samples:
    raise SystemExit("unexpected qwen37 P0 preflight sample count")
for protocol in protocols:
    validation = suite["validation"][protocol]
    if validation.get("successful_samples") != expected_samples:
        raise SystemExit(f"{protocol}: incomplete qwen37 P0 preflight")
    if validation.get("failed_samples") != 0:
        raise SystemExit(f"{protocol}: failed qwen37 P0 preflight rows")
PY

touch "${PREFLIGHT_ROOT}/.complete"
echo "Qwen3.7-max P0 preflight complete: ${suite_path}"
