#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${LONGCOQA_PYTHON:-python}"
ENV_FILE="${LONGCOQA_ENV_FILE:-${PROJECT_ROOT}/.env}"
CONFIG="${LONGCOQA_CLAUDE_FORMAL_CONFIG:-${PROJECT_ROOT}/configs/experiments/claude_opus_4_6_formal.yaml}"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/formal/models/claude_opus_4_6"
RUNTIME_ROOT="${PROJECT_ROOT}/runtime/claude_opus_4_6_formal"
RETRY_DELAY="${LONGCOQA_CLAUDE_FORMAL_RETRY_DELAY_SECONDS:-60}"
CANARY_ONLY=0

if [[ "${1:-}" == "--canary-only" ]]; then
  CANARY_ONLY=1
  shift
fi
[[ "$#" -eq 0 ]] || { echo "Unexpected arguments: $*" >&2; exit 2; }
[[ "${LONGCOQA_CLAUDE_FORMAL_EXECUTE:-0}" == 1 ]] || {
  echo "Set LONGCOQA_CLAUDE_FORMAL_EXECUTE=1 for authorized paid execution" >&2
  exit 2
}
command -v "${PYTHON}" >/dev/null 2>&1 || [[ -x "${PYTHON}" ]] || {
  echo "Missing Python executable: ${PYTHON}" >&2
  exit 2
}
[[ -f "${ENV_FILE}" ]] || { echo "Missing environment file: ${ENV_FILE}" >&2; exit 2; }
[[ -f "${CONFIG}" ]] || { echo "Missing formal config: ${CONFIG}" >&2; exit 2; }

mkdir -p "${RUNTIME_ROOT}"
exec > >(tee -a "${RUNTIME_ROOT}/supervisor.log") 2>&1
printf '%s\n' "$$" > "${RUNTIME_ROOT}/supervisor.pid"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

run_cli() {
  "${PYTHON}" scripts/run_claude_opus_46_formal.py \
    --config "${CONFIG}" \
    --env-file "${ENV_FILE}" \
    --execute \
    "$@"
}

primary_args=(--concurrency 8 --requests-per-minute 60)
fallback_args=(--concurrency 4 --requests-per-minute 30)
selected_args=()

if run_cli --stage canary "${primary_args[@]}"; then
  selected_args=("${primary_args[@]}")
  echo "CLAUDE_FORMAL_TIER concurrency=8 requests_per_minute=60"
elif run_cli --stage canary "${fallback_args[@]}"; then
  selected_args=("${fallback_args[@]}")
  echo "CLAUDE_FORMAL_TIER concurrency=4 requests_per_minute=30"
else
  echo "CLAUDE_FORMAL_CANARY_FAILED primary=8/60 fallback=4/30" >&2
  exit 1
fi

if [[ "${CANARY_ONLY}" == 1 ]]; then
  echo "CLAUDE_FORMAL_CANARY_COMPLETE ${RUNTIME_ROOT}/canary_summary.json"
  exit 0
fi

run_until_complete() {
  local protocol="$1"
  local marker="${OUTPUT_ROOT}/${protocol}/full/.complete"
  while [[ ! -f "${marker}" ]]; do
    if ! run_cli \
      --stage full \
      --protocol "${protocol}" \
      "${selected_args[@]}"; then
      echo "CLAUDE_FORMAL_RETRY protocol=${protocol} delay=${RETRY_DELAY}s" >&2
    fi
    [[ -f "${marker}" ]] || sleep "${RETRY_DELAY}"
  done
  echo "CLAUDE_FORMAL_PROTOCOL_COMPLETE protocol=${protocol} marker=${marker}"
}

run_until_complete "z1_teacher_forced"
run_until_complete "g3_aligned_raw"
run_until_complete "z2_free_running"

run_cli --stage finalize "${selected_args[@]}"
[[ -f "${PROJECT_ROOT}/outputs/formal/models/claude_opus_4_6/.complete" ]] || {
  echo "Claude formal suite finalization did not create the model marker" >&2
  exit 1
}
echo "CLAUDE_FORMAL_SUITE_COMPLETE ${OUTPUT_ROOT}/.complete"
