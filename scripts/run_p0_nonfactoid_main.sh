#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${LONGCOQA_PYTHON:-python}"
CONFIG="${LONGCOQA_P0_NONFACTOID_CONFIG:-${PROJECT_ROOT}/configs/experiments/p0_nonfactoid_main.yaml}"
ENV_FILE="${LONGCOQA_ENV_FILE:-${PROJECT_ROOT}/.env}"

command -v "${PYTHON}" >/dev/null 2>&1 || [[ -x "${PYTHON}" ]] || {
  echo "Missing Python executable: ${PYTHON}" >&2
  exit 1
}
[[ -f "${CONFIG}" ]] || { echo "Missing P0 Non-factoid config: ${CONFIG}" >&2; exit 1; }

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "${LONGCOQA_P0_NONFACTOID_EXECUTE:-0}" != "1" ]]; then
  exec "${PYTHON}" -m longcoqa.cli validate-p0-nonfactoid-main --config "${CONFIG}"
fi

[[ -f "${ENV_FILE}" ]] || { echo "Missing environment file: ${ENV_FILE}" >&2; exit 1; }
set -a
source "${ENV_FILE}"
set +a
export LONGCOQA_P0_NONFACTOID_EXECUTE=1

exec "${PYTHON}" -m longcoqa.cli evaluate-p0-nonfactoid-main \
  --config "${CONFIG}" \
  --execute \
  "$@"
