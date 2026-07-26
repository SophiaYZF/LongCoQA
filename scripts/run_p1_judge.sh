#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${LONGCOQA_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
CONFIG="${LONGCOQA_P1_JUDGE_CONFIG:-${PROJECT_ROOT}/configs/experiments/phase_f.yaml}"
ENV_FILE="${LONGCOQA_ENV_FILE:-${PROJECT_ROOT}/.env}"
source "${PROJECT_ROOT}/scripts/common.sh"

longcoqa_require_executable "${PYTHON}" "Missing Python executable" || exit 1
[[ -s "${PROJECT_ROOT}/outputs/formal/p1/.complete" || -e "${PROJECT_ROOT}/outputs/formal/p1/.complete" ]] || {
  echo "P1 formal inference is incomplete: outputs/formal/p1/.complete is missing" >&2
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

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

args=(judge-p1 --config "${CONFIG}")
if [[ -n "${LONGCOQA_P1_JUDGE_LIMIT:-}" ]]; then
  model_variant="${LONGCOQA_P1_JUDGE_MODEL_VARIANT:-}"
  eval_variant="${LONGCOQA_P1_JUDGE_EVAL_VARIANT:-}"
  [[ -n "${model_variant}" && -n "${eval_variant}" ]] || {
    echo "A limited preflight requires both model and evaluation variants" >&2
    exit 1
  }
  args+=(
    --limit "${LONGCOQA_P1_JUDGE_LIMIT}"
    --model-variant "${model_variant}"
    --eval-variant "${eval_variant}"
    --output-root "${PROJECT_ROOT}/runtime/p1_judge_preflight"
  )
fi

exec "${PYTHON}" -m longcoqa.cli "${args[@]}"
