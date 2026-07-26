#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${LONGCOQA_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
CONFIG="${LONGCOQA_P1_CPT_JUDGE_CONFIG:-${PROJECT_ROOT}/configs/experiments/phase_f_cpt.yaml}"
ENV_FILE="${LONGCOQA_ENV_FILE:-${PROJECT_ROOT}/.env}"
FORMAL_ROOT="${LONGCOQA_P1_CPT_FORMAL_ROOT:-${PROJECT_ROOT}/outputs/formal/p1_cpt}"
source "${PROJECT_ROOT}/scripts/common.sh"

longcoqa_require_executable "${PYTHON}" "Missing Python executable" || exit 1
[[ -e "${FORMAL_ROOT}/.complete" ]] || {
  echo "CPT P1 formal generation is incomplete: outputs/formal/p1_cpt/.complete" >&2
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

args=(
  judge-p1
  --config "${CONFIG}"
  --output-root "${PROJECT_ROOT}/outputs/judgments/qwen37/p1_cpt"
)
if [[ -n "${LONGCOQA_P1_CPT_JUDGE_LIMIT:-}" ]]; then
  args=(
    judge-p1
    --config "${CONFIG}"
    --limit "${LONGCOQA_P1_CPT_JUDGE_LIMIT}"
    --model-variant cpt
    --eval-variant raft
    --output-root "${PROJECT_ROOT}/runtime/p1_cpt_judge_preflight"
  )
fi

exec "${PYTHON}" -m longcoqa.cli "${args[@]}"
