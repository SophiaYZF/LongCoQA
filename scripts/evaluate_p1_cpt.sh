#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${LONGCOQA_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
FORMAL_ROOT="${LONGCOQA_P1_CPT_FORMAL_ROOT:-${PROJECT_ROOT}/outputs/formal/p1_cpt}"
OUTPUT_ROOT="${LONGCOQA_P1_CPT_REPORT_ROOT:-${PROJECT_ROOT}/reports/generated/evaluation/p1_cpt}"

[[ -x "${PYTHON}" ]] || {
  echo "Missing Python executable: ${PYTHON}" >&2
  exit 1
}
[[ -e "${FORMAL_ROOT}/.complete" ]] || {
  echo "CPT P1 formal generation is incomplete: ${FORMAL_ROOT}/.complete" >&2
  exit 1
}

cd "${PROJECT_ROOT}"
exec "${PYTHON}" -m longcoqa.cli evaluate-p1-cpt \
  --formal-root "${FORMAL_ROOT}" \
  --base-root "${PROJECT_ROOT}/outputs/formal/p1/merged" \
  --output-root "${OUTPUT_ROOT}" \
  --bootstrap-iterations "${LONGCOQA_P1_CPT_BOOTSTRAP_ITERATIONS:-2000}" \
  --seed "${LONGCOQA_P1_CPT_SEED:-42}"
