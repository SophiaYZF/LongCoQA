#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${LONGCOQA_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
INPUT="${1:-${LONGCOQA_JUDGE_SMOKE_INPUT:-}}"
CELL="${2:-${LONGCOQA_JUDGE_SMOKE_CELL:-}}"
OUTPUT_ROOT="${PROJECT_ROOT}/runtime/paper_metrics_api_smoke/judge"

[[ -x "${PYTHON}" ]] || { echo "Missing Python executable: ${PYTHON}" >&2; exit 1; }
[[ -n "${INPUT}" && -f "${INPUT}" && -n "${CELL}" ]] || {
  echo "Usage: $0 PATH_TO_ONE_CELL.jsonl MODEL/PROTOCOL_CELL" >&2
  exit 2
}

cd "${PROJECT_ROOT}"
exec "${PYTHON}" -m longcoqa.cli evaluate-judge-smoke \
  --input "${INPUT}" \
  --provider "${LONGCOQA_JUDGE_SMOKE_PROVIDER:-qwen37}" \
  --profile "${LONGCOQA_JUDGE_SMOKE_PROFILE:-qwen-structured-v1}" \
  --cell "${CELL}" \
  --output-root "${OUTPUT_ROOT}" \
  --smoke \
  --limit "12"
