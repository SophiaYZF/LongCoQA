#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${LONGCOQA_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
INPUT="${1:-${LONGCOQA_RAG_QUALITY_INPUT:-}}"
CELL="${2:-${LONGCOQA_RAG_QUALITY_CELL:-}}"
OUTPUT_ROOT="${PROJECT_ROOT}/runtime/paper_metrics_api_smoke/rag_quality"

[[ -x "${PYTHON}" ]] || { echo "Missing Python executable: ${PYTHON}" >&2; exit 1; }
[[ -n "${INPUT}" && -f "${INPUT}" && -n "${CELL}" ]] || {
  echo "Usage: $0 PATH_TO_ONE_CELL.jsonl MODEL/PROTOCOL_CELL" >&2
  exit 2
}

cd "${PROJECT_ROOT}"
exec "${PYTHON}" -m longcoqa.cli evaluate-rag-quality-smoke \
  --input "${INPUT}" \
  --provider "${LONGCOQA_RAG_QUALITY_PROVIDER:-qwen37}" \
  --cell "${CELL}" \
  --output-root "${OUTPUT_ROOT}" \
  --smoke \
  --limit "12"
