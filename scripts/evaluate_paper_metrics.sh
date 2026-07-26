#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${LONGCOQA_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
INPUT="${1:-${LONGCOQA_PAPER_METRICS_INPUT:-}}"
OUTPUT_ROOT="${LONGCOQA_PAPER_METRICS_OUTPUT_ROOT:-${PROJECT_ROOT}/reports/generated/evaluation/paper_metrics}"

[[ -x "${PYTHON}" ]] || { echo "Missing Python executable: ${PYTHON}" >&2; exit 1; }
[[ -n "${INPUT}" && -f "${INPUT}" ]] || {
  echo "Usage: $0 PATH_TO_INFERENCE_RESULTS.jsonl" >&2
  exit 2
}

args=(
  evaluate-paper-metrics
  --input "${INPUT}"
  --output-root "${OUTPUT_ROOT}"
  --bootstrap-iterations "${LONGCOQA_PAPER_METRICS_BOOTSTRAP_ITERATIONS:-2000}"
  --seed "${LONGCOQA_PAPER_METRICS_SEED:-42}"
  --device "${LONGCOQA_PAPER_METRICS_DEVICE:-cpu}"
  --batch-size "${LONGCOQA_PAPER_METRICS_BATCH_SIZE:-16}"
)
if [[ -n "${LONGCOQA_PAPER_METRICS_PROFILE:-}" ]]; then
  args+=(--profile "${LONGCOQA_PAPER_METRICS_PROFILE}")
fi

cd "${PROJECT_ROOT}"
exec "${PYTHON}" -m longcoqa.cli "${args[@]}"
