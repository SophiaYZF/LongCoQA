#!/usr/bin/env bash
set -euo pipefail

VLLM_PYTHON="${VLLM_PYTHON:-${LONGCOQA_VLLM_PYTHON:-${LONGCOQA_PYTHON:-python}}}"
MODEL_CONFIG="${1:-configs/models/qwen3_5_9b.yaml}"

exec "${VLLM_PYTHON}" -m longcoqa.cli run-local-suite \
  --config configs/experiments/phase_c.yaml \
  --model-config "${MODEL_CONFIG}"
