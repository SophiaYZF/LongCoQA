#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_PYTHON="${VLLM_PYTHON:-${LONGCOQA_VLLM_PYTHON:-${LONGCOQA_PYTHON:-python}}}"

if ! command -v "${VLLM_PYTHON}" >/dev/null 2>&1 && [[ ! -x "${VLLM_PYTHON}" ]]; then
  echo "Missing vLLM Python: ${VLLM_PYTHON}" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=true
"${VLLM_PYTHON}" -m longcoqa.cli run-dense-retrieval
