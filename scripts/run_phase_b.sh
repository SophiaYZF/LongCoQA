#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing ${PYTHON}; run Phase A setup first." >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
"${PYTHON}" -m longcoqa.cli build-corpus
"${PYTHON}" -m longcoqa.cli align-evidence
"${PYTHON}" -m longcoqa.cli build-bm25
"${PYTHON}" -m longcoqa.cli evaluate-bm25
"${PYTHON}" -m pytest tests/test_corpus.py tests/test_alignment.py tests/test_bm25.py
