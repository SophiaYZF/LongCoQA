#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing ${PYTHON}; create the environment with: python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
"${PYTHON}" -m pytest
"${PYTHON}" -m longcoqa.cli audit-data
"${PYTHON}" -m longcoqa.cli build-manifests
