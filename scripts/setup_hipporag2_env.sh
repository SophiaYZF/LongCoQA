#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_PYTHON="${VLLM_PYTHON:-${LONGCOQA_VLLM_PYTHON:-${LONGCOQA_PYTHON:-python}}}"
ENV="${HIPPO_ENV:-$ROOT/runtime/hipporag2-env}"
HIPPO_ROOT="${HIPPO_ROOT:-$ROOT/external/HippoRAG}"
HIPPO_REVISION=1e8f60981bf760b64003aa5bf5668126d0c106b3
HIPPO_PATCH="$ROOT/scripts/hipporag2_compat.patch"
HIPPO_STATE="$ROOT/scripts/hipporag2_compat.blobs"

longcoqa_hipporag2_state_is_exact() {
  local repository=$1 state_file=$2 actual_changes expected_changes
  local expected_blob relative_file actual_blob

  git -C "$repository" diff --cached --quiet -- || return 1
  [[ -z "$(git -C "$repository" diff --summary --)" ]] || return 1
  actual_changes=$(git -C "$repository" diff --name-only --) || return 1
  expected_changes=$(awk 'NF == 2 {print $2}' "$state_file") || return 1
  [[ "$actual_changes" == "$expected_changes" ]] || return 1

  while read -r expected_blob relative_file; do
    [[ "$expected_blob" =~ ^[0-9a-f]{40}$ && -n "$relative_file" ]] || return 1
    actual_blob=$(git -C "$repository" hash-object "$relative_file") || return 1
    [[ "$actual_blob" == "$expected_blob" ]] || return 1
  done < "$state_file"
}

longcoqa_apply_hipporag2_compat() {
  local repository=$1 expected_revision=$2 patch_file=$3 state_file=$4 actual_revision
  [[ -d "$repository/.git" ]] || {
    echo "Missing HippoRAG Git checkout: $repository" >&2
    return 2
  }
  [[ -f "$patch_file" ]] || {
    echo "Missing HippoRAG compatibility patch: $patch_file" >&2
    return 2
  }
  [[ -f "$state_file" ]] || {
    echo "Missing HippoRAG patched-state inventory: $state_file" >&2
    return 2
  }
  actual_revision=$(git -C "$repository" rev-parse HEAD) || return 2
  [[ "$actual_revision" == "$expected_revision" ]] || {
    echo "Expected HippoRAG revision $expected_revision, found $actual_revision" >&2
    return 2
  }

  if ! git -C "$repository" diff --cached --quiet --; then
    echo "Refusing HippoRAG tracked state: staged changes are present" >&2
    return 2
  fi
  if longcoqa_hipporag2_state_is_exact "$repository" "$state_file"; then
    echo "HippoRAG compatibility patch already applied"
    return 0
  fi
  if ! git -C "$repository" diff --quiet --; then
    echo "Refusing HippoRAG tracked state: changes do not exactly match the patch inventory" >&2
    return 2
  fi
  if ! git -C "$repository" apply --unidiff-zero --check "$patch_file"; then
    echo "Refusing HippoRAG tracked state: patch does not apply to the clean base" >&2
    return 2
  fi
  git -C "$repository" apply --unidiff-zero "$patch_file"
  if ! longcoqa_hipporag2_state_is_exact "$repository" "$state_file"; then
    echo "Refusing HippoRAG tracked state: post-apply inventory validation failed" >&2
    return 2
  fi
  echo "HippoRAG compatibility patch applied"
}

main() {
  longcoqa_apply_hipporag2_compat \
    "$HIPPO_ROOT" "$HIPPO_REVISION" "$HIPPO_PATCH" "$HIPPO_STATE"

  if [[ ! -x "$ENV/bin/python" ]]; then
    "$BASE_PYTHON" -m venv --system-site-packages "$ENV"
  fi
  "$ENV/bin/pip" install --no-deps \
    python-igraph==0.11.8 igraph==0.11.8 texttable \
    scipy==1.15.2 pyarrow==19.0.1 tenacity==8.5.0 nest_asyncio==1.6.0

  PYTHONPATH="$HIPPO_ROOT:$ROOT" "$ENV/bin/python" - <<'PY'
from src.hipporag.HippoRAG import HippoRAG
from src.hipporag.utils.misc_utils import text_processing
assert text_processing("哈利·波特")
print("HippoRAG 2 environment import check: OK")
PY
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
