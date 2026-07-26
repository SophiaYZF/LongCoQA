# Third-party integrations

Third-party source code and Git metadata are not included in this repository. The revisions below
record the upstream checkouts used to test the retained integration points. The official HippoRAG
integration requires a separate upstream checkout. The ChainRAG-inspired adaptations are internal
and do not require ChainRAG to be installed; its checkout is optional for comparison.

## HippoRAG

- Upstream repository: <https://github.com/OSU-NLP-Group/HippoRAG>
- Tested Git revision: `1e8f60981bf760b64003aa5bf5668126d0c106b3`
- Upstream license: [MIT License](https://github.com/OSU-NLP-Group/HippoRAG/blob/1e8f60981bf760b64003aa5bf5668126d0c106b3/LICENSE)
- LongCoQA integration scripts: `scripts/prepare_hipporag2_data.py`,
  `scripts/setup_hipporag2_env.sh`, `scripts/run_hipporag2_official.py`, and
  `scripts/run_modern_rag_generation_debug_2gpu.sh`.

The upstream checkout is expected at `external/HippoRAG` when these integration scripts are used.
It is not installed or vendored by this repository. Reproduce the tested source state from the
repository root with:

```bash
git clone https://github.com/OSU-NLP-Group/HippoRAG.git external/HippoRAG
git -C external/HippoRAG checkout 1e8f60981bf760b64003aa5bf5668126d0c106b3
bash scripts/setup_hipporag2_env.sh
```

The setup script verifies that `HEAD` is exactly the tested revision before changing the checkout,
then safely applies `scripts/hipporag2_compat.patch`. Re-running setup recognizes an already
applied patch and makes no source change. Exact patched state requires no staged or mode changes,
the same four changed paths, and all four Git blob IDs listed in
`scripts/hipporag2_compat.blobs`. A partial, conflicting, extra-tracked-change, staged, or
wrong-revision checkout is rejected before environment creation or dependency installation. The
patch has SHA-256
`10427f5f86f445204cca944cc25c5d4324d874af8e8cf68655f8b71c2e177332` and contains only these
tested compatibility changes:

- lazy imports for optional embedding and LLM backends;
- a four-worker cap for both online OpenIE executor pools;
- CJK-preserving entity normalization.

`scripts/hipporag2_compat.patch` is a modification of HippoRAG source and is distributed under
HippoRAG's MIT License, not the repository's Apache-2.0 software license. The applicable full MIT
text is included in `scripts/hipporag2_compat.LICENSE`.

## ChainRAG

- Upstream repository: <https://github.com/nju-websoft/ChainRAG>
- Tested Git revision: `0f115b1fa03698b5d9cd2662a26dfdfde62d074a`
- Upstream license: [GNU General Public License v3.0](https://github.com/nju-websoft/ChainRAG/blob/0f115b1fa03698b5d9cd2662a26dfdfde62d074a/LICENSE)
- LongCoQA integration entry points: `python -m longcoqa.cli run-graph-progressive`,
  `python -m longcoqa.cli generate-progressive-plans`, and
  `python -m longcoqa.cli retrieve-progressive`. Their implementations are in
  `longcoqa/retrieval/graph_progressive.py` and `longcoqa/retrieval/progressive.py`.

These are controlled adaptations and do not import or redistribute ChainRAG source code. They run
without a ChainRAG checkout. Upstream ChainRAG may optionally be placed at `external/ChainRAG` for
comparison; it is not installed or vendored by this repository.
