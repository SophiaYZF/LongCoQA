# LongCoQA

Anonymous review release for a conversational question-answering benchmark based on long novels.

## Install

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
```

## Validate released data

This audit is offline and uses only the files committed under `data/`:

```bash
python -m longcoqa.cli audit-data --output runtime/data_audit.json
```

Checksums can be verified independently from the repository root:

```bash
(cd data && shasum -a 256 -c SHA256SUMS)
```

## Data

The repository includes 3,019 questions, 251 dialogues, and 2,319 released evidence rows.
The source novels are not distributed. See [DATASET_CARD.md](DATASET_CARD.md) and
[data/README.md](data/README.md) for schemas, split definitions, limitations, and provenance
boundaries.

## Workflows

List every available command without starting a provider call, model load, or training job:

```bash
python -m longcoqa.cli --help
```

The retained entry points are grouped as follows:

- Phase A (data audit and frozen split manifests): `bash scripts/run_phase_a.sh`, or inspect
  `python -m longcoqa.cli build-manifests --help`.
- Phase B (corpus reconstruction, evidence alignment, and retrieval baselines):
  `bash scripts/run_phase_b.sh`, after supplying lawful source novels and required model assets.
- Inference: inspect `python -m longcoqa.cli run-local-suite --help`,
  `python -m longcoqa.cli run-p1-local --help`, and the retained `scripts/run_*` launchers.
- Training-data preparation and training launchers: inspect
  `python -m longcoqa.cli build-training-data --help`,
  `python scripts/prepare_evidence_aware_sft.py --help`, and the retained training scripts.
- RAG and retrieval: inspect `python -m longcoqa.cli build-rag-contexts --help`,
  `python -m longcoqa.cli run-rag-local --help`, and the retrieval commands shown by the main
  CLI help. Third-party integration details are in [THIRD_PARTY.md](THIRD_PARTY.md).
- Evaluation: inspect `python -m longcoqa.cli evaluate-inference --help`,
  `python -m longcoqa.cli evaluate-paper-metrics --help`, and the retained evaluation scripts.

Commands that use remote providers require credentials and an explicit execution gate where
applicable. No remote provider call, model download, GPU inference, or training job is required
for the released-data audit above.

## Reproducibility boundary

Source-novel corpus reconstruction and full retrieval experiments require user-supplied lawful
source texts and separately installed model assets. The repository does not include source novels,
third-party retrieval repositories, trained checkpoints, generated artifacts, or experimental
results.

## License

Code is Apache-2.0. Released benchmark data and fixed splits are CC BY-NC 4.0. See
[LICENSE](LICENSE) and [DATA_LICENSE](DATA_LICENSE) for the full terms and exact scope. The
HippoRAG compatibility patch retains the upstream MIT terms documented in
[THIRD_PARTY.md](THIRD_PARTY.md).
