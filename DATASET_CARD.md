# LongCoQA Dataset Card

## Dataset summary

LongCoQA is a Chinese conversational question-answering benchmark based on long novels. This
anonymous review release contains question-answer rows, cited evidence rows, and fixed evaluation
protocols. It does not contain the source novels.

| Subset | Questions | Dialogues |
|---|---:|---:|
| Factoid | 1,172 | 100 |
| Non-factoid | 1,847 | 151 |
| Total | 3,019 | 251 |

The evidence table contains 2,319 released rows and the question tables contain 6,972 evidence
references in total. Evidence identifiers occupy the range `1..15515`; only the 2,319 cited rows
are released, so identifiers are intentionally non-contiguous.

## File schemas

Column names and order are validated by `longcoqa/data/loader.py`:

- `data/LongCoQA_factoid.csv`: `id`, `num`, `question`, `answer`, `evidence1`,
  `evidence2`, `evidence3`, `evidence4`.
- `data/LongCoQA_non_factoid.csv`: `id`, `abstract`, `style`, `num`, `question`,
  `answer`, `evidence1`, `evidence2`, `evidence3`, `evidence4`.
- `data/LongCoQA_evidence.csv`: `id`, `value`.

In both question tables, `id` is a dialogue identifier and `num` is the one-indexed turn within
that dialogue. Evidence columns contain optional references to IDs in the released evidence table.
The known zero-evidence counts are 6 factoid questions and 292 non-factoid questions.

## Protocols and splits

- P0 (`data/splits/p0_full.json`) is the full 3,019-question benchmark across all 251 dialogues.
- P1 (`data/splits/p1_dialogue_split.json`) is the frozen dialogue-level, leakage-minimized split.
  Whole dialogues are assigned to train, development, or test; the manifest records 2,133/284/602
  questions and 176/24/51 dialogues in those partitions.

P1 minimizes, but does not claim to eliminate, evidence overlap between partitions. The committed
manifests are fixed protocols; their SHA-256 hashes are recorded in `data/SHA256SUMS`.

## Evidence and source-text boundary

Released evidence is a citation table, not a complete retrieval corpus. Building a retriever from
only these 2,319 cited rows exposes gold evidence and causes retrieval leakage. Retrieval research
must instead reconstruct a corpus from the underlying source texts and evaluate against the fixed
question protocols.

The source novels are not distributed. Users must obtain and supply source texts lawfully and are
responsible for complying with all applicable copyright and access terms. The release does not
grant rights to any source novel.

## Intended use

The dataset is intended for non-commercial research on long-context conversational question
answering, retrieval, evidence grounding, evaluation, and leakage-aware experimental design. Users
should report which fixed protocol they used, preserve dialogue-level partitions for P1, and verify
the released hashes before analysis.

## Limitations

- The benchmark is based on a limited set of novels and does not represent all genres, writing
  styles, languages, cultures, or conversational settings.
- Answers and evidence references may contain annotation error, ambiguity, incomplete support, or
  subjective judgments, especially for non-factoid questions.
- Zero-evidence rows cannot support evidence-based scoring in the same way as cited rows.
- Evidence identifiers are sparse and the released evidence is not suitable as a standalone
  retrieval corpus.
- Model-based metrics and provider-backed judges may introduce model, prompt, and service-version
  dependencies; those calls are not necessary to validate this release.

## License and warranty

The released CSV files and fixed split manifests are licensed for non-commercial use under the
Creative Commons Attribution-NonCommercial 4.0 International license. The exact file scope and
full legal code are in [DATA_LICENSE](DATA_LICENSE). Software is licensed separately under
Apache-2.0. The dataset is provided without warranties; see the respective license texts for the
applicable warranty disclaimers and limitations of liability.
