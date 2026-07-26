# Released data inventory

The release contains five benchmark files:

| File | Role | Released rows or samples |
|---|---|---:|
| `LongCoQA_factoid.csv` | Factoid conversational questions | 1,172 questions in 100 dialogues |
| `LongCoQA_non_factoid.csv` | Non-factoid conversational questions | 1,847 questions in 151 dialogues |
| `LongCoQA_evidence.csv` | Evidence rows cited by the question tables | 2,319 rows; IDs span `1..15515` |
| `splits/p0_full.json` | Full benchmark protocol | 3,019 sample records in 251 dialogues |
| `splits/p1_dialogue_split.json` | Frozen dialogue-level leakage-minimized split | 3,019 sample records; 251 dialogue assignments |

The two question tables contain 6,972 evidence references in total. The evidence table is a set of
cited rows, not a complete source-novel corpus and not a leakage-free retrieval index. Source novels
are not distributed.

Verify all five files against [`SHA256SUMS`](SHA256SUMS):

```bash
(cd data && shasum -a 256 -c SHA256SUMS)
```

See [`../DATASET_CARD.md`](../DATASET_CARD.md) for schemas, split definitions, intended use, and
limitations. The files listed above are licensed under the scope stated in
[`../DATA_LICENSE`](../DATA_LICENSE).
