from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from longcoqa.data import LongCoQADataset
from longcoqa.data.corpus import discover_novel_paths
from longcoqa.inference.prompts import SYSTEM_PROMPT
from longcoqa.retrieval.bm25 import BM25Index
from longcoqa.utils.hashing import sha256_file, stable_json_hash
from longcoqa.utils.io import atomic_write_json, atomic_write_jsonl, read_jsonl

GROUNDED_SYSTEM = """你是《哈利·波特》系列小说的证据约束问答助手。
只依据用户提供的证据与对话历史回答。证据足够时先给出中文答案，再在末尾列出引用的证据编号；
证据不足时明确回答“提供的证据不足以回答该问题”。不得凭记忆补充证据中没有的信息。"""


def _dialogues(samples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[f"{sample['question_type']}:{sample['dialogue_id']}"].append(sample)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["turn"]))
    return dict(grouped)


def _history(
    sample: dict[str, Any], dialogues: dict[str, list[dict[str, Any]]]
) -> list[list[str]]:
    key = f"{sample['question_type']}:{sample['dialogue_id']}"
    return [
        [str(row["question"]), str(row["answer"])]
        for row in dialogues[key]
        if int(row["turn"]) < int(sample["turn"])
    ]


def _evidence_block(evidence: list[tuple[str, str]]) -> str:
    return "\n\n".join(f"[EVIDENCE {label}]\n{text}" for label, text in evidence)


def _grounded_output(answer: str, evidence_labels: list[str]) -> str:
    labels = ", ".join(f"EVIDENCE {label}" for label in evidence_labels)
    return f"{answer}\n\n引用：[{labels}]"


def _metadata(sample: dict[str, Any], split: str, variant: str) -> dict[str, Any]:
    return {
        "sample_id": sample["sample_id"],
        "split": split,
        "variant": variant,
        "question_type": sample["question_type"],
        "dialogue_id": sample["dialogue_id"],
        "turn": sample["turn"],
    }


def _has_annotated_evidence(sample: dict[str, Any]) -> bool:
    evidence_ids = sample.get("evidence_ids", [])
    return isinstance(evidence_ids, (list, tuple)) and len(evidence_ids) > 0


def _example(
    *,
    sample: dict[str, Any],
    split: str,
    variant: str,
    system: str,
    prompt: str,
    response: str,
    history: list[list[str]],
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = _metadata(sample, split, variant)
    metadata.update(extra_metadata or {})
    return {
        "instruction": prompt,
        "input": "",
        "output": response,
        "system": system,
        "history": history,
        "metadata": metadata,
    }


def _negative_candidates(
    sample_id: str,
    *,
    retrieval: dict[str, dict[str, Any]],
    children: dict[str, dict[str, Any]],
    gold_paragraph_ids: set[str],
    count: int,
) -> list[dict[str, Any]]:
    row = retrieval.get(sample_id)
    if row is None:
        return []
    negatives: list[dict[str, Any]] = []
    for ranked in row["ranked"]:
        child = children[str(ranked["child_id"])]
        if gold_paragraph_ids.intersection(map(str, child["paragraph_ids"])):
            continue
        negatives.append(child)
        if len(negatives) == count:
            break
    return negatives


def build_supervised_datasets(
    *,
    manifest_path: Path,
    alignment_path: Path,
    children_path: Path,
    retrieval_path: Path,
    bm25_index_path: Path,
    output_dir: Path,
    seed: int = 42,
    hard_negatives: int = 4,
    raft_gold_probability: float = 0.80,
    raft_gold_only_probability: float = 0.10,
    unannotated_evidence_policy: str = "exclude",
) -> dict[str, Any]:
    if raft_gold_probability + raft_gold_only_probability > 1:
        raise ValueError("RAFT probabilities must sum to at most one")
    if unannotated_evidence_policy != "exclude":
        raise ValueError(
            "Unsupported unannotated_evidence_policy; missing evidence annotations "
            "must not be converted into abstention targets"
        )

    dataset = LongCoQADataset()
    samples = list(dataset.iter_samples())
    samples_by_id = {str(row["sample_id"]): row for row in samples}
    dialogues = _dialogues(samples)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_by_id = {str(row["sample_id"]): str(row["split"]) for row in manifest["samples"]}
    if set(split_by_id) != set(samples_by_id):
        raise ValueError("P1 manifest sample IDs do not match the released dataset")

    alignments = {
        int(row["evidence_id"]): row for row in read_jsonl(alignment_path)
    }
    released = {
        int(row.id): str(row.value)
        for row in dataset.evidence.itertuples(index=False)
    }
    children = {
        str(row["child_id"]): row for row in read_jsonl(children_path)
    }
    retrieval = {
        str(row["sample_id"]): row for row in read_jsonl(retrieval_path)
    }
    bm25 = BM25Index.load(bm25_index_path)
    for sample_id, sample in samples_by_id.items():
        if sample_id not in retrieval:
            retrieval[sample_id] = {
                "ranked": [
                    {"child_id": child_id, "score": score}
                    for child_id, score in bm25.search(str(sample["question"]), top_k=50)
                ]
            }

    variants: dict[str, dict[str, list[dict[str, Any]]]] = {
        name: {split: [] for split in ("train", "dev", "test")}
        for name in ("conversation", "grounded", "raft")
    }
    raft_modes: Counter[str] = Counter()
    missing_retrieval: list[str] = []
    excluded_unannotated: dict[str, list[str]] = {
        split: [] for split in ("train", "dev", "test")
    }
    excluded_unannotated_by_type: Counter[str] = Counter()

    ordered = sorted(
        samples,
        key=lambda row: (
            str(row["question_type"]),
            int(row["dialogue_id"]),
            int(row["turn"]),
        ),
    )
    for sample in ordered:
        sample_id = str(sample["sample_id"])
        split = split_by_id[sample_id]
        history = _history(sample, dialogues)
        variants["conversation"][split].append(
            _example(
                sample=sample,
                split=split,
                variant="conversation",
                system=SYSTEM_PROMPT,
                prompt=str(sample["question"]),
                response=str(sample["answer"]),
                history=history,
            )
        )

        gold: list[tuple[str, str]] = []
        gold_paragraph_ids: set[str] = set()
        evidence_sources: dict[str, str] = {}
        for evidence_id in sample["evidence_ids"]:
            alignment = alignments.get(int(evidence_id))
            if alignment and alignment.get("source_text"):
                text = str(alignment["source_text"])
                evidence_sources[str(evidence_id)] = "aligned_raw"
                gold_paragraph_ids.update(map(str, alignment.get("paragraph_ids", [])))
            else:
                text = released[int(evidence_id)]
                evidence_sources[str(evidence_id)] = "released_fallback"
            gold.append((str(evidence_id), text))

        # Missing annotation is not evidence that a question is unanswerable.
        # Keep the original answer in Conversation SFT, but exclude the sample
        # from evidence-conditioned Grounded/RAFT objectives.
        if not _has_annotated_evidence(sample):
            excluded_unannotated[split].append(sample_id)
            excluded_unannotated_by_type[str(sample["question_type"])] += 1
            continue

        grounded_prompt = (
            "请依据以下证据回答问题，并在答案末尾引用证据编号。\n\n"
            f"{_evidence_block(gold)}\n\n[QUESTION]\n{sample['question']}"
        )
        grounded_response = _grounded_output(
            str(sample["answer"]), [label for label, _ in gold]
        )
        variants["grounded"][split].append(
            _example(
                sample=sample,
                split=split,
                variant="grounded",
                system=GROUNDED_SYSTEM,
                prompt=grounded_prompt,
                response=grounded_response,
                history=history,
                extra_metadata={"evidence_sources": evidence_sources},
            )
        )

        negatives = _negative_candidates(
            sample_id,
            retrieval=retrieval,
            children=children,
            gold_paragraph_ids=gold_paragraph_ids,
            count=hard_negatives,
        )
        if not negatives:
            missing_retrieval.append(sample_id)
        rng = random.Random(f"{seed}:{sample_id}")
        draw = rng.random()
        if draw < raft_gold_probability:
            mode = "gold_plus_hard_negatives"
            passages = gold + [
                (f"NEG-{index + 1}", str(row["text"]))
                for index, row in enumerate(negatives)
            ]
            response = _grounded_output(
                str(sample["answer"]), [label for label, _ in gold]
            )
        elif draw < raft_gold_probability + raft_gold_only_probability:
            mode = "gold_only"
            passages = gold
            response = _grounded_output(
                str(sample["answer"]), [label for label, _ in gold]
            )
        else:
            mode = "missing_gold_abstention"
            passages = [
                (f"NEG-{index + 1}", str(row["text"]))
                for index, row in enumerate(negatives)
            ]
            response = "提供的证据不足以回答该问题。"
        rng.shuffle(passages)
        raft_modes[f"{split}:{mode}"] += 1
        raft_prompt = (
            "下列检索结果可能包含无关内容。请只使用能直接支持答案的证据；"
            "如果没有充分证据，请明确拒答。答案末尾应引用实际使用的证据编号。\n\n"
            f"{_evidence_block(passages)}\n\n[QUESTION]\n{sample['question']}"
        )
        variants["raft"][split].append(
            _example(
                sample=sample,
                split=split,
                variant="raft",
                system=GROUNDED_SYSTEM,
                prompt=raft_prompt,
                response=response,
                history=history,
                extra_metadata={
                    "raft_mode": mode,
                    "gold_evidence_ids": [label for label, _ in gold],
                    "negative_child_ids": [
                        str(row["child_id"]) for row in negatives
                    ],
                },
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, str]] = {}
    file_checksums: dict[str, dict[str, str]] = {}
    counts: dict[str, dict[str, int]] = {}
    for variant, splits in variants.items():
        files[variant] = {}
        file_checksums[variant] = {}
        counts[variant] = {}
        for split, rows in splits.items():
            path = output_dir / f"{variant}_{split}.jsonl"
            atomic_write_jsonl(path, rows)
            files[variant][split] = str(path)
            file_checksums[variant][split] = sha256_file(path)
            counts[variant][split] = len(rows)

    dataset_info = {
        f"longcoqa_{variant}_{split}": {
            "file_name": f"{variant}_{split}.jsonl",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "system": "system",
                "history": "history",
            },
        }
        for variant in variants
        for split in ("train", "dev", "test")
    }
    atomic_write_json(output_dir / "dataset_info.json", dataset_info)
    report = {
        "schema_version": "longcoqa-training-data-v2",
        "seed": seed,
        "p1_content_hash": manifest["content_hash"],
        "counts": counts,
        "unannotated_evidence": {
            "policy": unannotated_evidence_policy,
            "excluded_from": ["grounded", "raft"],
            "retained_in": ["conversation"],
            "counts": {
                split: len(sample_ids)
                for split, sample_ids in excluded_unannotated.items()
            },
            "total": sum(len(sample_ids) for sample_ids in excluded_unannotated.values()),
            "by_question_type": dict(sorted(excluded_unannotated_by_type.items())),
            "sample_ids_hash": stable_json_hash(
                [
                    sample_id
                    for split in ("train", "dev", "test")
                    for sample_id in excluded_unannotated[split]
                ]
            ),
            "examples": {
                split: sample_ids[:10]
                for split, sample_ids in excluded_unannotated.items()
            },
        },
        "raft_modes": dict(sorted(raft_modes.items())),
        "hard_negatives": hard_negatives,
        "missing_retrieval_count": len(missing_retrieval),
        "missing_retrieval_examples": missing_retrieval[:20],
        "inputs": {
            str(path): sha256_file(path)
            for path in (
                manifest_path,
                alignment_path,
                children_path,
                retrieval_path,
                bm25_index_path,
            )
        },
        "sample_ids_hash": stable_json_hash(
            [str(row["sample_id"]) for row in ordered]
        ),
        "files": files,
        "file_checksums": file_checksums,
    }
    atomic_write_json(output_dir / "manifest.json", report)
    return report


def build_cpt_dataset(
    *,
    novel_dir: Path,
    output_dir: Path,
    encoding: str = "gb18030",
    chunk_chars: int = 8000,
    overlap_chars: int = 500,
) -> dict[str, Any]:
    if chunk_chars <= overlap_chars or overlap_chars < 0:
        raise ValueError("Require chunk_chars > overlap_chars >= 0")
    rows: list[dict[str, Any]] = []
    sources: dict[str, str] = {}
    stride = chunk_chars - overlap_chars
    novel_paths = discover_novel_paths(novel_dir)
    for path in novel_paths:
        text = path.read_text(encoding=encoding).strip()
        sources[str(path)] = sha256_file(path)
        for start in range(0, len(text), stride):
            chunk = text[start : start + chunk_chars].strip()
            if not chunk:
                continue
            rows.append(
                {
                    "text": chunk,
                    "metadata": {
                        "source": path.name,
                        "start_char": start,
                        "end_char": start + len(chunk),
                    },
                }
            )
            if start + chunk_chars >= len(text):
                break
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "novels_cpt.jsonl"
    atomic_write_jsonl(path, rows)
    report = {
        "schema_version": "longcoqa-cpt-data-v1",
        "examples": len(rows),
        "encoding": encoding,
        "chunk_chars": chunk_chars,
        "overlap_chars": overlap_chars,
        "sources": sources,
        "artifact_checksum": sha256_file(path),
    }
    atomic_write_json(output_dir / "novels_cpt_manifest.json", report)
    dataset_info_path = output_dir / "dataset_info.json"
    dataset_info = (
        json.loads(dataset_info_path.read_text(encoding="utf-8"))
        if dataset_info_path.exists()
        else {}
    )
    dataset_info["longcoqa_novels_cpt"] = {
        "file_name": "novels_cpt.jsonl",
        "columns": {"prompt": "text"},
    }
    atomic_write_json(dataset_info_path, dataset_info)
    return report
