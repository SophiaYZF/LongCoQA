from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from longcoqa.data import LongCoQADataset
from longcoqa.inference.prompts import index_samples
from longcoqa.utils.hashing import sha256_file, stable_json_hash
from longcoqa.utils.io import atomic_write_json, atomic_write_jsonl, read_jsonl

RAG_SYSTEM = """你是《哈利·波特》系列小说的检索增强问答助手。
只依据检索片段回答事实，但可使用对话历史和独立检索查询消解当前问题的指代。允许从证据中的直接陈述、
同义表达或明确身份称谓作普通语义推断；不要因为措辞不完全相同而拒答。证据确实不足时才明确拒答。
事实型问题只输出一行简短答案和一行引用，不解释推理过程；解释型问题应覆盖问题要求。
答案末尾引用实际使用的 RETRIEVED 编号。"""


def normalize_ranking_row(
    row: dict[str, Any], ranking_level: str
) -> dict[str, Any]:
    if ranking_level not in {"child", "parent"}:
        raise ValueError(
            f"ranking_level must be 'child' or 'parent', found {ranking_level!r}"
        )
    sample_id = row.get("sample_id", row.get("id"))
    if sample_id is None or not str(sample_id):
        raise ValueError("Ranking row is missing sample_id/id")
    ranked = row.get("ranked")
    if not isinstance(ranked, list) or not ranked:
        raise ValueError(f"Ranking row {sample_id} has no ranked results")
    id_key = f"{ranking_level}_id"
    normalized_ranked: list[dict[str, Any]] = []
    for rank, item in enumerate(ranked, start=1):
        if not isinstance(item, dict) or id_key not in item:
            raise ValueError(f"Ranking row {sample_id} rank {rank} is missing {id_key}")
        normalized_ranked.append(
            {**item, id_key: str(item[id_key]), "score": float(item["score"])}
        )
    return {
        **row,
        "sample_id": str(sample_id),
        "query": row.get("query", row.get("question")),
        "ranked": normalized_ranked,
    }


def select_ranked_parent_contexts(
    ranked: list[dict[str, Any]],
    *,
    ranking_level: str,
    children: dict[str, dict[str, Any]],
    parents: dict[str, dict[str, Any]],
    top_k: int,
    chronological: bool,
) -> list[dict[str, Any]]:
    if ranking_level not in {"child", "parent"}:
        raise ValueError(
            f"ranking_level must be 'child' or 'parent', found {ranking_level!r}"
        )
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in ranked:
        matched_child_id: str | None
        if ranking_level == "child":
            matched_child_id = str(item["child_id"])
            if matched_child_id not in children:
                raise ValueError(f"Unknown child_id: {matched_child_id}")
            parent_id = str(children[matched_child_id]["parent_id"])
        else:
            matched_child_id = None
            parent_id = str(item["parent_id"])
        if parent_id not in parents:
            raise ValueError(f"Unknown parent_id: {parent_id}")
        if parent_id in seen:
            continue
        parent = dict(parents[parent_id])
        parent["retrieval_score"] = float(item["score"])
        parent["matched_child_id"] = matched_child_id
        selected.append(parent)
        seen.add(parent_id)
        if len(selected) == top_k:
            break
    if chronological:
        selected.sort(
            key=lambda value: (
                int(value["book_id"]),
                str(value["chapter_id"]),
                str(value["parent_id"]),
            )
        )
    return selected


def select_parent_contexts(
    ranked: list[dict[str, Any]],
    *,
    children: dict[str, dict[str, Any]],
    parents: dict[str, dict[str, Any]],
    top_k: int,
    chronological: bool,
) -> list[dict[str, Any]]:
    return select_ranked_parent_contexts(
        ranked,
        ranking_level="child",
        children=children,
        parents=parents,
        top_k=top_k,
        chronological=chronological,
    )


def build_rag_contexts(
    *,
    retrieval_path: Path,
    children_path: Path,
    parents_path: Path,
    output_path: Path,
    top_k: int = 5,
    chronological: bool = True,
    ranking_level: str = "child",
    expected_samples: int = 3019,
) -> dict[str, Any]:
    raw_retrieval = read_jsonl(retrieval_path)
    if len(raw_retrieval) != expected_samples:
        raise ValueError(
            f"Expected {expected_samples} retrieval samples, found {len(raw_retrieval)}"
        )
    normalized = [normalize_ranking_row(row, ranking_level) for row in raw_retrieval]
    retrieval_ids = [str(row["sample_id"]) for row in normalized]
    if len(set(retrieval_ids)) != len(retrieval_ids):
        duplicates = sorted(
            sample_id
            for sample_id, count in Counter(retrieval_ids).items()
            if count > 1
        )
        raise ValueError(f"Duplicate retrieval sample IDs: {duplicates[:5]}")
    retrieval = {str(row["sample_id"]): row for row in normalized}
    children = {
        str(row["child_id"]): row for row in read_jsonl(children_path)
    }
    parents = {
        str(row["parent_id"]): row for row in read_jsonl(parents_path)
    }
    all_samples = list(LongCoQADataset().iter_samples())
    dataset_ids = {str(sample["sample_id"]) for sample in all_samples}
    unknown = sorted(set(retrieval) - dataset_ids)
    if unknown:
        raise ValueError(f"Retrieval contains unknown sample IDs: {unknown[:5]}")
    samples = [
        sample
        for sample in all_samples
        if str(sample["sample_id"]) in retrieval
    ]
    dialogues = index_samples(samples)
    if len(samples) != expected_samples:
        raise ValueError(
            f"Expected {expected_samples} dataset matches, found {len(samples)}"
        )

    rows = []
    for sample in samples:
        result = retrieval[str(sample["sample_id"])]
        contexts = select_ranked_parent_contexts(
            result["ranked"],
            ranking_level=ranking_level,
            children=children,
            parents=parents,
            top_k=top_k,
            chronological=chronological,
        )
        evidence = "\n\n".join(
            (
                f"[RETRIEVED {index + 1} | {row['book_title']} | "
                f"{row['chapter_title']}]\n{row['text']}"
            )
            for index, row in enumerate(contexts)
        )
        key = f"{sample['question_type']}:{sample['dialogue_id']}"
        messages: list[dict[str, str]] = [
            {"role": "system", "content": RAG_SYSTEM}
        ]
        for prior in dialogues[key]:
            if int(prior["turn"]) >= int(sample["turn"]):
                break
            messages.extend(
                [
                    {"role": "user", "content": str(prior["question"])},
                    {"role": "assistant", "content": str(prior["answer"])},
                ]
            )
        messages.append(
            {
                "role": "user",
                "content": (
                    "请依据下列检索片段回答当前问题；片段可能含有噪声。证据不足时明确拒答，"
                    "不要使用未提供的小说知识。\n\n"
                    f"{evidence}\n\n[STANDALONE RETRIEVAL QUERY]\n{result.get('query')}"
                    f"\n\n[CURRENT QUESTION]\n{sample['question']}"
                ),
            }
        )
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "question_type": sample["question_type"],
                "dialogue_id": sample["dialogue_id"],
                "turn": sample["turn"],
                "question": sample["question"],
                "reference": sample["answer"],
                "evidence_ids": sample["evidence_ids"],
                "query": result.get("query"),
                "retrieval_method": result.get("method"),
                "retrieved_parent_ids": [row["parent_id"] for row in contexts],
                "retrieved_child_ids": [
                    row["matched_child_id"]
                    for row in contexts
                    if row["matched_child_id"] is not None
                ],
                "messages": messages,
            }
        )
    atomic_write_jsonl(output_path, rows)
    report = {
        "schema_version": "longcoqa-rag-context-v1",
        "examples": len(rows),
        "expected_samples": expected_samples,
        "ranking_level": ranking_level,
        "retrieval_method_counts": dict(
            sorted(
                Counter(
                    str(row.get("method") or ranking_level) for row in normalized
                ).items()
            )
        ),
        "top_k": top_k,
        "parent_expansion": True,
        "chronological": chronological,
        "retrieval_checksum": sha256_file(retrieval_path),
        "children_checksum": sha256_file(children_path),
        "parents_checksum": sha256_file(parents_path),
        "sample_ids_hash": stable_json_hash([row["sample_id"] for row in rows]),
        "artifact_checksum": sha256_file(output_path),
    }
    atomic_write_json(output_path.with_suffix(".summary.json"), report)
    return report
