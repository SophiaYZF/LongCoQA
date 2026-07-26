from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from longcoqa.data import LongCoQADataset
from longcoqa.retrieval.bm25 import BM25Index
from longcoqa.utils.hashing import sha256_file, stable_json_hash
from longcoqa.utils.io import atomic_write_json, atomic_write_jsonl, read_jsonl


def _gold_child_mapping(
    alignments: list[dict[str, Any]],
    children: list[dict[str, Any]],
) -> dict[int, set[str]]:
    paragraph_to_child = {
        paragraph_id: str(child["child_id"]) for child in children for paragraph_id in child["paragraph_ids"]
    }
    mapping: dict[int, set[str]] = {}
    for alignment in alignments:
        mapping[int(alignment["evidence_id"])] = {
            paragraph_to_child[paragraph_id]
            for paragraph_id in alignment["paragraph_ids"]
            if paragraph_id in paragraph_to_child
        }
    return mapping


def build_retrieval_queries(
    dataset: LongCoQADataset,
    alignments: list[dict[str, Any]],
    children: list[dict[str, Any]],
    *,
    query_mode: str = "history_questions",
) -> list[dict[str, Any]]:
    if query_mode not in {"current_question", "history_questions"}:
        raise ValueError(f"Unsupported query_mode={query_mode!r}")
    gold_by_evidence = _gold_child_mapping(alignments, children)
    history: dict[tuple[str, int], list[str]] = defaultdict(list)
    queries: list[dict[str, Any]] = []
    for sample in dataset.iter_samples():
        dialogue_key = (str(sample["question_type"]), int(sample["dialogue_id"]))
        history[dialogue_key].append(str(sample["question"]))
        query = (
            str(sample["question"]) if query_mode == "current_question" else "\n".join(history[dialogue_key])
        )
        gold_children = {
            child_id
            for evidence_id in sample["evidence_ids"]
            for child_id in gold_by_evidence.get(int(evidence_id), set())
        }
        if not gold_children:
            continue
        queries.append(
            {
                "sample_id": sample["sample_id"],
                "query_mode": query_mode,
                "query": query,
                "gold_child_ids": sorted(gold_children),
            }
        )
    return queries


def build_all_retrieval_queries(
    dataset: LongCoQADataset,
    *,
    query_mode: str = "history_questions",
    rewrites: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    if query_mode not in {
        "current_question",
        "history_questions",
        "rewritten_dialogue",
    }:
        raise ValueError(f"Unsupported query_mode={query_mode!r}")
    if query_mode == "rewritten_dialogue" and rewrites is None:
        raise ValueError("rewritten_dialogue requires rewrites")
    history: dict[tuple[str, int], list[str]] = defaultdict(list)
    queries: list[dict[str, Any]] = []
    for sample in dataset.iter_samples():
        sample_id = str(sample["sample_id"])
        dialogue_key = (str(sample["question_type"]), int(sample["dialogue_id"]))
        history[dialogue_key].append(str(sample["question"]))
        if query_mode == "current_question":
            query = str(sample["question"])
        elif query_mode == "history_questions":
            query = "\n".join(history[dialogue_key])
        else:
            assert rewrites is not None
            if sample_id not in rewrites:
                raise ValueError(f"Missing rewrite for sample {sample_id}")
            query = str(rewrites[sample_id])
        queries.append(
            {
                "sample_id": sample_id,
                "query_mode": query_mode,
                "query": query,
            }
        )
    return queries


def build_all_ranking_results(
    queries: list[dict[str, Any]],
    rankings: list[list[tuple[str, float]]],
    *,
    method: str,
) -> list[dict[str, Any]]:
    if len(queries) != len(rankings):
        raise ValueError("queries and rankings differ in size")
    return [
        {
            **query,
            "method": method,
            "ranked": [
                {"child_id": str(document_id), "score": float(score)}
                for document_id, score in ranked
            ],
        }
        for query, ranked in zip(queries, rankings, strict=True)
    ]


def project_rankings_for_evaluation(
    all_results: list[dict[str, Any]],
    evaluation_queries: list[dict[str, Any]],
) -> list[list[tuple[str, float]]]:
    by_id: dict[str, list[tuple[str, float]]] = {}
    for row in all_results:
        sample_id = str(row["sample_id"])
        if sample_id in by_id:
            raise ValueError(f"Duplicate all-results sample ID: {sample_id}")
        by_id[sample_id] = [
            (str(item["child_id"]), float(item["score"]))
            for item in row["ranked"]
        ]
    missing = [
        str(query["sample_id"])
        for query in evaluation_queries
        if str(query["sample_id"]) not in by_id
    ]
    if missing:
        raise ValueError(f"All-results rankings are missing evaluation IDs: {missing[:5]}")
    return [by_id[str(query["sample_id"])] for query in evaluation_queries]


def evaluate_rankings(
    queries: list[dict[str, Any]],
    rankings: list[list[tuple[str, float]]],
    *,
    method: str,
    cutoffs: tuple[int, ...] = (1, 5, 10, 20),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(queries) != len(rankings):
        raise ValueError("queries and rankings differ in size")
    results: list[dict[str, Any]] = []
    for query_item, ranked in zip(queries, rankings, strict=True):
        gold_children = set(query_item["gold_child_ids"])
        ranked_ids = [document_id for document_id, _ in ranked]
        relevant_ranks = [
            rank for rank, document_id in enumerate(ranked_ids, start=1) if document_id in gold_children
        ]
        item: dict[str, Any] = {
            **query_item,
            "method": method,
            "gold_child_ids": sorted(gold_children),
            "ranked": [{"child_id": document_id, "score": score} for document_id, score in ranked],
            "reciprocal_rank": 1.0 / relevant_ranks[0] if relevant_ranks else 0.0,
        }
        for cutoff in cutoffs:
            retrieved = set(ranked_ids[:cutoff])
            relevant = len(gold_children & retrieved)
            item[f"hit_at_{cutoff}"] = float(relevant > 0)
            item[f"recall_at_{cutoff}"] = relevant / len(gold_children)
            item[f"all_gold_at_{cutoff}"] = float(relevant == len(gold_children))
            gains = [
                1.0 / math.log2(rank + 1)
                for rank, document_id in enumerate(ranked_ids[:cutoff], start=1)
                if document_id in gold_children
            ]
            ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold_children), cutoff) + 1))
            item[f"ndcg_at_{cutoff}"] = sum(gains) / ideal if ideal else 0.0
        results.append(item)

    if not results:
        raise ValueError("No retrieval queries with gold evidence were evaluated")
    metric_names = ["reciprocal_rank"] + [
        f"{name}_at_{cutoff}" for cutoff in cutoffs for name in ("hit", "recall", "all_gold", "ndcg")
    ]
    summary = {
        "schema_version": "longcoqa-retrieval-eval-v1",
        "method": method,
        "query_mode": str(queries[0]["query_mode"]),
        "evaluated_queries": len(results),
        "cutoffs": list(cutoffs),
        "metrics": {
            metric: sum(float(item[metric]) for item in results) / len(results) for metric in metric_names
        },
        "result_content_hash": stable_json_hash(
            [
                {
                    "sample_id": item["sample_id"],
                    "ranked_ids": [ranked["child_id"] for ranked in item["ranked"]],
                }
                for item in results
            ]
        ),
    }
    return results, summary


def evaluate_retriever(
    dataset: LongCoQADataset,
    index: BM25Index,
    alignments: list[dict[str, Any]],
    children: list[dict[str, Any]],
    *,
    query_mode: str = "history_questions",
    cutoffs: tuple[int, ...] = (1, 5, 10, 20),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    queries = build_retrieval_queries(
        dataset,
        alignments,
        children,
        query_mode=query_mode,
    )
    max_k = max(cutoffs)
    rankings = [index.search(str(item["query"]), top_k=max_k) for item in queries]
    return evaluate_rankings(
        queries,
        rankings,
        method="bm25",
        cutoffs=cutoffs,
    )


def write_retrieval_evaluation(
    dataset: LongCoQADataset,
    index_path: Path,
    alignment_path: Path,
    children_path: Path,
    output_dir: Path,
    *,
    query_modes: tuple[str, ...] = ("current_question", "history_questions"),
) -> dict[str, Any]:
    index = BM25Index.load(index_path)
    alignments = read_jsonl(alignment_path)
    children = read_jsonl(children_path)
    reports: dict[str, Any] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for query_mode in query_modes:
        results, summary = evaluate_retriever(
            dataset,
            index,
            alignments,
            children,
            query_mode=query_mode,
        )
        result_path = output_dir / f"bm25_{query_mode}.jsonl"
        atomic_write_jsonl(result_path, results)
        summary["artifact_checksum"] = sha256_file(result_path)
        atomic_write_json(output_dir / f"bm25_{query_mode}_summary.json", summary)
        reports[query_mode] = summary
    return reports
