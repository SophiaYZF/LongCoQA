from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from longcoqa.data import LongCoQADataset
from longcoqa.retrieval.evaluate import (
    build_all_ranking_results,
    build_retrieval_queries,
    evaluate_rankings,
    project_rankings_for_evaluation,
)
from longcoqa.utils.hashing import sha256_file
from longcoqa.utils.io import atomic_write_json, atomic_write_jsonl, read_jsonl


def build_position_neighbors(children: list[dict[str, Any]], window: int = 3) -> dict[str, set[str]]:
    by_chapter: dict[str, list[str]] = defaultdict(list)
    for row in children:
        by_chapter[str(row["chapter_id"])].append(str(row["child_id"]))
    neighbors: dict[str, set[str]] = defaultdict(set)
    for ids in by_chapter.values():
        for index, child_id in enumerate(ids):
            for other in ids[max(0, index - window) : index + window + 1]:
                if other != child_id:
                    neighbors[child_id].add(other)
    return neighbors


def progressive_graph_ranking(
    base: list[tuple[str, float]],
    *,
    semantic_neighbors: dict[str, list[tuple[str, float]]],
    position_neighbors: dict[str, set[str]],
    seed_k: int = 3,
    steps: int = 3,
    semantic_weight: float = 0.25,
    position_weight: float = 0.10,
    base_weight: float = 0.65,
    top_k: int = 100,
) -> list[tuple[str, float]]:
    if not base:
        return []
    values = np.asarray([score for _, score in base], dtype=np.float32)
    span = max(float(values.max() - values.min()), 1e-8)
    scores = {child_id: base_weight * float((score - values.min()) / span) for child_id, score in base}
    frontier = {child_id: 1.0 for child_id, _ in base[:seed_k]}
    for step in range(steps):
        next_frontier: dict[str, float] = defaultdict(float)
        decay = 1.0 / (step + 1)
        for source, source_score in frontier.items():
            for target, similarity in semantic_neighbors.get(source, []):
                contribution = decay * semantic_weight * source_score * max(0.0, similarity)
                scores[target] = scores.get(target, 0.0) + contribution
                next_frontier[target] += contribution
            for target in position_neighbors.get(source, set()):
                contribution = decay * position_weight * source_score
                scores[target] = scores.get(target, 0.0) + contribution
                next_frontier[target] += contribution
        frontier = dict(sorted(next_frontier.items(), key=lambda item: -item[1])[:seed_k])
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]


def run_graph_progressive(
    *,
    dense_index_path: Path,
    base_rankings_path: Path,
    children_path: Path,
    alignment_path: Path,
    output_dir: Path,
    semantic_k: int = 10,
    position_window: int = 3,
    seed_k: int = 3,
    steps: int = 3,
    top_k: int = 100,
    device: str = "cuda",
) -> dict[str, Any]:
    import torch

    index = np.load(dense_index_path)
    ids = [str(value) for value in index["document_ids"].tolist()]
    vectors = torch.from_numpy(np.asarray(index["vectors"], dtype=np.float32)).to(device, torch.float16)
    semantic: dict[str, list[tuple[str, float]]] = {}
    for start in range(0, len(ids), 512):
        scores = vectors[start : start + 512] @ vectors.T
        values, indices = torch.topk(scores, k=semantic_k + 1, dim=1)
        for offset, (row_values, row_indices) in enumerate(
            zip(values.float().cpu().tolist(), indices.cpu().tolist(), strict=True)
        ):
            source = ids[start + offset]
            semantic[source] = [
                (ids[target], float(score))
                for score, target in zip(row_values, row_indices, strict=True)
                if ids[target] != source
            ][:semantic_k]
    del vectors
    torch.cuda.empty_cache()

    children = read_jsonl(children_path)
    position = build_position_neighbors(children, window=position_window)
    base_rows = read_jsonl(base_rankings_path)
    base_by_id = {
        str(row["sample_id"]): [
            (str(item["child_id"]), float(item["score"])) for item in row["ranked"]
        ]
        for row in base_rows
    }
    evaluation_queries = build_retrieval_queries(
        LongCoQADataset(), read_jsonl(alignment_path), children, query_mode="current_question"
    )
    rewritten = {str(row["sample_id"]): str(row.get("query", "")) for row in base_rows}
    all_queries = [
        {
            "sample_id": str(row["sample_id"]),
            "query_mode": "graph_progressive_rewritten",
            "query": str(row.get("query", "")),
        }
        for row in base_rows
    ]
    all_rankings = [
        progressive_graph_ranking(
            base_by_id[str(row["sample_id"])],
            semantic_neighbors=semantic,
            position_neighbors=position,
            seed_k=seed_k,
            steps=steps,
            top_k=top_k,
        )
        for row in all_queries
    ]
    all_results = build_all_ranking_results(
        all_queries,
        all_rankings,
        method="chainrag_graph_progressive_adaptation",
    )
    eval_queries = [
        {
            **row,
            "query_mode": "graph_progressive_rewritten",
            "query": rewritten[str(row["sample_id"])],
        }
        for row in evaluation_queries
        if str(row["sample_id"]) in base_by_id
    ]
    rankings = project_rankings_for_evaluation(all_results, eval_queries)
    results, summary = evaluate_rankings(
        eval_queries, rankings, method="chainrag_graph_progressive_adaptation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    all_result_path = output_dir / "all_results.jsonl"
    atomic_write_jsonl(all_result_path, all_results)
    result_path = output_dir / "results.jsonl"
    atomic_write_jsonl(result_path, results)
    summary.update(
        {
            "adaptation": (
                "Chinese controlled adaptation: dense seed retrieval plus semantic and "
                "within-chapter positional graph propagation; no English spaCy pipeline."
            ),
            "semantic_k": semantic_k,
            "position_window": position_window,
            "seed_k": seed_k,
            "steps": steps,
            "base_rankings_checksum": sha256_file(base_rankings_path),
            "all_queries": len(all_results),
            "all_results_checksum": sha256_file(all_result_path),
            "artifact_checksum": sha256_file(result_path),
        }
    )
    atomic_write_json(output_dir / "summary.json", summary)
    return summary
