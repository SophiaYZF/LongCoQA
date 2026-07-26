from __future__ import annotations

import json
from pathlib import Path

from longcoqa.data import LongCoQADataset
from longcoqa.retrieval.evaluate import (
    build_all_retrieval_queries,
    build_retrieval_queries,
)
from longcoqa.utils.io import atomic_write_json, read_jsonl

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parents = read_jsonl(ROOT / "artifacts/corpus/parents.jsonl")
    children = read_jsonl(ROOT / "artifacts/corpus/children.jsonl")
    alignments = read_jsonl(ROOT / "artifacts/alignments/evidence_alignment.jsonl")
    rewrites = {
        str(row["sample_id"]): str(row["rewritten_query"])
        for row in read_jsonl(ROOT / "reports/generated/rag/query_rewrites.jsonl")
    }
    child_to_parent = {str(row["child_id"]): str(row["parent_id"]) for row in children}
    corpus = [
        {
            "idx": str(row["parent_id"]),
            "title": f"{row['book_title']} / {row['chapter_title']}",
            "text": str(row["text"]),
        }
        for row in parents
    ]
    evaluation_queries = build_retrieval_queries(
        LongCoQADataset(), alignments, children, query_mode="current_question"
    )
    gold_by_id = {
        str(row["sample_id"]): sorted(
            {child_to_parent[child_id] for child_id in row["gold_child_ids"]}
        )
        for row in evaluation_queries
    }
    current_by_id = {
        str(row["sample_id"]): str(row["query"])
        for row in build_all_retrieval_queries(
            LongCoQADataset(), query_mode="current_question"
        )
    }
    queries = build_all_retrieval_queries(
        LongCoQADataset(), query_mode="rewritten_dialogue", rewrites=rewrites
    )
    samples = [
        {
            "id": str(row["sample_id"]),
            "question": str(row["query"]),
            "original_question": current_by_id[str(row["sample_id"])],
            "gold_parent_ids": gold_by_id.get(str(row["sample_id"]), []),
        }
        for row in queries
    ]
    output = ROOT / "artifacts/hipporag2"
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "longcoqa_parents.json", corpus)
    atomic_write_json(output / "longcoqa_queries.json", samples)
    print(
        json.dumps(
            {"parents": len(corpus), "queries": len(samples), "output": str(output)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
