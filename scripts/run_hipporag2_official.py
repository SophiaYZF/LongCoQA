from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from longcoqa.config import load_provider_settings
from longcoqa.utils.io import atomic_write_json, atomic_write_jsonl

ROOT = Path(__file__).resolve().parents[1]
HIPPO_ROOT = ROOT / "external/HippoRAG"
sys.path.insert(0, str(HIPPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-queries", type=int, default=0)
    parser.add_argument("--smoke-distractors", type=int, default=24)
    parser.add_argument("--embedding-url", default="http://127.0.0.1:18001/v1/embeddings")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/generated/rag/hipporag2")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    provider = load_provider_settings("qwen37")
    if not provider.api_keys:
        raise RuntimeError("qwen37 API key is not configured")
    os.environ["OPENAI_API_KEY"] = provider.api_keys[0]

    from src.hipporag.HippoRAG import HippoRAG
    from src.hipporag.utils.config_utils import BaseConfig
    from src.hipporag.utils.misc_utils import Chunk

    corpus = json.loads(
        (ROOT / "artifacts/hipporag2/longcoqa_parents.json").read_text(encoding="utf-8")
    )
    queries = json.loads(
        (ROOT / "artifacts/hipporag2/longcoqa_queries.json").read_text(encoding="utf-8")
    )
    label = "full"
    if args.smoke_queries:
        queries = queries[: args.smoke_queries]
        gold_ids = {parent_id for row in queries for parent_id in row["gold_parent_ids"]}
        selected = [row for row in corpus if row["idx"] in gold_ids]
        selected_ids = {row["idx"] for row in selected}
        distractors = [
            row
            for row in corpus
            if row["idx"] not in selected_ids
        ][: args.smoke_distractors]
        corpus = selected + distractors
        label = f"smoke{args.smoke_queries}"

    output_dir = args.output_dir / label
    config = BaseConfig(
        save_dir=str(output_dir / "index"),
        dataset="longcoqa",
        llm_base_url=provider.base_url,
        llm_name=provider.model,
        embedding_model_name=(
            "VLLM/models/Qwen/Qwen3-Embedding-4B"
        ),
        embedding_base_url=args.embedding_url,
        force_index_from_scratch=args.force,
        force_openie_from_scratch=args.force,
        openie_mode="online",
        retrieval_top_k=100,
        linking_top_k=5,
        synonymy_edge_topk=64,
        embedding_batch_size=32,
        max_new_tokens=512,
        max_retry_attempts=8,
    )
    rag = HippoRAG(global_config=config)
    docs = [
        Chunk(
            content=f"{row['title']}\n{row['text']}",
            source_id=str(row["idx"]),
            metadata={"parent_id": str(row["idx"])},
        )
        for row in corpus
    ]
    rag.index(docs)
    solutions = rag.retrieve([row["question"] for row in queries], num_to_retrieve=100)
    rows = []
    for query, solution in zip(queries, solutions, strict=True):
        ranked = [
            {
                "parent_id": str(metadata.get("parent_id", "")),
                "score": float(score),
            }
            for metadata, score in zip(
                solution.doc_metadata or [], solution.doc_scores.tolist(), strict=True
            )
        ]
        ranked_ids = [row["parent_id"] for row in ranked]
        gold = set(query["gold_parent_ids"])
        row = {**query, "ranked": ranked}
        if gold:
            row.update(
                {
                    f"hit_at_{cutoff}": float(bool(gold & set(ranked_ids[:cutoff])))
                    for cutoff in (1, 5, 10, 20)
                }
            )
        rows.append(row)
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluated_rows = [row for row in rows if row["gold_parent_ids"]]
    atomic_write_jsonl(output_dir / "all_results.jsonl", rows)
    atomic_write_jsonl(output_dir / "results.jsonl", evaluated_rows)
    summary = {
        "schema_version": "longcoqa-hipporag2-official-v1",
        "official_commit": "1e8f60981bf760b64003aa5bf5668126d0c106b3",
        "compatibility_patches": [
            "lazy optional imports for current vLLM environment",
            "cap online OpenIE concurrency at four workers",
            "retain CJK characters in entity normalization",
        ],
        "indexed_parents": len(corpus),
        "generation_queries": len(rows),
        "evaluated_queries": len(evaluated_rows),
        "metrics": {
            f"hit_at_{cutoff}": sum(row[f"hit_at_{cutoff}"] for row in evaluated_rows)
            / len(evaluated_rows)
            for cutoff in (1, 5, 10, 20)
        },
    }
    atomic_write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
