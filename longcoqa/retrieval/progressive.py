from __future__ import annotations

import gc
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from longcoqa.data import LongCoQADataset
from longcoqa.retrieval.dense import (
    DEFAULT_QUERY_INSTRUCTION,
    _dense_rankings,
    _embed_with_vllm,
    _query_prompt,
    reciprocal_rank_fusion,
)
from longcoqa.retrieval.evaluate import (
    build_all_ranking_results,
    build_retrieval_queries,
    evaluate_rankings,
    project_rankings_for_evaluation,
)
from longcoqa.utils.hashing import sha256_file
from longcoqa.utils.io import atomic_write_json, atomic_write_jsonl, read_jsonl

PLAN_SYSTEM = """你是多跳小说检索规划器。根据独立问题和第一轮检索片段，判断还缺哪些事实，
生成1至3条可独立检索的中文子查询。每条子查询必须补全人物指代，并聚焦一个事实，不要回答问题。
只输出JSON：{"queries":["..."]}。"""


def parse_subqueries(text: str, fallback: str) -> list[str]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group())
            values = payload.get("queries", [])
            if isinstance(values, list):
                queries = [str(value).strip() for value in values if str(value).strip()]
                if queries:
                    return list(dict.fromkeys(queries))[:3]
        except json.JSONDecodeError:
            pass
    return [fallback]


def generate_progressive_plans(
    *,
    model_path: Path,
    base_rankings_path: Path,
    children_path: Path,
    output_path: Path,
    limit: int | None = None,
    seed_k: int = 3,
    batch_size: int = 256,
    gpu_memory_utilization: float = 0.90,
    max_model_len: int = 4096,
    max_num_seqs: int = 256,
) -> dict[str, Any]:
    from vllm import LLM, SamplingParams

    rows = read_jsonl(base_rankings_path)
    if limit is not None:
        rows = rows[:limit]
    child_by_id = {str(row["child_id"]): row for row in read_jsonl(children_path)}
    messages = []
    for row in rows:
        contexts = []
        for rank, item in enumerate(row["ranked"][:seed_k], start=1):
            child = child_by_id[str(item["child_id"])]
            contexts.append(f"[{rank}] {child['book_title']} / {child['chapter_title']}\n{child['text']}")
        messages.append(
            [
                {"role": "system", "content": PLAN_SYSTEM},
                {
                    "role": "user",
                    "content": f"[问题]\n{row['query']}\n\n[第一轮检索]\n" + "\n\n".join(contexts),
                },
            ]
        )
    model = LLM(
        model=str(model_path),
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        enable_prefix_caching=True,
    )
    outputs = []
    params = SamplingParams(temperature=0, max_tokens=192)
    for start in range(0, len(messages), batch_size):
        outputs.extend(
            model.chat(
                messages[start : start + batch_size],
                sampling_params=params,
                chat_template_kwargs={"enable_thinking": False},
                use_tqdm=True,
            )
        )
    del model
    gc.collect()
    plans = [
        {
            "sample_id": row["sample_id"],
            "query": row["query"],
            "subqueries": parse_subqueries(output.outputs[0].text, str(row["query"])),
            "raw_plan": output.outputs[0].text,
        }
        for row, output in zip(rows, outputs, strict=True)
    ]
    atomic_write_jsonl(output_path, plans)
    summary = {
        "schema_version": "longcoqa-progressive-plan-v1",
        "examples": len(plans),
        "mean_subqueries": sum(len(row["subqueries"]) for row in plans) / len(plans),
        "artifact_checksum": sha256_file(output_path),
    }
    atomic_write_json(output_path.with_suffix(".summary.json"), summary)
    return summary


def retrieve_progressive_plans(
    *,
    embedding_model_path: Path,
    dense_index_path: Path,
    plans_path: Path,
    base_rankings_path: Path,
    children_path: Path,
    alignment_path: Path,
    output_dir: Path,
    dimensions: int = 1024,
    top_k: int = 100,
    gpu_memory_utilization: float = 0.90,
    max_model_len: int = 2048,
    max_num_seqs: int = 512,
) -> dict[str, Any]:
    from vllm import LLM

    plans = read_jsonl(plans_path)
    base = {
        str(row["sample_id"]): [
            (str(item["child_id"]), float(item["score"])) for item in row["ranked"]
        ]
        for row in read_jsonl(base_rankings_path)
    }
    flat_queries = [
        str(query)
        for plan in plans
        for query in plan["subqueries"]
    ]
    index = np.load(dense_index_path)
    document_ids = [str(value) for value in index["document_ids"].tolist()]
    document_vectors = np.asarray(index["vectors"], dtype=np.float32)
    model = LLM(
        model=str(embedding_model_path),
        runner="pooling",
        convert="embed",
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
    )
    vectors = _embed_with_vllm(
        model,
        [_query_prompt(query, DEFAULT_QUERY_INSTRUCTION) for query in flat_queries],
        dimensions=dimensions,
    )
    del model
    gc.collect()
    sub_rankings = _dense_rankings(vectors, document_vectors, document_ids, top_k=top_k)
    cursor = 0
    fused_rankings = []
    for plan in plans:
        ranking = base[str(plan["sample_id"])]
        for _ in plan["subqueries"]:
            ranking = reciprocal_rank_fusion(ranking, sub_rankings[cursor], top_k=top_k)
            cursor += 1
        fused_rankings.append(ranking)

    children = read_jsonl(children_path)
    gold = {
        str(row["sample_id"]): row
        for row in build_retrieval_queries(
            LongCoQADataset(),
            read_jsonl(alignment_path),
            children,
            query_mode="current_question",
        )
    }
    all_queries = [
        {
            "sample_id": str(plan["sample_id"]),
            "query_mode": "chainrag_progressive",
            "query": str(plan["query"]),
        }
        for plan in plans
    ]
    all_results = build_all_ranking_results(
        all_queries,
        fused_rankings,
        method="chainrag_progressive_query_adaptation",
    )
    queries = [
        {
            **gold[str(plan["sample_id"])],
            "query_mode": "chainrag_progressive",
            "query": plan["query"],
        }
        for plan in plans
        if str(plan["sample_id"]) in gold
    ]
    rankings = project_rankings_for_evaluation(all_results, queries)
    results, summary = evaluate_rankings(
        queries, rankings, method="chainrag_progressive_query_adaptation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    all_result_path = output_dir / "all_results.jsonl"
    atomic_write_jsonl(all_result_path, all_results)
    result_path = output_dir / "results.jsonl"
    atomic_write_jsonl(result_path, results)
    summary.update(
        {
            "plans_checksum": sha256_file(plans_path),
            "all_queries": len(all_results),
            "all_results_checksum": sha256_file(all_result_path),
            "artifact_checksum": sha256_file(result_path),
        }
    )
    atomic_write_json(output_dir / "summary.json", summary)
    return summary
