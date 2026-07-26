from __future__ import annotations

import gc
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from longcoqa.data import LongCoQADataset
from longcoqa.retrieval.dense import (
    DEFAULT_QUERY_INSTRUCTION,
    _dense_rankings,
    _embed_with_vllm,
    _query_prompt,
)
from longcoqa.retrieval.evaluate import (
    build_retrieval_queries,
    evaluate_rankings,
)
from longcoqa.utils.hashing import sha256_file, stable_json_hash
from longcoqa.utils.io import atomic_write_json, atomic_write_jsonl, read_jsonl

REWRITE_SYSTEM = """你负责为小说问答检索器改写查询。
结合对话历史，把当前问题改写成一句可独立检索的中文查询：消解“他/她/这件事”等指代，保留人物、地点、
事件和时间等关键约束。不要回答问题，不要解释，不要添加原对话没有的事实，只输出改写后的查询。"""


def build_rewrite_messages(
    samples: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[list[dict[str, str]]]]:
    history: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    messages: list[list[dict[str, str]]] = []
    for sample in samples:
        key = (str(sample["question_type"]), int(sample["dialogue_id"]))
        prior = history[key]
        transcript = "\n".join(
            f"用户：{row['question']}\n助手：{row['answer']}" for row in prior
        )
        content = (
            f"[对话历史]\n{transcript or '（无）'}\n\n"
            f"[当前问题]\n{sample['question']}"
        )
        messages.append(
            [
                {"role": "system", "content": REWRITE_SYSTEM},
                {"role": "user", "content": content},
            ]
        )
        prior.append(sample)
    return samples, messages


def run_local_query_rewrite(
    *,
    model_path: Path,
    output_path: Path,
    model_revision: str,
    gpu_memory_utilization: float = 0.90,
    max_model_len: int = 8192,
    max_num_seqs: int = 256,
    limit: int | None = None,
) -> dict[str, Any]:
    from vllm import LLM, SamplingParams

    samples = list(LongCoQADataset().iter_samples())
    if limit is not None:
        samples = samples[:limit]
    samples, messages = build_rewrite_messages(samples)
    model = LLM(
        model=str(model_path),
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        enable_prefix_caching=True,
    )
    outputs = model.chat(
        messages,
        sampling_params=SamplingParams(temperature=0, max_tokens=96),
        chat_template_kwargs={"enable_thinking": False},
        use_tqdm=True,
    )
    rows = []
    for sample, output in zip(samples, outputs, strict=True):
        rewritten = output.outputs[0].text.strip().strip('"“”')
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "original_query": sample["question"],
                "rewritten_query": rewritten,
                "question_type": sample["question_type"],
                "dialogue_id": sample["dialogue_id"],
                "turn": sample["turn"],
            }
        )
    del model
    gc.collect()
    atomic_write_jsonl(output_path, rows)
    report = {
        "schema_version": "longcoqa-query-rewrite-v1",
        "model_path": str(model_path),
        "model_revision": model_revision,
        "examples": len(rows),
        "changed": sum(
            row["original_query"].strip() != row["rewritten_query"].strip()
            for row in rows
        ),
        "sample_ids_hash": stable_json_hash([row["sample_id"] for row in rows]),
        "artifact_checksum": sha256_file(output_path),
    }
    atomic_write_json(output_path.with_suffix(".summary.json"), report)
    return report


def retrieve_rewritten_queries(
    *,
    embedding_model_path: Path,
    dense_index_path: Path,
    rewrites_path: Path,
    alignment_path: Path,
    children_path: Path,
    output_dir: Path,
    dimensions: int = 1024,
    top_k: int = 100,
    instruction: str = DEFAULT_QUERY_INSTRUCTION,
    gpu_memory_utilization: float = 0.90,
    max_model_len: int = 2048,
    max_num_seqs: int = 512,
) -> dict[str, Any]:
    from vllm import LLM

    rewrites = read_jsonl(rewrites_path)
    rewrite_by_id = {
        str(row["sample_id"]): str(row["rewritten_query"]) for row in rewrites
    }
    samples = list(LongCoQADataset().iter_samples())
    if set(rewrite_by_id) != {str(row["sample_id"]) for row in samples}:
        raise ValueError("Query rewrite file must contain all LongCoQA samples")

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
    query_vectors = _embed_with_vllm(
        model,
        [
            _query_prompt(rewrite_by_id[str(sample["sample_id"])], instruction)
            for sample in samples
        ],
        dimensions=dimensions,
    )
    del model
    gc.collect()
    rankings = _dense_rankings(
        query_vectors,
        document_vectors,
        document_ids,
        top_k=top_k,
    )
    all_rows = [
        {
            "sample_id": sample["sample_id"],
            "query_mode": "rewritten_dialogue",
            "query": rewrite_by_id[str(sample["sample_id"])],
            "original_query": sample["question"],
            "ranked": [
                {"child_id": child_id, "score": score}
                for child_id, score in ranked
            ],
            "method": "qwen3_embedding_rewritten",
        }
        for sample, ranked in zip(samples, rankings, strict=True)
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    all_path = output_dir / "all_results.jsonl"
    atomic_write_jsonl(all_path, all_rows)

    children = read_jsonl(children_path)
    alignments = read_jsonl(alignment_path)
    gold_queries = build_retrieval_queries(
        LongCoQADataset(), alignments, children, query_mode="current_question"
    )
    gold_by_id = {str(row["sample_id"]): row for row in gold_queries}
    eval_queries = []
    eval_rankings = []
    for row, ranked in zip(all_rows, rankings, strict=True):
        gold = gold_by_id.get(str(row["sample_id"]))
        if gold is None:
            continue
        eval_queries.append(
            {
                "sample_id": row["sample_id"],
                "query_mode": "rewritten_dialogue",
                "query": row["query"],
                "gold_child_ids": gold["gold_child_ids"],
            }
        )
        eval_rankings.append(ranked)
    results, summary = evaluate_rankings(
        eval_queries, eval_rankings, method="qwen3_embedding_rewritten"
    )
    results_path = output_dir / "results.jsonl"
    atomic_write_jsonl(results_path, results)
    summary.update(
        {
            "rewrites_checksum": sha256_file(rewrites_path),
            "dense_index_checksum": sha256_file(dense_index_path),
            "all_results_checksum": sha256_file(all_path),
            "artifact_checksum": sha256_file(results_path),
            "embedding_model_path": str(embedding_model_path),
        }
    )
    atomic_write_json(output_dir / "summary.json", summary)
    return summary

