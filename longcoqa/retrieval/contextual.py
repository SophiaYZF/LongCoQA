from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import numpy as np

from longcoqa.data import LongCoQADataset
from longcoqa.retrieval.bm25 import BM25Index
from longcoqa.retrieval.dense import (
    DEFAULT_QUERY_INSTRUCTION,
    _atomic_save_npz,
    _dense_rankings,
    _embed_with_vllm,
    _query_prompt,
    reciprocal_rank_fusion,
)
from longcoqa.retrieval.evaluate import (
    build_all_ranking_results,
    build_all_retrieval_queries,
    build_retrieval_queries,
    evaluate_rankings,
    project_rankings_for_evaluation,
)
from longcoqa.utils.hashing import sha256_file, stable_json_hash
from longcoqa.utils.io import atomic_write_json, atomic_write_jsonl, read_jsonl

CONTEXT_SYSTEM = """你是检索语料预处理器。根据完整段落和其中的文本块，生成一段50至100个中文字符的
简短上下文，说明该文本块在小说中的人物、地点、事件和前因后果。只使用给定文本，不推测，不回答任何
问题，只输出上下文本身。"""


def build_context_messages(
    children: list[dict[str, Any]], parents: list[dict[str, Any]]
) -> list[list[dict[str, str]]]:
    parent_by_id = {str(row["parent_id"]): row for row in parents}
    messages = []
    for child in children:
        parent = parent_by_id[str(child["parent_id"])]
        document = (
            f"书名：{parent['book_title']}\n章节：{parent['chapter_title']}\n"
            f"{parent['text']}"
        )
        messages.append(
            [
                {"role": "system", "content": CONTEXT_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"<document>\n{document}\n</document>\n"
                        f"<chunk>\n{child['text']}\n</chunk>\n"
                        "请给出能帮助检索定位该文本块的简短上下文。"
                    ),
                },
            ]
        )
    return messages


def generate_contextual_corpus(
    *,
    model_path: Path,
    children_path: Path,
    parents_path: Path,
    output_path: Path,
    model_revision: str,
    limit: int | None = None,
    batch_size: int = 512,
    gpu_memory_utilization: float = 0.90,
    max_model_len: int = 4096,
    max_num_seqs: int = 256,
) -> dict[str, Any]:
    from vllm import LLM, SamplingParams

    children = read_jsonl(children_path)
    if limit is not None:
        children = children[:limit]
    parents = read_jsonl(parents_path)
    messages = build_context_messages(children, parents)
    checkpoint_dir = output_path.parent / f".{output_path.stem}.checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    pending = [
        index
        for index, child in enumerate(children)
        if not (checkpoint_dir / f"{child['child_id']}.json").is_file()
    ]
    if pending:
        model = LLM(
            model=str(model_path),
            dtype="bfloat16",
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            enable_prefix_caching=True,
        )
        params = SamplingParams(temperature=0, max_tokens=128)
        for start in range(0, len(pending), batch_size):
            indices = pending[start : start + batch_size]
            outputs = model.chat(
                [messages[index] for index in indices],
                sampling_params=params,
                chat_template_kwargs={"enable_thinking": False},
                use_tqdm=True,
            )
            for index, output in zip(indices, outputs, strict=True):
                child = children[index]
                context = output.outputs[0].text.strip().strip('"“”')
                atomic_write_json(
                    checkpoint_dir / f"{child['child_id']}.json",
                    {"child_id": child["child_id"], "context": context},
                )
        del model
        gc.collect()

    rows = []
    for child in children:
        checkpoint = json.loads(
            (checkpoint_dir / f"{child['child_id']}.json").read_text(encoding="utf-8")
        )
        rows.append(
            {
                **child,
                "original_text": child["text"],
                "context": checkpoint["context"],
                "text": f"{checkpoint['context']}\n{child['text']}",
            }
        )
    atomic_write_jsonl(output_path, rows)
    summary = {
        "schema_version": "longcoqa-contextual-corpus-v1",
        "method": "anthropic_contextual_retrieval",
        "model_path": str(model_path),
        "model_revision": model_revision,
        "documents": len(rows),
        "complete_corpus": len(rows) == len(read_jsonl(children_path)),
        "document_ids_hash": stable_json_hash([row["child_id"] for row in rows]),
        "artifact_checksum": sha256_file(output_path),
    }
    atomic_write_json(output_path.with_suffix(".summary.json"), summary)
    return summary


def run_contextual_retrieval(
    *,
    embedding_model_path: Path,
    contextual_children_path: Path,
    original_children_path: Path,
    alignment_path: Path,
    rewrites_path: Path,
    output_dir: Path,
    index_path: Path,
    bm25_index_path: Path,
    dimensions: int = 1024,
    top_k: int = 100,
    gpu_memory_utilization: float = 0.90,
    max_model_len: int = 2048,
    max_num_seqs: int = 512,
) -> dict[str, Any]:
    from vllm import LLM

    contextual = read_jsonl(contextual_children_path)
    original = read_jsonl(original_children_path)
    if [row["child_id"] for row in contextual] != [row["child_id"] for row in original]:
        raise ValueError("Contextual corpus must contain every child in original order")
    alignments = read_jsonl(alignment_path)
    evaluation_current_queries = build_retrieval_queries(
        LongCoQADataset(), alignments, original, query_mode="current_question"
    )
    rewrites = {str(row["sample_id"]): str(row["rewritten_query"]) for row in read_jsonl(rewrites_path)}
    all_query_sets = {
        "current_question": build_all_retrieval_queries(
            LongCoQADataset(), query_mode="current_question"
        ),
        "rewritten_dialogue": build_all_retrieval_queries(
            LongCoQADataset(), query_mode="rewritten_dialogue", rewrites=rewrites
        ),
    }
    evaluation_query_sets = {
        "current_question": evaluation_current_queries,
        "rewritten_dialogue": [
            {
                **row,
                "query_mode": "rewritten_dialogue",
                "query": rewrites[str(row["sample_id"])],
            }
            for row in evaluation_current_queries
        ],
    }
    document_ids = [str(row["child_id"]) for row in contextual]
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
    document_vectors = _embed_with_vllm(
        model,
        [str(row["text"]) for row in contextual],
        dimensions=dimensions,
    )
    query_vectors = {
        name: _embed_with_vllm(
            model,
            [_query_prompt(str(row["query"]), DEFAULT_QUERY_INSTRUCTION) for row in rows],
            dimensions=dimensions,
        )
        for name, rows in all_query_sets.items()
    }
    del model
    gc.collect()
    _atomic_save_npz(
        index_path,
        document_ids=np.asarray(document_ids),
        vectors=document_vectors.astype(np.float16),
    )
    bm25 = BM25Index.build(contextual)
    bm25.save(bm25_index_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, Any] = {}
    for query_mode, all_queries in all_query_sets.items():
        dense = _dense_rankings(query_vectors[query_mode], document_vectors, document_ids, top_k=top_k)
        sparse = [bm25.search(str(row["query"]), top_k=top_k) for row in all_queries]
        fused = [
            reciprocal_rank_fusion(left, right, top_k=top_k)
            for left, right in zip(dense, sparse, strict=True)
        ]
        for method, rankings in (
            ("contextual_bm25", sparse),
            ("contextual_dense", dense),
            ("contextual_rrf", fused),
        ):
            all_results = build_all_ranking_results(
                all_queries, rankings, method=method
            )
            all_path = output_dir / f"{method}_{query_mode}_all_results.jsonl"
            atomic_write_jsonl(all_path, all_results)
            evaluation_queries = evaluation_query_sets[query_mode]
            evaluation_rankings = project_rankings_for_evaluation(
                all_results, evaluation_queries
            )
            results, summary = evaluate_rankings(
                evaluation_queries, evaluation_rankings, method=method
            )
            path = output_dir / f"{method}_{query_mode}.jsonl"
            atomic_write_jsonl(path, results)
            summary["artifact_checksum"] = sha256_file(path)
            summary["all_queries"] = len(all_results)
            summary["all_results_checksum"] = sha256_file(all_path)
            atomic_write_json(output_dir / f"{method}_{query_mode}_summary.json", summary)
            reports[f"{method}:{query_mode}"] = summary
    return reports
