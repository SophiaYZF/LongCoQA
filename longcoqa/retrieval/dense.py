from __future__ import annotations

import gc
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from longcoqa.data import LongCoQADataset
from longcoqa.retrieval.bm25 import BM25Index
from longcoqa.retrieval.evaluate import (
    build_retrieval_queries,
    evaluate_rankings,
)
from longcoqa.utils.hashing import sha256_file, stable_json_hash
from longcoqa.utils.io import atomic_write_json, atomic_write_jsonl, read_jsonl

DEFAULT_QUERY_INSTRUCTION = (
    "Given a Chinese multi-turn question about a novel, retrieve passages that "
    "provide the evidence needed to answer it."
)


def _query_prompt(query: str, instruction: str) -> str:
    return f"Instruct: {instruction}\nQuery:{query}"


def _local_huggingface_revision(model_path: Path) -> str | None:
    metadata_path = model_path / ".cache/huggingface/download/config.json.metadata"
    if not metadata_path.is_file():
        return None
    first_line = metadata_path.read_text(encoding="utf-8").splitlines()[0].strip()
    return first_line or None


def _truncate_and_normalize(vectors: np.ndarray, dimensions: int) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if dimensions <= 0 or dimensions > vectors.shape[1]:
        raise ValueError(f"Requested dimensions={dimensions}, model output dimensions={vectors.shape[1]}")
    vectors = vectors[:, :dimensions]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def _embed_with_vllm(
    model: Any,
    texts: list[str],
    *,
    dimensions: int,
) -> np.ndarray:
    outputs = model.embed(texts, use_tqdm=True)
    vectors = np.asarray([output.outputs.embedding for output in outputs], dtype=np.float32)
    return _truncate_and_normalize(vectors, dimensions)


def _atomic_save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(temporary, **arrays)
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _dense_rankings(
    query_vectors: np.ndarray,
    document_vectors: np.ndarray,
    document_ids: list[str],
    *,
    top_k: int,
    device: str = "cuda",
    batch_size: int = 256,
) -> list[list[tuple[str, float]]]:
    import torch

    documents = torch.from_numpy(document_vectors).to(
        device=device,
        dtype=torch.float16,
    )
    rankings: list[list[tuple[str, float]]] = []
    for start in range(0, len(query_vectors), batch_size):
        queries = torch.from_numpy(query_vectors[start : start + batch_size]).to(
            device=device,
            dtype=torch.float16,
        )
        scores = queries @ documents.T
        values, indices = torch.topk(scores, k=min(top_k, scores.shape[1]), dim=1)
        for row_values, row_indices in zip(
            values.float().cpu().tolist(),
            indices.cpu().tolist(),
            strict=True,
        ):
            rankings.append(
                [
                    (document_ids[index], float(score))
                    for score, index in zip(row_values, row_indices, strict=True)
                ]
            )
        del queries, scores, values, indices
    del documents
    torch.cuda.empty_cache()
    return rankings


def reciprocal_rank_fusion(
    left: list[tuple[str, float]],
    right: list[tuple[str, float]],
    *,
    top_k: int = 20,
    rrf_k: int = 60,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in (left, right):
        for rank, (document_id, _) in enumerate(ranking, start=1):
            scores[document_id] = scores.get(document_id, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]


def run_dense_and_hybrid(
    *,
    model_path: Path,
    children_path: Path,
    alignment_path: Path,
    bm25_index_path: Path,
    index_path: Path,
    output_dir: Path,
    dimensions: int = 1024,
    gpu_memory_utilization: float = 0.90,
    max_model_len: int = 2048,
    max_num_seqs: int = 512,
    query_modes: tuple[str, ...] = ("current_question", "history_questions"),
    instruction: str = DEFAULT_QUERY_INSTRUCTION,
) -> dict[str, Any]:
    from vllm import LLM

    children = read_jsonl(children_path)
    alignments = read_jsonl(alignment_path)
    document_ids = [str(item["child_id"]) for item in children]
    document_texts = [str(item["text"]) for item in children]
    queries_by_mode = {
        mode: build_retrieval_queries(
            LongCoQADataset(),
            alignments,
            children,
            query_mode=mode,
        )
        for mode in query_modes
    }

    model = LLM(
        model=str(model_path),
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
        document_texts,
        dimensions=dimensions,
    )
    query_vectors_by_mode = {
        mode: _embed_with_vllm(
            model,
            [_query_prompt(str(item["query"]), instruction) for item in queries],
            dimensions=dimensions,
        )
        for mode, queries in queries_by_mode.items()
    }
    del model
    gc.collect()

    _atomic_save_npz(
        index_path,
        document_ids=np.asarray(document_ids),
        vectors=document_vectors.astype(np.float16),
    )
    index_metadata = {
        "schema_version": "longcoqa-qwen-dense-v1",
        "model_path": str(model_path),
        "model_revision": _local_huggingface_revision(model_path),
        "documents": len(document_ids),
        "dimensions": dimensions,
        "document_id_hash": stable_json_hash(document_ids),
        "children_checksum": sha256_file(children_path),
        "artifact_checksum": sha256_file(index_path),
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "max_num_seqs": max_num_seqs,
        "query_instruction": instruction,
    }
    atomic_write_json(index_path.with_suffix(".metadata.json"), index_metadata)

    bm25 = BM25Index.load(bm25_index_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, Any] = {"index": index_metadata}
    for mode, queries in queries_by_mode.items():
        dense_ranked = _dense_rankings(
            query_vectors_by_mode[mode],
            document_vectors,
            document_ids,
            top_k=100,
        )
        bm25_ranked = [bm25.search(str(query["query"]), top_k=100) for query in queries]
        hybrid_ranked = [
            reciprocal_rank_fusion(dense, sparse, top_k=100)
            for dense, sparse in zip(dense_ranked, bm25_ranked, strict=True)
        ]
        for method, rankings in (
            ("qwen3_embedding", dense_ranked),
            ("hybrid_rrf", hybrid_ranked),
        ):
            results, summary = evaluate_rankings(
                queries,
                rankings,
                method=method,
            )
            result_path = output_dir / f"{method}_{mode}.jsonl"
            atomic_write_jsonl(result_path, results)
            summary["artifact_checksum"] = sha256_file(result_path)
            atomic_write_json(
                output_dir / f"{method}_{mode}_summary.json",
                summary,
            )
            reports[f"{method}:{mode}"] = summary
    return reports
