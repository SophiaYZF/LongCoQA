from __future__ import annotations

import gc
import math
from pathlib import Path
from typing import Any

from longcoqa.retrieval.evaluate import evaluate_rankings
from longcoqa.utils.hashing import sha256_file, stable_json_hash
from longcoqa.utils.io import atomic_write_json, atomic_write_jsonl, read_jsonl

SYSTEM = (
    "Judge whether the Document meets the requirements based on the Query and "
    'the Instruct provided. Note that the answer can only be "yes" or "no".'
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def select_balanced_queries(
    rows: list[dict[str, Any]], limit: int | None
) -> list[dict[str, Any]]:
    if limit is None or limit >= len(rows):
        return rows
    factoid_count = limit // 2
    factoid = [row for row in rows if str(row["sample_id"]).startswith("factoid:")]
    non_factoid = [
        row for row in rows if str(row["sample_id"]).startswith("non_factoid:")
    ]
    return factoid[:factoid_count] + non_factoid[: limit - factoid_count]


def rerank_candidates(
    candidate_ids: list[str],
    scores: list[float],
    *,
    output_k: int,
) -> list[tuple[str, float]]:
    if len(candidate_ids) != len(scores):
        raise ValueError("candidate_ids and scores differ in size")
    return sorted(
        zip(candidate_ids, scores, strict=True),
        key=lambda item: (-item[1], item[0]),
    )[:output_k]


def _format_messages(instruction: str, query: str, document: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": (
                f"<Instruct>: {instruction}\n\n<Query>: {query}"
                f"\n\n<Document>: {document}"
            ),
        },
    ]


def run_qwen_reranker(
    *,
    model_path: Path,
    candidates_path: Path,
    children_path: Path,
    output_dir: Path,
    model_revision: str,
    instruction: str,
    candidate_k: int = 50,
    output_k: int = 20,
    limit: int | None = None,
    max_model_len: int = 4096,
    max_num_seqs: int = 512,
    gpu_memory_utilization: float = 0.90,
    batch_size: int = 2048,
) -> dict[str, Any]:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    rows = select_balanced_queries(read_jsonl(candidates_path), limit)
    children = {
        str(item["child_id"]): str(item["text"]) for item in read_jsonl(children_path)
    }
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    suffix_tokens = tokenizer.encode(SUFFIX, add_special_tokens=False)
    true_token = tokenizer("yes", add_special_tokens=False).input_ids[0]
    false_token = tokenizer("no", add_special_tokens=False).input_ids[0]
    sampling = SamplingParams(
        temperature=0,
        max_tokens=1,
        logprobs=20,
        allowed_token_ids=[true_token, false_token],
    )
    model = LLM(
        model=str(model_path),
        dtype="bfloat16",
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        enable_prefix_caching=True,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    pair_metadata: list[tuple[int, str]] = []
    prompts: list[Any] = []
    for row_index, row in enumerate(rows):
        for candidate in row["ranked"][:candidate_k]:
            child_id = str(candidate["child_id"])
            token_ids = tokenizer.apply_chat_template(
                _format_messages(instruction, str(row["query"]), children[child_id]),
                tokenize=True,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            token_ids = token_ids[: max_model_len - len(suffix_tokens)] + suffix_tokens
            prompts.append(TokensPrompt(prompt_token_ids=token_ids))
            pair_metadata.append((row_index, child_id))

    scores_by_row: list[dict[str, float]] = [dict() for _ in rows]
    for start in range(0, len(prompts), batch_size):
        outputs = model.generate(
            prompts[start : start + batch_size],
            sampling,
            use_tqdm=True,
        )
        for offset, output in enumerate(outputs):
            logprobs = output.outputs[0].logprobs[-1]
            true_logit = logprobs[true_token].logprob if true_token in logprobs else -10.0
            false_logit = (
                logprobs[false_token].logprob if false_token in logprobs else -10.0
            )
            true_score = math.exp(true_logit)
            false_score = math.exp(false_logit)
            score = true_score / (true_score + false_score)
            row_index, child_id = pair_metadata[start + offset]
            scores_by_row[row_index][child_id] = score

    del model
    gc.collect()

    rankings = [
        rerank_candidates(
            [str(item["child_id"]) for item in row["ranked"][:candidate_k]],
            [
                scores_by_row[row_index][str(item["child_id"])]
                for item in row["ranked"][:candidate_k]
            ],
            output_k=output_k,
        )
        for row_index, row in enumerate(rows)
    ]
    query_items = [
        {
            "sample_id": row["sample_id"],
            "query_mode": row["query_mode"],
            "query": row["query"],
            "gold_child_ids": row["gold_child_ids"],
        }
        for row in rows
    ]
    results, summary = evaluate_rankings(
        query_items,
        rankings,
        method="qwen3_reranker_4b",
    )
    for result, source in zip(results, rows, strict=True):
        result["candidate_method"] = source["method"]
        result["candidate_k"] = candidate_k
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    atomic_write_jsonl(results_path, results)
    summary.update(
        {
            "schema_version": "longcoqa-qwen-reranker-v1",
            "model_path": str(model_path),
            "model_revision": model_revision,
            "instruction": instruction,
            "candidate_k": candidate_k,
            "output_k": output_k,
            "input_checksum": sha256_file(candidates_path),
            "children_checksum": sha256_file(children_path),
            "artifact_checksum": sha256_file(results_path),
            "sample_ids_hash": stable_json_hash(
                [str(row["sample_id"]) for row in rows]
            ),
        }
    )
    atomic_write_json(output_dir / "summary.json", summary)
    return summary
