from __future__ import annotations

from collections import defaultdict
from typing import Any

SUPPORTED_PROTOCOLS = {
    "z0_current",
    "z1_teacher_forced",
    "g0_released_gold",
    "g3_aligned_raw",
}

SYSTEM_PROMPT = """你是《哈利·波特》系列小说问答助手。
请直接用中文回答。事实型问题应简洁准确；解释、情感和逻辑型问题应覆盖问题要求的各个方面。
不要声称看到了未提供的证据，不要编造人物、事件或因果关系。只输出最终答案。"""


def index_samples(samples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    dialogues: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        key = f"{sample['question_type']}:{sample['dialogue_id']}"
        dialogues[key].append(sample)
    for dialogue in dialogues.values():
        dialogue.sort(key=lambda item: int(item["turn"]))
    return dict(dialogues)


def build_messages(
    sample: dict[str, Any],
    *,
    protocol: str,
    dialogues: dict[str, list[dict[str, Any]]],
    released_evidence: dict[int, str],
    aligned_source: dict[int, str],
) -> list[dict[str, str]]:
    if protocol not in SUPPORTED_PROTOCOLS:
        raise ValueError(f"Unsupported protocol={protocol!r}")
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    dialogue_key = f"{sample['question_type']}:{sample['dialogue_id']}"
    if protocol != "z0_current":
        for prior in dialogues[dialogue_key]:
            if int(prior["turn"]) >= int(sample["turn"]):
                break
            messages.extend(
                [
                    {"role": "user", "content": str(prior["question"])},
                    {"role": "assistant", "content": str(prior["answer"])},
                ]
            )

    question = str(sample["question"])
    if protocol in {"g0_released_gold", "g3_aligned_raw"}:
        source = released_evidence if protocol == "g0_released_gold" else aligned_source
        evidence_blocks = [
            f"[EVIDENCE {evidence_id}]\n{source[evidence_id]}"
            for evidence_id in sample["evidence_ids"]
            if evidence_id in source
        ]
        context = "\n\n".join(evidence_blocks)
        question = (
            f"请依据下面的证据回答问题。若证据不足，请明确指出不足。\n\n{context}\n\n[QUESTION]\n{question}"
        )
    messages.append({"role": "user", "content": question})
    return messages
