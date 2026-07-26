from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from longcoqa.utils.hashing import sha256_file, stable_json_hash
from longcoqa.utils.io import atomic_write_json, atomic_write_jsonl, read_jsonl


def normalize_alignment_text(text: str) -> str:
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", str(text)).lower()


def character_ngrams(text: str, n: int = 2) -> set[str]:
    normalized = normalize_alignment_text(text)
    if len(normalized) < n:
        return {normalized} if normalized else set()
    return {normalized[index : index + n] for index in range(len(normalized) - n + 1)}


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    return len(left & right) / max(1, len(left | right))


def released_evidence_anomaly(text: str) -> str | None:
    markers = ("改写后的段落", "改写段落", "相似度估计", "对齐状态")
    if sum(marker in str(text) for marker in markers) >= 2:
        return "generation_instruction_leakage"
    return None


def align_evidence(
    evidence: pd.DataFrame,
    legacy_units: list[dict[str, Any]],
    *,
    candidate_window: int = 96,
    high_similarity: float = 0.24,
    medium_similarity: float = 0.14,
    structural_min_similarity: float = 0.08,
    high_margin: float = 0.025,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if tuple(evidence.columns) != ("id", "value"):
        raise ValueError(f"Unexpected evidence columns: {tuple(evidence.columns)!r}")
    candidate_grams = [character_ngrams(item["text"]) for item in legacy_units]
    rows = list(evidence.itertuples(index=False))
    max_evidence_id = max(int(row.id) for row in rows)
    max_offset = min(candidate_window, len(legacy_units) - max_evidence_id)
    if max_offset < 0:
        raise ValueError("Legacy corpus has fewer units than the largest released evidence ID")

    # Evidence IDs preserve source order. The reconstructed corpus differs only
    # where notes and short dialogue lines were merged, so the candidate offset
    # must be small and non-decreasing through the seven books.
    state_scores: list[list[float]] = []
    for row in rows:
        evidence_id = int(row.id)
        query_grams = character_ngrams(str(row.value))
        state_scores.append(
            [
                jaccard_similarity(
                    query_grams,
                    candidate_grams[evidence_id + offset - 1],
                )
                for offset in range(max_offset + 1)
            ]
        )

    transition_penalty = 0.002
    dynamic: list[list[float]] = [[float("-inf")] * (max_offset + 1) for _ in rows]
    backpointers: list[list[int]] = [[-1] * (max_offset + 1) for _ in rows]
    for offset in range(max_offset + 1):
        dynamic[0][offset] = state_scores[0][offset] - 0.01 * offset
    for row_index in range(1, len(rows)):
        for offset in range(max_offset + 1):
            previous_offset = max(
                range(offset + 1),
                key=lambda value: (
                    dynamic[row_index - 1][value] - transition_penalty * (offset - value),
                    -value,
                ),
            )
            dynamic[row_index][offset] = (
                dynamic[row_index - 1][previous_offset]
                + state_scores[row_index][offset]
                - transition_penalty * (offset - previous_offset)
            )
            backpointers[row_index][offset] = previous_offset

    final_offset = max(
        range(max_offset + 1),
        key=lambda value: (dynamic[-1][value], -value),
    )
    selected_offsets = [0] * len(rows)
    selected_offsets[-1] = final_offset
    for row_index in range(len(rows) - 1, 0, -1):
        selected_offsets[row_index - 1] = backpointers[row_index][selected_offsets[row_index]]

    alignments: list[dict[str, Any]] = []
    for row_index, (row, position_offset) in enumerate(zip(rows, selected_offsets, strict=True)):
        evidence_id = int(row.id)
        best_index = evidence_id + position_offset - 1
        best_similarity = state_scores[row_index][position_offset]
        alternatives = sorted(state_scores[row_index], reverse=True)
        second_similarity = alternatives[1] if len(alternatives) > 1 else 0.0
        margin = best_similarity - second_similarity
        lexical_best_offset = max(
            range(max_offset + 1),
            key=lambda value: (state_scores[row_index][value], -value),
        )
        anomaly = released_evidence_anomaly(str(row.value))
        if anomaly:
            confidence = "unverifiable"
        elif best_similarity >= high_similarity and (
            margin >= high_margin or position_offset == lexical_best_offset
        ):
            confidence = "high"
        elif best_similarity >= medium_similarity or best_similarity >= structural_min_similarity:
            confidence = "medium"
        else:
            confidence = "low"
        candidate = legacy_units[best_index]
        alignments.append(
            {
                "evidence_id": evidence_id,
                "candidate_id": int(candidate["candidate_id"]),
                "book_id": int(candidate["book_id"]),
                "book_title": candidate["book_title"],
                "chapter_id": candidate["chapter_id"],
                "chapter_number": candidate["chapter_number"],
                "chapter_title": candidate["chapter_title"],
                "paragraph_ids": candidate["paragraph_ids"],
                "similarity": best_similarity,
                "runner_up_similarity": second_similarity,
                "margin": margin,
                "position_offset": position_offset,
                "lexical_best_offset": lexical_best_offset,
                "alignment_method": "monotonic_offset_dynamic_programming",
                "confidence": confidence,
                "released_evidence_anomaly": anomaly,
                "evidence_text": str(row.value),
                "source_text": candidate["text"],
            }
        )

    confidence_counts = Counter(item["confidence"] for item in alignments)
    offsets = Counter(item["position_offset"] for item in alignments)
    monotonic_violations = sum(
        current["candidate_id"] <= previous["candidate_id"]
        for previous, current in zip(alignments, alignments[1:], strict=False)
    )
    summary = {
        "schema_version": "longcoqa-evidence-alignment-v1",
        "counts": {
            "evidence": len(alignments),
            "confidence": dict(sorted(confidence_counts.items())),
            "monotonic_violations": monotonic_violations,
        },
        "similarity": {
            "mean": sum(item["similarity"] for item in alignments) / len(alignments),
            "min": min(item["similarity"] for item in alignments),
            "max": max(item["similarity"] for item in alignments),
        },
        "position_offset_modes": [
            {"offset": offset, "count": count} for offset, count in offsets.most_common(20)
        ],
        "alignment_content_hash": stable_json_hash(
            [
                {
                    "evidence_id": item["evidence_id"],
                    "candidate_id": item["candidate_id"],
                    "confidence": item["confidence"],
                }
                for item in alignments
            ]
        ),
    }
    return alignments, summary


def write_evidence_alignment(
    evidence_path: Path,
    legacy_units_path: Path,
    output_dir: Path,
    **kwargs: Any,
) -> dict[str, Any]:
    evidence = pd.read_csv(evidence_path, encoding="utf-8-sig")
    legacy_units = read_jsonl(legacy_units_path)
    alignments, summary = align_evidence(evidence, legacy_units, **kwargs)
    output_dir.mkdir(parents=True, exist_ok=True)
    alignment_path = output_dir / "evidence_alignment.jsonl"
    atomic_write_jsonl(alignment_path, alignments)
    summary["source_checksums"] = {
        "evidence": sha256_file(evidence_path),
        "legacy_units": sha256_file(legacy_units_path),
    }
    summary["artifact_checksum"] = sha256_file(alignment_path)
    atomic_write_json(output_dir / "alignment_report.json", summary)
    return summary
