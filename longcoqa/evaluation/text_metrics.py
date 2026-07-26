from __future__ import annotations

import re
from collections import Counter
from typing import Any


def normalize_answer(text: str) -> str:
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", str(text)).lower()


def character_f1(prediction: str, reference: str) -> float:
    prediction_counts = Counter(normalize_answer(prediction))
    reference_counts = Counter(normalize_answer(reference))
    overlap = sum((prediction_counts & reference_counts).values())
    prediction_total = sum(prediction_counts.values())
    reference_total = sum(reference_counts.values())
    if not prediction_total or not reference_total:
        return float(prediction_total == reference_total)
    precision = overlap / prediction_total
    recall = overlap / reference_total
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def lcs_f1(prediction: str, reference: str) -> float:
    left = normalize_answer(prediction)
    right = normalize_answer(reference)
    if not left or not right:
        return float(left == right)
    previous = [0] * (len(right) + 1)
    for left_character in left:
        current = [0]
        for index, right_character in enumerate(right, start=1):
            if left_character == right_character:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    longest = previous[-1]
    precision = longest / len(left)
    recall = longest / len(right)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def score_answer(prediction: str, reference: str) -> dict[str, Any]:
    normalized_prediction = normalize_answer(prediction)
    normalized_reference = normalize_answer(reference)
    return {
        "exact_match": float(normalized_prediction == normalized_reference),
        "character_f1": character_f1(prediction, reference),
        "rouge_l_f1": lcs_f1(prediction, reference),
        "prediction_characters": len(str(prediction)),
        "reference_characters": len(str(reference)),
    }
