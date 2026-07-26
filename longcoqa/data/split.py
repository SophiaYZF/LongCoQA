from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from longcoqa.data.loader import Dialogue

SPLITS = ("train", "dev", "test")


@dataclass(frozen=True)
class SplitResult:
    assignments: dict[str, str]
    objective: float
    diagnostics: dict


def _allocate_counts(size: int, ratios: dict[str, float]) -> dict[str, int]:
    raw = {split: size * ratios[split] for split in SPLITS}
    counts = {split: math.floor(raw[split]) for split in SPLITS}
    remaining = size - sum(counts.values())
    order = sorted(SPLITS, key=lambda split: (raw[split] - counts[split], ratios[split]), reverse=True)
    for split in order[:remaining]:
        counts[split] += 1
    return counts


def _initial_assignment(
    dialogues: list[Dialogue], ratios: dict[str, float], random_state: random.Random
) -> dict[str, str]:
    by_stratum: dict[str, list[Dialogue]] = defaultdict(list)
    for dialogue in dialogues:
        by_stratum[dialogue.stratum].append(dialogue)

    assignments: dict[str, str] = {}
    for stratum in sorted(by_stratum):
        group = list(by_stratum[stratum])
        random_state.shuffle(group)
        quotas = _allocate_counts(len(group), ratios)
        cursor = 0
        for split in SPLITS:
            for dialogue in group[cursor : cursor + quotas[split]]:
                assignments[dialogue.key] = split
            cursor += quotas[split]
    return assignments


def _build_objective_terms(
    dialogues: Iterable[Dialogue],
) -> tuple[dict[int, list[str]], dict[str, list[str]]]:
    evidence_owners: dict[int, list[str]] = defaultdict(list)
    question_owners: dict[str, list[str]] = defaultdict(list)
    for dialogue in dialogues:
        for evidence_id in dialogue.evidence_ids:
            evidence_owners[evidence_id].append(dialogue.key)
        for question in dialogue.normalized_questions:
            question_owners[question].append(dialogue.key)
    return evidence_owners, question_owners


def _cut_cost(owners: Iterable[str], assignments: dict[str, str], weight: float) -> float:
    counts = Counter(assignments[owner] for owner in owners)
    total = sum(counts.values())
    same_pairs = sum(value * (value - 1) / 2 for value in counts.values())
    all_pairs = total * (total - 1) / 2
    return weight * (all_pairs - same_pairs)


def _objective(
    assignments: dict[str, str],
    evidence_owners: dict[int, list[str]],
    question_owners: dict[str, list[str]],
    evidence_cut_weight: float,
    duplicate_question_penalty: float,
) -> float:
    score = 0.0
    for owners in evidence_owners.values():
        if len(owners) > 1:
            frequency_weight = evidence_cut_weight / math.log2(2 + len(owners))
            score += _cut_cost(owners, assignments, frequency_weight)
    for owners in question_owners.values():
        unique_owners = sorted(set(owners))
        if len(unique_owners) > 1:
            score += _cut_cost(unique_owners, assignments, duplicate_question_penalty)
    return score


def _local_objective(
    assignments: dict[str, str],
    evidence_ids: Iterable[int],
    questions: Iterable[str],
    evidence_owners: dict[int, list[str]],
    question_owners: dict[str, list[str]],
    evidence_cut_weight: float,
    duplicate_question_penalty: float,
) -> float:
    score = 0.0
    for evidence_id in set(evidence_ids):
        owners = evidence_owners[evidence_id]
        if len(owners) > 1:
            frequency_weight = evidence_cut_weight / math.log2(2 + len(owners))
            score += _cut_cost(owners, assignments, frequency_weight)
    for question in set(questions):
        unique_owners = sorted(set(question_owners[question]))
        if len(unique_owners) > 1:
            score += _cut_cost(unique_owners, assignments, duplicate_question_penalty)
    return score


def _diagnostics(
    dialogues: list[Dialogue],
    assignments: dict[str, str],
    evidence_owners: dict[int, list[str]],
    question_owners: dict[str, list[str]],
) -> dict:
    by_key = {dialogue.key: dialogue for dialogue in dialogues}
    split_evidence: dict[str, set[int]] = {split: set() for split in SPLITS}
    split_questions: dict[str, int] = Counter()
    split_dialogues: dict[str, int] = Counter()
    split_strata: dict[str, Counter] = {split: Counter() for split in SPLITS}
    for key, split in assignments.items():
        dialogue = by_key[key]
        split_evidence[split].update(dialogue.evidence_ids)
        split_questions[split] += dialogue.turns
        split_dialogues[split] += 1
        split_strata[split][dialogue.stratum] += 1

    cross_evidence = {
        evidence_id: sorted({assignments[key] for key in owners})
        for evidence_id, owners in evidence_owners.items()
        if len({assignments[key] for key in owners}) > 1
    }
    duplicate_question_crossings = {
        question: sorted({assignments[key] for key in set(owners)})
        for question, owners in question_owners.items()
        if len(set(owners)) > 1 and len({assignments[key] for key in set(owners)}) > 1
    }
    return {
        "dialogues": dict(split_dialogues),
        "questions": dict(split_questions),
        "strata": {split: dict(values) for split, values in split_strata.items()},
        "unique_evidence": {split: len(values) for split, values in split_evidence.items()},
        "cross_split_evidence_ids": len(cross_evidence),
        "cross_split_evidence_examples": dict(list(sorted(cross_evidence.items()))[:20]),
        "duplicate_question_crossings": duplicate_question_crossings,
        "evidence_pairwise_overlap": {
            "train_dev": len(split_evidence["train"] & split_evidence["dev"]),
            "train_test": len(split_evidence["train"] & split_evidence["test"]),
            "dev_test": len(split_evidence["dev"] & split_evidence["test"]),
            "all_three": len(split_evidence["train"] & split_evidence["dev"] & split_evidence["test"]),
        },
    }


def _repair_duplicate_questions(
    dialogues: list[Dialogue],
    assignments: dict[str, str],
    evidence_owners: dict[int, list[str]],
    question_owners: dict[str, list[str]],
    evidence_cut_weight: float,
    duplicate_question_penalty: float,
) -> None:
    by_key = {dialogue.key: dialogue for dialogue in dialogues}
    split_order = {split: index for index, split in enumerate(SPLITS)}

    for _ in range(100):
        crossings = [
            (question, sorted(set(owners)))
            for question, owners in sorted(question_owners.items())
            if len(set(owners)) > 1 and len({assignments[key] for key in set(owners)}) > 1
        ]
        if not crossings:
            return

        _, owners = crossings[0]
        split_counts = Counter(assignments[key] for key in owners)
        target_split = min(
            split_counts,
            key=lambda split: (-split_counts[split], split_order[split]),
        )
        outsider = next(key for key in owners if assignments[key] != target_split)
        outsider_split = assignments[outsider]
        stratum = by_key[outsider].stratum

        candidates = [
            dialogue.key
            for dialogue in dialogues
            if dialogue.stratum == stratum
            and assignments[dialogue.key] == target_split
            and all(len(set(question_owners[q])) == 1 for q in dialogue.normalized_questions)
        ]
        best_candidate: str | None = None
        best_delta = math.inf
        for candidate in candidates:
            affected_evidence = by_key[outsider].evidence_ids | by_key[candidate].evidence_ids
            affected_questions = set(by_key[outsider].normalized_questions) | set(
                by_key[candidate].normalized_questions
            )
            before = _local_objective(
                assignments,
                affected_evidence,
                affected_questions,
                evidence_owners,
                question_owners,
                evidence_cut_weight,
                duplicate_question_penalty,
            )
            assignments[outsider], assignments[candidate] = target_split, outsider_split
            remaining_crossings = any(
                len({assignments[key] for key in set(question_owners[q])}) > 1
                for q in affected_questions
                if len(set(question_owners[q])) > 1
            )
            after = _local_objective(
                assignments,
                affected_evidence,
                affected_questions,
                evidence_owners,
                question_owners,
                evidence_cut_weight,
                duplicate_question_penalty,
            )
            assignments[outsider], assignments[candidate] = outsider_split, target_split
            if not remaining_crossings and after - before < best_delta:
                best_delta = after - before
                best_candidate = candidate

        if best_candidate is None:
            raise RuntimeError(f"Could not repair duplicate-question split for owners={owners}")
        assignments[outsider], assignments[best_candidate] = target_split, outsider_split

    raise RuntimeError("Duplicate-question repair exceeded 100 iterations")


def optimize_dialogue_split(
    dialogues: list[Dialogue],
    *,
    ratios: dict[str, float] | None = None,
    seed: int = 42,
    restarts: int = 24,
    local_steps: int = 4000,
    evidence_cut_weight: float = 1.0,
    duplicate_question_penalty: float = 1000.0,
) -> SplitResult:
    ratios = ratios or {"train": 0.70, "dev": 0.10, "test": 0.20}
    if set(ratios) != set(SPLITS) or not math.isclose(sum(ratios.values()), 1.0):
        raise ValueError(f"Invalid split ratios: {ratios}")

    by_key = {dialogue.key: dialogue for dialogue in dialogues}
    evidence_owners, question_owners = _build_objective_terms(dialogues)
    best_assignment: dict[str, str] | None = None
    best_score = math.inf

    for restart in range(restarts):
        random_state = random.Random(seed + 1009 * restart)
        assignments = _initial_assignment(dialogues, ratios, random_state)
        score = _objective(
            assignments,
            evidence_owners,
            question_owners,
            evidence_cut_weight,
            duplicate_question_penalty,
        )

        by_stratum: dict[str, list[str]] = defaultdict(list)
        for dialogue in dialogues:
            by_stratum[dialogue.stratum].append(dialogue.key)

        for _ in range(local_steps):
            stratum = random_state.choice(sorted(by_stratum))
            candidates = by_stratum[stratum]
            left, right = random_state.sample(candidates, 2)
            if assignments[left] == assignments[right]:
                continue
            affected_evidence = by_key[left].evidence_ids | by_key[right].evidence_ids
            affected_questions = set(by_key[left].normalized_questions) | set(
                by_key[right].normalized_questions
            )
            before = _local_objective(
                assignments,
                affected_evidence,
                affected_questions,
                evidence_owners,
                question_owners,
                evidence_cut_weight,
                duplicate_question_penalty,
            )
            assignments[left], assignments[right] = assignments[right], assignments[left]
            after = _local_objective(
                assignments,
                affected_evidence,
                affected_questions,
                evidence_owners,
                question_owners,
                evidence_cut_weight,
                duplicate_question_penalty,
            )
            candidate_score = score - before + after
            if candidate_score <= score:
                score = candidate_score
            else:
                assignments[left], assignments[right] = assignments[right], assignments[left]

        if score < best_score:
            best_score = score
            best_assignment = dict(assignments)

    assert best_assignment is not None
    _repair_duplicate_questions(
        dialogues,
        best_assignment,
        evidence_owners,
        question_owners,
        evidence_cut_weight,
        duplicate_question_penalty,
    )
    best_score = _objective(
        best_assignment,
        evidence_owners,
        question_owners,
        evidence_cut_weight,
        duplicate_question_penalty,
    )
    diagnostics = _diagnostics(dialogues, best_assignment, evidence_owners, question_owners)
    if diagnostics["duplicate_question_crossings"]:
        raise RuntimeError(
            "Could not keep normalized duplicate questions in one split; "
            f"crossings={diagnostics['duplicate_question_crossings']}"
        )
    if set(best_assignment) != set(by_key):
        raise RuntimeError("Split optimizer did not assign every dialogue")
    return SplitResult(best_assignment, best_score, diagnostics)
