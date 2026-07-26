from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from longcoqa.evaluation.paper_metric_profiles import MetricCompatibility
from longcoqa.utils.hashing import stable_json_hash
from longcoqa.utils.io import atomic_write_json, read_yaml


@dataclass(frozen=True)
class JudgeProfile:
    profile_id: str
    compatibility: MetricCompatibility
    scale_min: float
    scale_max: float
    score_field: str
    system_prompt: str
    transform: str
    scorer_model: str

    def __post_init__(self) -> None:
        for name in ("profile_id", "score_field", "system_prompt", "scorer_model"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty")
        if self.scale_min >= self.scale_max:
            raise ValueError("Judge score scale must have scale_min < scale_max")
        if self.transform not in {"identity", "linear_1_5_to_0_100"}:
            raise ValueError(f"Unsupported judge score transform: {self.transform}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "compatibility": self.compatibility.value,
            "scale_min": self.scale_min,
            "scale_max": self.scale_max,
            "score_field": self.score_field,
            "system_prompt": self.system_prompt,
            "transform": self.transform,
            "scorer_model": self.scorer_model,
        }

    @property
    def prompt_hash(self) -> str:
        return stable_json_hash({"system_prompt": self.system_prompt})

    @property
    def profile_hash(self) -> str:
        return stable_json_hash(self.to_dict())


def judge_profiles_from_mapping(raw: Mapping[str, Any]) -> dict[str, JudgeProfile]:
    values = raw.get("profiles")
    if not isinstance(values, list) or not values:
        raise ValueError("Judge config must contain a non-empty profiles list")
    profiles: dict[str, JudgeProfile] = {}
    for index, value in enumerate(values):
        if not isinstance(value, Mapping):
            raise ValueError(f"profiles[{index}] must be a mapping")
        profile = JudgeProfile(
            profile_id=str(value.get("profile_id", "")),
            compatibility=MetricCompatibility(str(value.get("compatibility", ""))),
            scale_min=float(value.get("scale_min", 0)),
            scale_max=float(value.get("scale_max", 0)),
            score_field=str(value.get("score_field", "")),
            system_prompt=str(value.get("system_prompt", "")),
            transform=str(value.get("transform", "")),
            scorer_model=str(value.get("scorer_model", "")),
        )
        if profile.profile_id in profiles:
            raise ValueError(f"Duplicate judge profile ID: {profile.profile_id}")
        profiles[profile.profile_id] = profile
    return profiles


def load_judge_profiles(path: Path) -> dict[str, JudgeProfile]:
    return judge_profiles_from_mapping(read_yaml(path))


def _json_object(text: str) -> Mapping[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as error:
        raise ValueError("Judge response must be a JSON object") from error
    if not isinstance(value, Mapping):
        raise ValueError("Judge response must be a JSON object")
    return value


def parse_judge_score(text: str, profile: JudgeProfile) -> dict[str, float]:
    value = _json_object(text).get(profile.score_field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Judge field {profile.score_field} must be numeric")
    raw_score = float(value)
    if not profile.scale_min <= raw_score <= profile.scale_max:
        raise ValueError(
            f"Judge score {raw_score} is outside range "
            f"[{profile.scale_min}, {profile.scale_max}]"
        )
    if profile.transform == "linear_1_5_to_0_100":
        score_100 = (raw_score - 1.0) * 25.0
    else:
        score_100 = (
            (raw_score - profile.scale_min)
            / (profile.scale_max - profile.scale_min)
            * 100.0
        )
    return {"raw_score": raw_score, "score_100": score_100}


def judge_messages(
    row: Mapping[str, Any], profile: JudgeProfile
) -> list[dict[str, str]]:
    visible = (
        f"Question type: {row.get('question_type', '')}\n"
        f"Question: {row.get('question', '')}\n"
        f"Reference: {row.get('reference', '')}\n"
        f"Candidate answer: {row.get('prediction', '')}\n"
        f"Evidence: {row.get('evidence_text', '')}\n"
        f"Return a JSON object with numeric field {profile.score_field!r}."
    )
    return [
        {"role": "system", "content": profile.system_prompt},
        {"role": "user", "content": visible},
    ]


def make_judge_evaluator(
    client: Any,
    profile: JudgeProfile,
    *,
    max_tokens: int = 256,
) -> Any:
    def evaluate(row: dict[str, Any]) -> dict[str, Any]:
        response = client.generate(
            judge_messages(row, profile),
            temperature=0.0,
            max_tokens=max_tokens,
        )
        return {
            "schema_version": "longcoqa-judge-score-v1",
            "profile_id": profile.profile_id,
            "profile_hash": profile.profile_hash,
            **parse_judge_score(response.text, profile),
        }

    return evaluate


_PUBLIC_FIELDS = (
    "presentation_id",
    "annotator_id",
    "question_type",
    "question",
    "reference",
    "prediction",
    "evidence_text",
    "correctness",
    "completeness",
    "faithfulness",
    "citation_correctness",
    "pairwise_choice",
)
_SCORE_FIELDS = (
    "correctness",
    "completeness",
    "faithfulness",
    "citation_correctness",
)


def _atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=_PUBLIC_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def export_human_annotation(
    rows: Sequence[Mapping[str, Any]],
    *,
    public_path: Path,
    key_path: Path,
    blind_salt: str,
) -> None:
    if not blind_salt:
        raise ValueError("blind_salt must be non-empty")
    public_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id:
            raise ValueError("Every annotation row requires sample_id")
        presentation_id = stable_json_hash(
            {"blind_salt": blind_salt, "sample_id": sample_id}
        )[:16]
        if presentation_id in seen:
            raise ValueError(f"Duplicate presentation_id for sample {sample_id}")
        seen.add(presentation_id)
        public_rows.append(
            {
                "presentation_id": presentation_id,
                "annotator_id": "",
                "question_type": row.get("question_type", ""),
                "question": row.get("question", ""),
                "reference": row.get("reference", ""),
                "prediction": row.get("prediction", ""),
                "evidence_text": row.get("evidence_text", ""),
                **{field: "" for field in _SCORE_FIELDS},
                "pairwise_choice": "",
            }
        )
        private_rows.append(
            {
                "presentation_id": presentation_id,
                "sample_id": sample_id,
                "model_variant": row.get("model_variant"),
                "method": row.get("method"),
            }
        )
    _atomic_write_csv(public_path, public_rows)
    atomic_write_json(
        key_path,
        {"schema_version": "longcoqa-human-blinding-key-v1", "rows": private_rows},
        mode=0o600,
    )


def import_human_annotations(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(_PUBLIC_FIELDS[:3] + _SCORE_FIELDS + ("pairwise_choice",)) - set(
            reader.fieldnames or ()
        )
        if missing:
            raise ValueError(f"Annotation CSV is missing columns: {sorted(missing)}")
        rows = list(reader)
    parsed: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, row in enumerate(rows, start=2):
        key = (row["presentation_id"].strip(), row["annotator_id"].strip())
        if not all(key):
            raise ValueError(f"Missing presentation_id or annotator_id at row {index}")
        if key in seen:
            raise ValueError(f"Duplicate annotation for {key[0]} and {key[1]}")
        seen.add(key)
        if row["question_type"] not in {"factoid", "non_factoid"}:
            raise ValueError("question_type must be factoid or non_factoid")
        if row["pairwise_choice"] not in {"", "A", "B", "TIE"}:
            raise ValueError("pairwise_choice must be empty, A, B, or TIE")
        value: dict[str, Any] = dict(row)
        for field in _SCORE_FIELDS:
            try:
                score = int(row[field])
            except (TypeError, ValueError) as error:
                raise ValueError(f"{field} must be an integer from 1 to 5") from error
            if not 1 <= score <= 5:
                raise ValueError(f"{field} must be in range 1 to 5")
            value[field] = score
        parsed.append(value)
    return parsed


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_delta = [value - left_mean for value in left]
    right_delta = [value - right_mean for value in right]
    denominator = math.sqrt(
        sum(value * value for value in left_delta)
        * sum(value * value for value in right_delta)
    )
    if denominator == 0:
        return None
    return sum(a * b for a, b in zip(left_delta, right_delta, strict=True)) / denominator


def _ranks(values: Sequence[float]) -> list[float]:
    ranks = [0.0] * len(values)
    ordered = sorted(range(len(values)), key=values.__getitem__)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        rank = (start + 1 + end) / 2
        for index in ordered[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def _mode(values: Sequence[str]) -> str | None:
    non_empty = [value for value in values if value]
    if not non_empty:
        return None
    counts = Counter(non_empty)
    best = counts.most_common()
    return best[0][0] if len(best) == 1 or best[0][1] > best[1][1] else None


def _cohen_kappa(rows: Sequence[Mapping[str, Any]]) -> float | None:
    annotators = sorted({str(row["annotator_id"]) for row in rows})
    if len(annotators) != 2:
        return None
    by_sample: dict[str, dict[str, int]] = defaultdict(dict)
    for row in rows:
        by_sample[str(row["presentation_id"])][str(row["annotator_id"])] = int(
            row["correctness"]
        )
    pairs = [
        (ratings[annotators[0]], ratings[annotators[1]])
        for ratings in by_sample.values()
        if all(annotator in ratings for annotator in annotators)
    ]
    if not pairs:
        return None
    observed = sum(left == right for left, right in pairs) / len(pairs)
    left_counts = Counter(left for left, _ in pairs)
    right_counts = Counter(right for _, right in pairs)
    expected = sum(
        left_counts[label] / len(pairs) * right_counts[label] / len(pairs)
        for label in set(left_counts) | set(right_counts)
    )
    if expected == 1:
        return 1.0 if observed == 1 else None
    return (observed - expected) / (1 - expected)


def calibrate_judges(
    judge_rows: Sequence[Mapping[str, Any]],
    human_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_sample: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in human_rows:
        by_sample[str(row["presentation_id"])].append(row)
    judge_by_sample = {str(row["presentation_id"]): row for row in judge_rows}
    common = sorted(set(by_sample) & set(judge_by_sample))
    judge_scores: list[float] = []
    human_scores: list[float] = []
    summaries: list[dict[str, Any]] = []
    pair_matches: list[bool] = []
    for presentation_id in common:
        humans = by_sample[presentation_id]
        mean_correctness = sum(float(row["correctness"]) for row in humans) / len(humans)
        human_score = (mean_correctness - 1.0) * 25.0
        judge = judge_by_sample[presentation_id]
        judge_score = float(judge["score_100"])
        judge_scores.append(judge_score)
        human_scores.append(human_score)
        human_choice = _mode([str(row.get("pairwise_choice", "")) for row in humans])
        judge_choice = str(judge.get("pairwise_choice", "")) or None
        if human_choice is not None and judge_choice is not None:
            pair_matches.append(human_choice == judge_choice)
        summaries.append(
            {
                "question_type": str(humans[0].get("question_type", "unknown")),
                "exact": round(judge_score / 25 + 1) == round(mean_correctness),
            }
        )

    pearson = _pearson(judge_scores, human_scores)
    spearman = _pearson(_ranks(judge_scores), _ranks(human_scores))
    by_type: dict[str, dict[str, Any]] = {}
    for question_type in sorted({row["question_type"] for row in summaries}):
        group = [row for row in summaries if row["question_type"] == question_type]
        by_type[question_type] = {
            "samples": len(group),
            "exact_agreement": sum(bool(row["exact"]) for row in group) / len(group),
        }
    report: dict[str, Any] = {
        "schema_version": "longcoqa-judge-calibration-v1",
        "samples": len(common),
        "pearson": pearson,
        "spearman": spearman,
        "pairwise_accuracy": (
            sum(pair_matches) / len(pair_matches) if pair_matches else None
        ),
        "inter_annotator": {"cohen_kappa": _cohen_kappa(human_rows)},
        "by_question_type": by_type,
    }
    if pearson is None:
        report["correlation_unavailable_reason"] = (
            "zero_variance"
            if len(common) >= 2
            and (len(set(judge_scores)) == 1 or len(set(human_scores)) == 1)
            else "insufficient_samples"
        )
    return report
