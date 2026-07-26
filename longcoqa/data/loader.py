from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from longcoqa.paths import DEFAULT_DATA_DIR
from longcoqa.utils.hashing import sha256_file

EVIDENCE_COLUMNS = ("evidence1", "evidence2", "evidence3", "evidence4")
FACTOID_COLUMNS = ("id", "num", "question", "answer", *EVIDENCE_COLUMNS)
NON_FACTOID_COLUMNS = (
    "id",
    "abstract",
    "style",
    "num",
    "question",
    "answer",
    *EVIDENCE_COLUMNS,
)
EVIDENCE_TABLE_COLUMNS = ("id", "value")


class DatasetValidationError(ValueError):
    """Raised when a released dataset invariant is violated."""


@dataclass(frozen=True)
class Dialogue:
    key: str
    question_type: str
    dialogue_id: int
    stratum: str
    style: str | None
    turns: int
    sample_ids: tuple[str, ...]
    evidence_ids: frozenset[int]
    normalized_questions: tuple[str, ...]


def normalize_question(value: str) -> str:
    return "".join(str(value).strip().lower().split())


def _require_columns(frame: pd.DataFrame, expected: Iterable[str], path: Path) -> None:
    expected_tuple = tuple(expected)
    actual_tuple = tuple(frame.columns)
    if actual_tuple != expected_tuple:
        raise DatasetValidationError(
            f"Unexpected columns in {path}: expected={expected_tuple!r}, actual={actual_tuple!r}"
        )


class LongCoQADataset:
    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR) -> None:
        self.data_dir = Path(data_dir)
        self.factoid_path = self.data_dir / "LongCoQA_factoid.csv"
        self.non_factoid_path = self.data_dir / "LongCoQA_non_factoid.csv"
        self.evidence_path = self.data_dir / "LongCoQA_evidence.csv"
        self.novel_dir = self.data_dir / "novel"

        self.factoid = pd.read_csv(self.factoid_path, encoding="utf-8-sig")
        self.non_factoid = pd.read_csv(self.non_factoid_path, encoding="utf-8-sig")
        self.evidence = pd.read_csv(self.evidence_path, encoding="utf-8-sig")

    def source_checksums(self) -> dict[str, str]:
        paths = [self.factoid_path, self.non_factoid_path, self.evidence_path]
        paths.extend(sorted(self.novel_dir.glob("*.txt")))
        return {str(path.relative_to(self.data_dir.parent)): sha256_file(path) for path in paths}

    def validate(self) -> dict[str, Any]:
        _require_columns(self.factoid, FACTOID_COLUMNS, self.factoid_path)
        _require_columns(self.non_factoid, NON_FACTOID_COLUMNS, self.non_factoid_path)
        _require_columns(self.evidence, EVIDENCE_TABLE_COLUMNS, self.evidence_path)

        failures: list[str] = []
        if self.evidence["id"].duplicated().any():
            failures.append("evidence IDs are not unique")
        if self.evidence[["id", "value"]].isna().any(axis=None):
            failures.append("evidence table contains missing id/value")

        evidence_ids = set(self.evidence["id"].astype(int))
        referenced: list[int] = []
        summary: dict[str, Any] = {}

        for question_type, frame in (("factoid", self.factoid), ("non_factoid", self.non_factoid)):
            missing_qa = int(frame[["question", "answer"]].isna().any(axis=1).sum())
            if missing_qa:
                failures.append(f"{question_type} has {missing_qa} rows with missing question/answer")

            invalid_turn_dialogues: list[int] = []
            for dialogue_id, group in frame.groupby("id", sort=True):
                actual = list(group["num"].astype(int))
                expected = list(range(1, len(group) + 1))
                if actual != expected:
                    invalid_turn_dialogues.append(int(dialogue_id))
            if invalid_turn_dialogues:
                failures.append(
                    f"{question_type} has non-contiguous turns in dialogue IDs {invalid_turn_dialogues[:20]}"
                )

            refs = [int(value) for column in EVIDENCE_COLUMNS for value in frame[column].dropna().tolist()]
            referenced.extend(refs)
            missing_refs = set(refs) - evidence_ids
            if missing_refs:
                failures.append(f"{question_type} references absent evidence IDs {sorted(missing_refs)[:20]}")

            evidence_count = frame[list(EVIDENCE_COLUMNS)].notna().sum(axis=1)
            summary[question_type] = {
                "dialogues": int(frame["id"].nunique()),
                "questions": len(frame),
                "turns_min": int(frame.groupby("id").size().min()),
                "turns_max": int(frame.groupby("id").size().max()),
                "turns_mean": float(frame.groupby("id").size().mean()),
                "evidence_mentions": len(refs),
                "evidence_per_question_mean": float(evidence_count.mean()),
                "zero_evidence_questions": int((evidence_count == 0).sum()),
                "duplicate_question_rows": int(frame["question"].duplicated().sum()),
            }

        unreferenced = evidence_ids - set(referenced)
        if unreferenced:
            failures.append(f"{len(unreferenced)} evidence rows are never referenced")

        novel_stats: dict[str, Any] = {}
        for path in sorted(self.novel_dir.glob("*.txt")):
            text = path.read_text(encoding="gb18030")
            novel_stats[path.name] = {
                "characters": len(text),
                "lines": len(text.splitlines()),
                "sha256": sha256_file(path),
            }

        if failures:
            raise DatasetValidationError("; ".join(failures))

        return {
            "status": "valid",
            "factoid": summary["factoid"],
            "non_factoid": summary["non_factoid"],
            "evidence": {
                "rows": len(self.evidence),
                "unique_ids": len(evidence_ids),
                "id_min": min(evidence_ids),
                "id_max": max(evidence_ids),
                "total_reference_mentions": len(referenced),
            },
            "novels": novel_stats,
            "source_checksums": self.source_checksums(),
        }

    def iter_samples(self) -> Iterable[dict[str, Any]]:
        for question_type, frame in (("factoid", self.factoid), ("non_factoid", self.non_factoid)):
            for row in frame.itertuples(index=False):
                sample_id = f"{question_type}:{int(row.id)}:{int(row.num)}"
                evidence_ids = [
                    int(value)
                    for value in (row.evidence1, row.evidence2, row.evidence3, row.evidence4)
                    if pd.notna(value)
                ]
                yield {
                    "sample_id": sample_id,
                    "question_type": question_type,
                    "dialogue_id": int(row.id),
                    "turn": int(row.num),
                    "question": str(row.question),
                    "answer": str(row.answer),
                    "evidence_ids": evidence_ids,
                    "abstract": str(row.abstract) if question_type == "non_factoid" else None,
                    "style": str(row.style) if question_type == "non_factoid" else None,
                }

    def dialogues(self) -> list[Dialogue]:
        dialogues: list[Dialogue] = []
        for question_type, frame in (("factoid", self.factoid), ("non_factoid", self.non_factoid)):
            for dialogue_id, group in frame.groupby("id", sort=True):
                refs = {
                    int(value) for column in EVIDENCE_COLUMNS for value in group[column].dropna().tolist()
                }
                style = str(group.iloc[0]["style"]) if question_type == "non_factoid" else None
                stratum = question_type if style is None else f"{question_type}:{style}"
                dialogues.append(
                    Dialogue(
                        key=f"{question_type}:{int(dialogue_id)}",
                        question_type=question_type,
                        dialogue_id=int(dialogue_id),
                        stratum=stratum,
                        style=style,
                        turns=len(group),
                        sample_ids=tuple(
                            f"{question_type}:{int(dialogue_id)}:{int(turn)}"
                            for turn in group["num"].astype(int)
                        ),
                        evidence_ids=frozenset(refs),
                        normalized_questions=tuple(
                            normalize_question(question) for question in group["question"]
                        ),
                    )
                )
        return dialogues
