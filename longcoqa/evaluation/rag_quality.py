from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class CitationScore:
    status: str
    precision: float | None
    recall: float | None
    f1: float | None
    cited_ids: tuple[str, ...]
    gold_ids: tuple[str, ...]
    correct_ids: tuple[str, ...]
    unknown_ids: tuple[str, ...]
    reason: str | None = None


@dataclass(frozen=True)
class ClaimAssessment:
    text: str
    material: bool
    supported: bool | None
    matched_reference: int | None = None


def parse_citations(text: str) -> tuple[str, ...]:
    footer = re.search(r"引用\s*[：:]\s*(.*)$", str(text), flags=re.DOTALL)
    if footer is None:
        return ()
    values = re.findall(
        r"(?:EVIDENCE|RETRIEVED)?\s*((?:NEG-)?\d+)",
        footer.group(1),
        flags=re.IGNORECASE,
    )
    return tuple(dict.fromkeys(values))


def citation_scores(
    cited_ids: Sequence[str],
    gold_ids: Sequence[str],
    *,
    answerable: bool,
) -> CitationScore:
    cited = tuple(dict.fromkeys(str(value) for value in cited_ids))
    gold = tuple(dict.fromkeys(str(value) for value in gold_ids))
    if not answerable and not gold:
        return CitationScore(
            status="inapplicable",
            precision=None,
            recall=None,
            f1=None,
            cited_ids=cited,
            gold_ids=gold,
            correct_ids=(),
            unknown_ids=cited,
            reason="missing_gold",
        )
    gold_set = set(gold)
    correct = tuple(value for value in cited if value in gold_set)
    unknown = tuple(value for value in cited if value not in gold_set)
    precision = len(correct) / len(cited) if cited else None
    recall = len(correct) / len(gold) if gold else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return CitationScore(
        status="ok",
        precision=precision,
        recall=recall,
        f1=f1,
        cited_ids=cited,
        gold_ids=gold,
        correct_ids=correct,
        unknown_ids=unknown,
    )


def claim_quality_record(
    *,
    sample_id: str,
    question_type: str,
    answerable: bool,
    reference_claims: Sequence[str],
    candidate_claims: Sequence[ClaimAssessment],
    evidence_available: bool,
    extraction_error: str | None = None,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "schema_version": "longcoqa-claim-quality-v1",
        "sample_id": sample_id,
        "question_type": question_type,
        "answerable": answerable,
        "reference_claims": list(reference_claims),
        "candidate_claims": [asdict(claim) for claim in candidate_claims],
    }
    if extraction_error is not None:
        return {**base, "status": "error", "error": extraction_error}

    matched: set[int] = set()
    for claim in candidate_claims:
        matched_reference = claim.matched_reference
        if matched_reference is None:
            normalized_claim = "".join(str(claim.text).lower().split())
            matched_reference = next(
                (
                    index
                    for index, reference in enumerate(reference_claims)
                    if "".join(str(reference).lower().split()) == normalized_claim
                ),
                None,
            )
        if matched_reference is None:
            continue
        if not 0 <= matched_reference < len(reference_claims):
            raise ValueError(
                f"matched_reference {matched_reference} is outside reference claims"
            )
        matched.add(matched_reference)
    claim_recall = (
        len(matched) / len(reference_claims)
        if reference_claims
        else None
    )

    material = [claim for claim in candidate_claims if claim.material]
    if not evidence_available:
        support_rate = None
        answer_supported = None
        support_status = "inapplicable"
    elif not material:
        support_rate = None
        answer_supported = False if answerable and reference_claims else None
        support_status = "ok"
    elif any(claim.supported is None for claim in material):
        support_rate = None
        answer_supported = None
        support_status = "error"
    else:
        supported = sum(claim.supported is True for claim in material)
        support_rate = supported / len(material)
        answer_supported = supported == len(material)
        support_status = "ok"

    return {
        **base,
        "status": "ok",
        "claim_recall": claim_recall,
        "claim_support_rate": support_rate,
        "answer_supported": answer_supported,
        "support_status": support_status,
    }


RAGCHECKER_FIELDS = (
    "claim_recall",
    "context_precision",
    "context_utilization",
    "noise_sensitivity",
    "faithfulness",
)


def adapt_ragchecker_result(raw: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in RAGCHECKER_FIELDS if field not in raw]
    if missing:
        raise ValueError(f"RAGChecker result is missing: {', '.join(missing)}")
    for field in RAGCHECKER_FIELDS:
        value = raw[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"RAGChecker field {field} must be numeric")
    return {
        "schema_version": "longcoqa-ragchecker-adapter-v1",
        "official": dict(raw),
    }


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return sum(values) / len(values) if values else None


def _citation_mean(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    values = []
    for row in rows:
        citation = row.get("citation")
        if isinstance(citation, CitationScore):
            value = getattr(citation, field)
        elif isinstance(citation, Mapping):
            value = citation.get(field)
        else:
            continue
        if value is not None:
            values.append(float(value))
    return sum(values) / len(values) if values else None


def _group(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    supported = [row["answer_supported"] for row in rows if row.get("answer_supported") is not None]
    return {
        "samples": len(rows),
        "citation_precision": _citation_mean(rows, "precision"),
        "citation_recall": _citation_mean(rows, "recall"),
        "citation_f1": _citation_mean(rows, "f1"),
        "claim_recall": _mean(rows, "claim_recall"),
        "claim_support_rate": _mean(rows, "claim_support_rate"),
        "answer_supported_rate": (
            sum(value is True for value in supported) / len(supported) if supported else None
        ),
    }


def aggregate_rag_quality(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failures = [row for row in records if row.get("status") == "error"]
    successful = [row for row in records if row.get("status") == "ok"]
    answerable = [row for row in successful if bool(row.get("answerable", True))]
    missing_gold = [row for row in successful if not bool(row.get("answerable", True))]
    return {
        "schema_version": "longcoqa-rag-quality-report-v1",
        "requested_samples": len(records),
        "successful_samples": len(successful),
        "failed_samples": len(failures),
        "answerable": _group(answerable),
        "missing_gold": _group(missing_gold),
    }


RAG_QUALITY_SYSTEM_PROMPT = """你是检索增强问答评测器。只依据给定问题、参考答案、
候选答案和可见证据拆分事实主张并判断证据支持。只输出 JSON，不输出 Markdown。"""


def build_rag_quality_messages(row: Mapping[str, Any]) -> list[dict[str, str]]:
    required = ("question", "reference", "prediction", "evidence_text")
    missing = [field for field in required if field not in row]
    if missing:
        raise ValueError(f"RAG quality input is missing: {', '.join(missing)}")
    user = f"""[问题]
{row['question']}

[参考答案]
{row['reference']}

[候选答案]
{row['prediction']}

[可见证据]
{row['evidence_text']}

输出以下 JSON：
{{
  "reference_claims": ["参考答案中的原子主张"],
  "candidate_claims": [
    {{
      "text": "候选答案中的原子主张",
      "material": true,
      "supported": true,
      "matched_reference": 0
    }}
  ],
  "citation_assessments": [
    {{"citation_id": "候选引用编号", "supported": true}}
  ]
}}

`matched_reference` 是对应 reference_claims 的零起始下标，无对应时为 null；
`supported` 只能依据可见证据判断。"""
    return [
        {"role": "system", "content": RAG_QUALITY_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _json_object(text: str) -> dict[str, Any]:
    cleaned = str(text).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as error:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("RAG quality response is not JSON") from error
        try:
            value = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as nested:
            raise ValueError("RAG quality response contains invalid JSON") from nested
    if not isinstance(value, dict):
        raise ValueError("RAG quality response must be a JSON object")
    return value


def parse_rag_quality_response(
    row: Mapping[str, Any], text: str
) -> dict[str, Any]:
    value = _json_object(text)
    for field in ("reference_claims", "candidate_claims", "citation_assessments"):
        if not isinstance(value.get(field), list):
            raise ValueError(f"RAG quality field {field} must be a list")
    reference_claims = tuple(str(claim) for claim in value["reference_claims"])
    candidate_claims: list[ClaimAssessment] = []
    for index, raw in enumerate(value["candidate_claims"]):
        if not isinstance(raw, Mapping):
            raise ValueError(f"candidate_claims[{index}] must be an object")
        if not isinstance(raw.get("material"), bool):
            raise ValueError(f"candidate_claims[{index}].material must be boolean")
        supported = raw.get("supported")
        if supported is not None and not isinstance(supported, bool):
            raise ValueError(f"candidate_claims[{index}].supported must be boolean or null")
        matched = raw.get("matched_reference")
        if matched is not None and (isinstance(matched, bool) or not isinstance(matched, int)):
            raise ValueError(
                f"candidate_claims[{index}].matched_reference must be integer or null"
            )
        candidate_claims.append(
            ClaimAssessment(
                text=str(raw.get("text", "")),
                material=raw["material"],
                supported=supported,
                matched_reference=matched,
            )
        )
    record = claim_quality_record(
        sample_id=str(row["sample_id"]),
        question_type=str(row["question_type"]),
        answerable=bool(row.get("answerable", True)),
        reference_claims=reference_claims,
        candidate_claims=tuple(candidate_claims),
        evidence_available=bool(str(row.get("evidence_text", "")).strip()),
    )
    citation_assessments = value["citation_assessments"]
    supported_citations = 0
    normalized_citations: list[dict[str, Any]] = []
    for index, raw in enumerate(citation_assessments):
        if not isinstance(raw, Mapping) or not isinstance(raw.get("supported"), bool):
            raise ValueError(f"citation_assessments[{index}] has invalid schema")
        normalized = {
            "citation_id": str(raw.get("citation_id", "")),
            "supported": raw["supported"],
        }
        normalized_citations.append(normalized)
        supported_citations += raw["supported"] is True
    return {
        **record,
        "citation_assessments": normalized_citations,
        "citation_semantic_precision": (
            supported_citations / len(normalized_citations)
            if normalized_citations
            else None
        ),
    }


def make_rag_quality_evaluator(client: Any, *, max_tokens: int = 1024):
    def evaluate(row: dict[str, Any]) -> dict[str, Any]:
        result = client.generate(
            build_rag_quality_messages(row),
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return parse_rag_quality_response(row, result.text)

    return evaluate
