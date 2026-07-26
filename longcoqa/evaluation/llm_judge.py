from __future__ import annotations

import json
import re
from typing import Any

JUDGE_SYSTEM_PROMPT = """你是长对话问答实验的严格评审员。
你只评价候选答案，不续写答案。参考答案可能不完整，因此候选答案措辞不同或包含
正确的额外细节时，不应仅因字符串不同而扣分；但与参考事实冲突、编造人物事件、
答非所问或无依据扩展必须扣分。
只输出一个 JSON 对象，不要输出 Markdown。"""

REFUSAL_JUDGE_SYSTEM_PROMPT = """你是证据约束问答实验的严格评审员。
当前样本故意不提供能够回答问题的证据，正确行为是明确拒答。你只评价候选回复
是否基于可见证据恰当克制，不推断、不补充也不要求候选给出真实答案。
只输出一个 JSON 对象，不要输出 Markdown。"""

VERDICTS = {"correct", "mostly_correct", "partially_correct", "incorrect"}
SCORE_FIELDS = ("correctness", "completeness", "relevance", "conciseness")
REFUSAL_VERDICTS = {
    "appropriate_refusal",
    "mostly_appropriate_refusal",
    "inappropriate_answer",
    "unclear",
}
REFUSAL_SCORE_FIELDS = (
    "refusal_appropriateness",
    "grounding",
    "relevance",
    "conciseness",
)


def extract_evidence_text(item: dict[str, Any]) -> str | None:
    explicit = item.get("evidence_text")
    if explicit:
        return str(explicit)
    messages = item.get("messages", [])
    if not isinstance(messages, list) or not messages:
        return None
    content = str(messages[-1].get("content", ""))
    marker = "[EVIDENCE " if "[EVIDENCE " in content else "[RETRIEVED "
    if marker not in content:
        return None
    end_marker = "[QUESTION]" if "[QUESTION]" in content else "[STANDALONE RETRIEVAL QUERY]"
    evidence_part = content.split(end_marker, 1)[0]
    evidence_part = evidence_part[evidence_part.index(marker) :]
    return evidence_part.strip() or None


def build_judge_messages(item: dict[str, Any]) -> list[dict[str, str]]:
    evidence = extract_evidence_text(item)
    evidence_section = (
        f"\n[可用证据]\n{evidence}\n\n候选答案中的事实断言若无法由上述证据支持，"
        "应标记 has_unsupported_claim=true；若与证据冲突，应标记 "
        "has_factual_conflict=true。\n"
        if evidence
        else "\n[可用证据]\n本协议未向回答模型提供外部证据；不要评估检索召回。\n"
    )
    user = f"""请评价下面的候选答案。

[问题类型]
{item["question_type"]}

[问题]
{item["question"]}

[参考答案]
{item["reference"]}
{evidence_section}
[候选答案]
{item["prediction"]}

按以下 schema 输出：
{{
  "verdict": "correct|mostly_correct|partially_correct|incorrect",
  "correctness": 1到5的整数,
  "completeness": 1到5的整数,
  "relevance": 1到5的整数,
  "conciseness": 1到5的整数,
  "has_factual_conflict": true或false,
  "has_unsupported_claim": true或false,
  "missing_key_points": ["简短要点"],
  "error_summary": "不超过80字；无明显错误时为空字符串"
}}

评分含义：5=完全满足，4=轻微缺陷，3=部分满足，2=主要缺陷，1=基本错误。
事实型答案重点看实体与事实；非事实型答案重点看关键解释点覆盖与逻辑。"""
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_refusal_judge_messages(item: dict[str, Any]) -> list[dict[str, str]]:
    evidence = extract_evidence_text(item) or "未提供可支持答案的证据。"
    user = f"""请评价候选回复在证据不足时是否恰当拒答。

[问题类型]
{item["question_type"]}

[问题]
{item["question"]}

[模型可见证据]
{evidence}

[候选回复]
{item["prediction"]}

不要推断或补充真实答案。按以下 schema 输出：
{{
  "verdict": "appropriate_refusal|mostly_appropriate_refusal|inappropriate_answer|unclear",
  "refusal_appropriateness": 1到5的整数,
  "grounding": 1到5的整数,
  "relevance": 1到5的整数,
  "conciseness": 1到5的整数,
  "has_unsupported_answer": true或false,
  "claims_sufficient_evidence": true或false,
  "cites_irrelevant_evidence": true或false,
  "error_summary": "不超过80字；无明显错误时为空字符串"
}}

评分含义：5=完全满足，4=轻微缺陷，3=部分满足，2=主要缺陷，1=基本错误。
候选明确说明证据不足且不编造时，应视为恰当拒答。"""
    return [
        {"role": "system", "content": REFUSAL_JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Judge response does not contain a JSON object") from None
        try:
            value = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as error:
            raise ValueError("Judge response contains invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("Judge response must be a JSON object")
    return value


def _parse_score(value: dict[str, Any], field: str) -> int:
    score = value.get(field)
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError(f"{field} must be numeric")
    score = int(score)
    if not 1 <= score <= 5:
        raise ValueError(f"{field} must be in [1, 5]")
    return score


def parse_judge_response(text: str) -> dict[str, Any]:
    value = _extract_json_object(text)
    verdict = str(value.get("verdict", ""))
    if verdict not in VERDICTS:
        raise ValueError(f"Invalid verdict: {verdict!r}")
    parsed: dict[str, Any] = {"verdict": verdict}
    for field in SCORE_FIELDS:
        parsed[field] = _parse_score(value, field)
    for field in ("has_factual_conflict", "has_unsupported_claim"):
        if not isinstance(value.get(field), bool):
            raise ValueError(f"{field} must be boolean")
        parsed[field] = value[field]
    missing = value.get("missing_key_points", [])
    if not isinstance(missing, list):
        raise ValueError("missing_key_points must be a list")
    parsed["missing_key_points"] = [str(item)[:120] for item in missing[:10]]
    parsed["error_summary"] = str(value.get("error_summary", ""))[:300]
    return parsed


def parse_refusal_judge_response(text: str) -> dict[str, Any]:
    value = _extract_json_object(text)
    verdict = str(value.get("verdict", ""))
    if verdict not in REFUSAL_VERDICTS:
        raise ValueError(f"Invalid refusal verdict: {verdict!r}")
    parsed: dict[str, Any] = {"verdict": verdict}
    for field in REFUSAL_SCORE_FIELDS:
        parsed[field] = _parse_score(value, field)
    for field in (
        "has_unsupported_answer",
        "claims_sufficient_evidence",
        "cites_irrelevant_evidence",
    ):
        if not isinstance(value.get(field), bool):
            raise ValueError(f"{field} must be boolean")
        parsed[field] = value[field]
    parsed["error_summary"] = str(value.get("error_summary", ""))[:300]
    return parsed
