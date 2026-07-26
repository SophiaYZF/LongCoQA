import pytest

from longcoqa.evaluation.judge_runner import select_judge_inputs, summarize_judgments
from longcoqa.evaluation.llm_judge import (
    build_refusal_judge_messages,
    extract_evidence_text,
    parse_judge_response,
    parse_refusal_judge_response,
)

VALID = """{
  "verdict": "mostly_correct",
  "correctness": 4,
  "completeness": 3,
  "relevance": 5,
  "conciseness": 4,
  "has_factual_conflict": false,
  "has_unsupported_claim": true,
  "missing_key_points": ["一个要点"],
  "error_summary": "有额外断言"
}"""


def test_parse_judge_response_accepts_fenced_json() -> None:
    parsed = parse_judge_response(f"```json\n{VALID}\n```")
    assert parsed["verdict"] == "mostly_correct"
    assert parsed["correctness"] == 4
    assert parsed["has_unsupported_claim"] is True


def test_parse_judge_response_rejects_out_of_range_score() -> None:
    with pytest.raises(ValueError, match=r"correctness must be in"):
        parse_judge_response(VALID.replace('"correctness": 4', '"correctness": 6'))


def test_summarize_judgments_is_stratified() -> None:
    result = {
        "status": "ok",
        "question_type": "factoid",
        "judgment": parse_judge_response(VALID),
    }
    summary = summarize_judgments([result])
    assert summary["by_question_type"]["factoid"]["scores"]["correctness"] == 4
    assert summary["by_question_type"]["non_factoid"]["samples"] == 0


def test_summarize_judgments_separates_refusal_rows() -> None:
    answer = {
        "status": "ok",
        "question_type": "factoid",
        "judgment_type": "answer_quality",
        "judgment": parse_judge_response(VALID),
    }
    refusal = {
        "status": "ok",
        "question_type": "factoid",
        "judgment_type": "refusal_quality",
        "judgment": parse_refusal_judge_response(
            '{"verdict":"appropriate_refusal","refusal_appropriateness":5,'
            '"grounding":5,"relevance":5,"conciseness":4,'
            '"has_unsupported_answer":false,'
            '"claims_sufficient_evidence":false,'
            '"cites_irrelevant_evidence":false,"error_summary":""}'
        ),
    }

    summary = summarize_judgments([answer, refusal])

    assert summary["answer_quality"]["successful_samples"] == 1
    assert summary["answer_quality"]["judge_100"] == 75.0
    assert summary["refusal_quality"]["successful_samples"] == 1
    assert summary["refusal_quality"]["appropriate_refusal_rate"] == 1.0
    assert summary["by_question_type"]["factoid"]["samples"] == 1


def test_select_judge_inputs_balances_question_types() -> None:
    inputs = [
        {"question_type": "factoid", "sample_id": str(index)} for index in range(5)
    ] + [
        {"question_type": "non_factoid", "sample_id": str(index)}
        for index in range(5)
    ]
    selected = select_judge_inputs(inputs, 4)
    assert [item["question_type"] for item in selected] == [
        "factoid",
        "non_factoid",
        "factoid",
        "non_factoid",
    ]


def test_extract_evidence_text_from_grounded_protocol() -> None:
    item = {
        "protocol": "g3_aligned_raw",
        "messages": [
            {
                "role": "user",
                "content": (
                    "请依据证据回答。\n\n[EVIDENCE 7]\n原文段落"
                    "\n\n[QUESTION]\n发生了什么？"
                ),
            }
        ],
    }
    assert extract_evidence_text(item) == "[EVIDENCE 7]\n原文段落"


def test_extract_evidence_text_from_rag() -> None:
    item = {
        "retrieval_method": "dense",
        "messages": [
            {
                "role": "user",
                "content": (
                    "使用检索结果。\n\n[RETRIEVED 1 | 标题]\n原文段落"
                    "\n\n[STANDALONE RETRIEVAL QUERY]\n查询"
                ),
            }
        ],
    }
    assert extract_evidence_text(item) == "[RETRIEVED 1 | 标题]\n原文段落"


def test_refusal_prompt_hides_reference_and_keeps_evidence() -> None:
    messages = build_refusal_judge_messages(
        {
            "question_type": "factoid",
            "question": "谁拿走了物品？",
            "reference": "不得暴露的真实答案",
            "prediction": "提供的证据不足以回答该问题。",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "[EVIDENCE NEG-1]\n无关片段"
                        "\n\n[QUESTION]\n谁拿走了物品？"
                    ),
                }
            ],
        }
    )
    prompt = messages[-1]["content"]
    assert "不得暴露的真实答案" not in prompt
    assert "无关片段" in prompt
    assert "提供的证据不足" in prompt


def test_parse_refusal_judge_response_validates_schema() -> None:
    parsed = parse_refusal_judge_response(
        '{"verdict":"appropriate_refusal","refusal_appropriateness":5,'
        '"grounding":5,"relevance":5,"conciseness":4,'
        '"has_unsupported_answer":false,'
        '"claims_sufficient_evidence":false,'
        '"cites_irrelevant_evidence":false,"error_summary":""}'
    )
    assert parsed["verdict"] == "appropriate_refusal"
    assert parsed["refusal_appropriateness"] == 5


def test_parse_refusal_judge_response_rejects_invalid_boolean() -> None:
    invalid = (
        '{"verdict":"appropriate_refusal","refusal_appropriateness":5,'
        '"grounding":5,"relevance":5,"conciseness":4,'
        '"has_unsupported_answer":"false",'
        '"claims_sufficient_evidence":false,'
        '"cites_irrelevant_evidence":false,"error_summary":""}'
    )
    with pytest.raises(ValueError, match="has_unsupported_answer must be boolean"):
        parse_refusal_judge_response(invalid)
