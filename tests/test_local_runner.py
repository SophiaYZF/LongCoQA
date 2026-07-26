from longcoqa.inference.local_runner import (
    normalize_final_answer,
    summarize_local_results,
)


def test_normalize_final_answer_preserves_plain_text() -> None:
    assert normalize_final_answer(" 最终答案 ") == " 最终答案 "


def test_normalize_final_answer_keeps_suffix_after_last_reasoning_tag() -> None:
    assert normalize_final_answer("推理</think>最终答案") == "最终答案"
    assert normalize_final_answer("草稿</analysis> 最终答案 ") == "最终答案"
    assert normalize_final_answer("旧</think>中间</analysis>最终") == "最终"


def test_summarize_local_results() -> None:
    results = [
        {
            "status": "ok",
            "sample_id": "factoid:1:1",
            "prediction": "答案",
            "metrics": {
                "exact_match": 1.0,
                "character_f1": 1.0,
                "rouge_l_f1": 1.0,
            },
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        },
        {
            "status": "ok",
            "sample_id": "non_factoid:1:1",
            "prediction": "另一答案",
            "metrics": {
                "exact_match": 0.0,
                "character_f1": 0.5,
                "rouge_l_f1": 0.25,
            },
            "usage": {"prompt_tokens": 20, "completion_tokens": 4},
        },
    ]
    summary = summarize_local_results(
        results,
        model={"name": "test"},
        protocol="z1_teacher_forced",
        seed=42,
    )
    assert summary["successful_samples"] == 2
    assert summary["metrics"]["character_f1"] == 0.75
    assert summary["usage"] == {"prompt_tokens": 30, "completion_tokens": 6}
