from longcoqa.evaluation.judge_sanitization import (
    sanitize_pathological_char_runs,
    sanitize_pathological_fragment_runs,
)


def test_normal_text_is_unchanged() -> None:
    text = "卡多根爵士每天更换复杂的开门咒语。"

    sanitized, runs = sanitize_pathological_char_runs(text)

    assert sanitized == text
    assert runs == []


def test_long_unicode_run_is_compressed_with_exact_count() -> None:
    text = "看守员" + "员" * 100 + "。"

    sanitized, runs = sanitize_pathological_char_runs(
        text, threshold=64, keep=8
    )

    assert sanitized == (
        "看守" + "员" * 8 + "［字符“员”连续重复 101 次；已保留前 8 次］。"
    )
    assert runs == [
        {
            "character": "员",
            "run_length": 101,
            "kept_characters": 8,
            "replacement": "［字符“员”连续重复 101 次；已保留前 8 次］",
            "start": 2,
            "end": 103,
        }
    ]


def test_long_fragment_run_is_compressed_with_exact_count() -> None:
    text = "开头" + "校长、" * 100 + "结束"

    sanitized, runs = sanitize_pathological_fragment_runs(text)

    assert sanitized == (
        "开头"
        + "校长、" * 8
        + "［片段“校长、”连续重复 100 次；已保留前 8 次］结束"
    )
    assert runs == [
        {
            "fragment": "校长、",
            "fragment_length": 3,
            "repeat_count": 100,
            "kept_repeats": 8,
            "replacement": "［片段“校长、”连续重复 100 次；已保留前 8 次］",
            "start": 2,
            "end": 302,
        }
    ]


def test_short_fragment_repetition_is_unchanged() -> None:
    text = "很好。很好。"

    assert sanitize_pathological_fragment_runs(text) == (text, [])
