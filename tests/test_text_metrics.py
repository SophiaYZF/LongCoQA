from __future__ import annotations

from longcoqa.evaluation.text_metrics import character_f1, lcs_f1, score_answer


def test_text_metrics_ignore_punctuation_and_reward_overlap() -> None:
    assert score_answer("哈利·波特！", "哈利波特")["exact_match"] == 1.0
    assert character_f1("哈利和罗恩", "哈利与赫敏") > 0.0
    assert lcs_f1("甲乙丙", "甲乙丁") == 2 / 3
    assert lcs_f1("甲", "乙") == 0.0
