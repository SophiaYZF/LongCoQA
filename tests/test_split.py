from __future__ import annotations

from longcoqa.data import LongCoQADataset
from longcoqa.data.split import optimize_dialogue_split


def test_p1_split_is_deterministic_and_respects_dialog_boundaries() -> None:
    dataset = LongCoQADataset()
    kwargs = {
        "seed": 42,
        "ratios": {"train": 0.7, "dev": 0.1, "test": 0.2},
        "restarts": 2,
        "local_steps": 100,
        "evidence_cut_weight": 1.0,
        "duplicate_question_penalty": 1000.0,
    }

    first = optimize_dialogue_split(dataset.dialogues(), **kwargs)
    second = optimize_dialogue_split(dataset.dialogues(), **kwargs)

    assert first == second
    assert len(first.assignments) == 251
    assigned = list(first.assignments)
    assert len(assigned) == len(set(assigned))
    assert first.diagnostics["duplicate_question_crossings"] == {}
    assert first.diagnostics["dialogues"] == {
        "train": 176,
        "dev": 24,
        "test": 51,
    }
