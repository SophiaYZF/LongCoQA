import json

import pytest

from longcoqa.inference.merge_shards import merge_local_shards


def test_merge_local_shards_rejects_duplicates(tmp_path) -> None:
    for index in range(2):
        run = tmp_path / "shards" / f"shard_{index:02d}" / "z0_current" / "full"
        run.mkdir(parents=True)
        result = {
            "status": "ok",
            "sample_id": "factoid:1:1",
            "question_type": "factoid",
            "dialogue_id": 1,
            "turn": 1,
            "prediction": "x",
            "metrics": {
                "exact_match": 0,
                "character_f1": 0,
                "rouge_l_f1": 0,
            },
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        (run / "results.jsonl").write_text(
            json.dumps(result, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (run / "summary.json").write_text(
            json.dumps(
                {
                    "failed_samples": 0,
                    "model": {"name": "test"},
                    "seed": 42,
                }
            ),
            encoding="utf-8",
        )
    with pytest.raises(ValueError, match="duplicate sample IDs"):
        merge_local_shards(
            tmp_path / "shards",
            tmp_path / "merged",
            protocols=("z0_current",),
            shard_count=2,
            expected_samples=2,
        )
