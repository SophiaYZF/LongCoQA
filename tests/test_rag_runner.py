import json

import pytest
from typer.testing import CliRunner

import longcoqa.cli as cli_module
from longcoqa.cli import app
from longcoqa.rag.runner import (
    checkpoint_is_current,
    merge_rag_shards,
    summarize_rag_metrics,
)


def test_merge_rag_shards_rejects_duplicate(tmp_path) -> None:
    for index in range(2):
        path = tmp_path / "shards" / f"shard_{index:02d}" / "results.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps({"sample_id": "same"}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    with pytest.raises(ValueError, match="Duplicate"):
        merge_rag_shards(
            shard_root=tmp_path / "shards",
            output_dir=tmp_path / "merged",
            shard_count=2,
            expected_samples=2,
        )


def _write_shard(
    tmp_path, index: int, row: dict, *, contexts_checksum: str = "same"
) -> None:
    directory = tmp_path / "shards" / f"shard_{index:02d}"
    directory.mkdir(parents=True)
    (directory / "results.jsonl").write_text(
        json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (directory / "summary.json").write_text(
        json.dumps(
            {
                "examples": 1,
                "shard": {"index": index, "count": 2},
                "contexts_checksum": contexts_checksum,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_merge_rag_shards_rejects_context_checksum_mismatch(tmp_path) -> None:
    _write_shard(
        tmp_path,
        0,
        {"sample_id": "a", "status": "ok", "question_type": "factoid", "dialogue_id": 1, "turn": 1},
        contexts_checksum="left",
    )
    _write_shard(
        tmp_path,
        1,
        {"sample_id": "b", "status": "ok", "question_type": "factoid", "dialogue_id": 1, "turn": 2},
        contexts_checksum="right",
    )
    with pytest.raises(ValueError, match="context checksum"):
        merge_rag_shards(
            shard_root=tmp_path / "shards",
            output_dir=tmp_path / "merged",
            shard_count=2,
            expected_samples=2,
        )


def test_merge_rag_shards_rejects_non_ok_status(tmp_path) -> None:
    _write_shard(
        tmp_path,
        0,
        {"sample_id": "a", "status": "failed", "question_type": "factoid", "dialogue_id": 1, "turn": 1},
    )
    _write_shard(
        tmp_path,
        1,
        {"sample_id": "b", "status": "ok", "question_type": "factoid", "dialogue_id": 1, "turn": 2},
    )
    with pytest.raises(ValueError, match="non-ok"):
        merge_rag_shards(
            shard_root=tmp_path / "shards",
            output_dir=tmp_path / "merged",
            shard_count=2,
            expected_samples=2,
        )


def test_run_rag_local_cli_accepts_context_override(tmp_path, monkeypatch) -> None:
    model_config = tmp_path / "model.yaml"
    model_config.write_text(
        "name: test\npath: /tmp/model\nrevision: test\ndtype: bfloat16\n"
        "tensor_parallel_size: 1\ngpu_memory_utilization: 0.9\n"
        "max_model_len: 1024\nmax_num_seqs: 1\n",
        encoding="utf-8",
    )
    config = tmp_path / "phase.yaml"
    config.write_text("rag:\n  contexts: default.jsonl\n", encoding="utf-8")
    contexts = tmp_path / "override.jsonl"
    contexts.write_text("{}\n", encoding="utf-8")
    captured = {}

    def fake_run_local_rag(**kwargs):
        captured.update(kwargs)
        return {"examples": 1}

    monkeypatch.setattr(cli_module, "run_local_rag", fake_run_local_rag)
    result = CliRunner().invoke(
        app,
        [
            "run-rag-local",
            "--model-config",
            str(model_config),
            "--config",
            str(config),
            "--contexts",
            str(contexts),
            "--output-dir",
            str(tmp_path / "output"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["contexts_path"] == contexts


def test_checkpoint_is_current_requires_matching_context_checksum(tmp_path) -> None:
    path = tmp_path / "checkpoint.json"
    path.write_text(
        json.dumps({"sample_id": "a", "status": "ok", "contexts_checksum": "new"}),
        encoding="utf-8",
    )
    assert checkpoint_is_current(path, "new")
    assert not checkpoint_is_current(path, "old")
    path.write_text(json.dumps({"sample_id": "a", "status": "ok"}), encoding="utf-8")
    assert not checkpoint_is_current(path, "new")


def test_summarize_rag_metrics_allows_empty_shard() -> None:
    assert summarize_rag_metrics([]) == {}
