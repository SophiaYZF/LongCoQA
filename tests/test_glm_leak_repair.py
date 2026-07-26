from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPARER = PROJECT_ROOT / "scripts/prepare_glm_leak_repair.py"
ORCHESTRATOR = PROJECT_ROOT / "scripts/run_glm_leak_repair_2gpu.sh"
EXPECTED = {
    "z0_current": 5,
    "z1_teacher_forced": 8,
    "g0_released_gold": 1,
    "g3_aligned_raw": 1,
}


def _write_fixture(root: Path, counts: dict[str, int]) -> tuple[Path, list[Path]]:
    model_root = root / "outputs/formal/models/glm4_7_flash"
    checkpoints: list[Path] = []
    for protocol_index, (protocol, count) in enumerate(counts.items()):
        result_dir = model_root / protocol / "full"
        result_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for item_index in range(count):
            sample_id = f"non_factoid:{protocol_index + 1}:{item_index + 1}"
            rows.append(
                {
                    "sample_id": sample_id,
                    "status": "ok",
                    "prediction": "推理</think>最终答案",
                }
            )
            shard_index = (protocol_index + item_index) % 8
            checkpoint = (
                model_root
                / "shards"
                / f"shard_{shard_index:02d}"
                / protocol
                / "full/checkpoints"
                / f"{sample_id.replace(':', '__')}.json"
            )
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_text(json.dumps(rows[-1]), encoding="utf-8")
            checkpoints.append(checkpoint)
        (result_dir / "results.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    return model_root, checkpoints


def _run_preparer(model_root: Path, run_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(PREPARER),
            "--model-root",
            str(model_root),
            "--run-dir",
            str(run_dir),
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def test_preparer_backs_up_exact_expected_leaks(tmp_path: Path) -> None:
    model_root, checkpoints = _write_fixture(tmp_path, EXPECTED)
    run_dir = tmp_path / "runtime/repair"

    completed = _run_preparer(model_root, run_dir)

    assert completed.returncode == 0, completed.stderr
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["rows"]) == 15
    assert manifest["counts"] == EXPECTED
    assert all(not checkpoint.exists() for checkpoint in checkpoints)
    backups = [Path(row["backup_checkpoint"]) for row in manifest["rows"]]
    assert all(path.is_file() for path in backups)
    expected_shards = sorted({int(row["shard_index"]) for row in manifest["rows"]})
    actual_shards = [
        int(line)
        for line in (run_dir / "affected_shards.txt").read_text().splitlines()
    ]
    assert actual_shards == expected_shards


def test_preparer_refuses_unexpected_counts_before_moving(tmp_path: Path) -> None:
    counts = {**EXPECTED, "z0_current": 4}
    model_root, checkpoints = _write_fixture(tmp_path, counts)
    run_dir = tmp_path / "runtime/refused"

    completed = _run_preparer(model_root, run_dir)

    assert completed.returncode != 0
    assert "Unexpected leak counts" in completed.stderr
    assert all(checkpoint.is_file() for checkpoint in checkpoints)
    assert not (run_dir / "manifest.json").exists()


def test_two_gpu_orchestrator_contract() -> None:
    text = ORCHESTRATOR.read_text(encoding="utf-8")
    assert "prepare_glm_leak_repair.py" in text
    assert "LONGCOQA_EXPECTED_GPUS=2" in text
    assert "CUDA_VISIBLE_DEVICES=$gpu" in text
    assert "--shard-count 8" in text
    assert "merge-local-shards" in text
    assert "--bootstrap-iterations 2000" in text
    assert "glm4_7_flash/.complete" in text
    assert "GLM_LEAK_REPAIR_OK" in text
