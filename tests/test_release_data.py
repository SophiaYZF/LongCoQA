from __future__ import annotations

import hashlib
import json
from pathlib import Path

from longcoqa.data import LongCoQADataset
from longcoqa.paths import PROJECT_ROOT

EXPECTED = {
    "LongCoQA_factoid.csv": "b2bd914f3e830c220e45ac6140930bc07d03b64c4c3b6fdd8c50b988bc087201",
    "LongCoQA_non_factoid.csv": "f7bc1d59de921100c996272b6543c9bccf609942acd0ee9e238405ca78b71238",
    "LongCoQA_evidence.csv": "e22bee840b8a570bb157c79cf3a982b0a01ac21abfba713e3e0a6dd6607cc123",
    "splits/p0_full.json": "366f3005d421426dbd0c16ec0d590b414e446f7a2c4ca841ef873073e857c5e3",
    "splits/p1_dialogue_split.json": "e537b7c6944a6e9aebb967f6cb122a497f8fb9425239fb126d65203240fabe02",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_released_files_match_locked_hashes() -> None:
    for relative, expected in EXPECTED.items():
        assert sha256(PROJECT_ROOT / "data" / relative) == expected


def test_released_counts_and_splits_are_complete() -> None:
    report = LongCoQADataset().validate()
    assert report["factoid"]["questions"] == 1172
    assert report["non_factoid"]["questions"] == 1847
    assert report["evidence"]["rows"] == 2319
    assert report["novels"] == {}
    for name in ("p0_full.json", "p1_dialogue_split.json"):
        value = json.loads((PROJECT_ROOT / "data/splits" / name).read_text(encoding="utf-8"))
        assert len(value["samples"]) == 3019
        assert len({row["sample_id"] for row in value["samples"]}) == 3019
