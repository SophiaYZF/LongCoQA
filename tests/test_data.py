from __future__ import annotations

from longcoqa.data import LongCoQADataset


def test_released_data_validates_with_expected_counts() -> None:
    report = LongCoQADataset().validate()

    assert report["status"] == "valid"
    assert report["factoid"]["questions"] == 1172
    assert report["factoid"]["dialogues"] == 100
    assert report["non_factoid"]["questions"] == 1847
    assert report["non_factoid"]["dialogues"] == 151
    assert report["evidence"]["rows"] == 2319
    assert report["evidence"]["total_reference_mentions"] == 6972
    assert report["novels"] == {}
