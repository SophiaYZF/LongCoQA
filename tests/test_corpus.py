from __future__ import annotations

from pathlib import Path

import pytest

from longcoqa.data.corpus import parse_novels


def test_parse_novels_explains_unbundled_source_requirement(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="user-supplied lawful source texts"):
        parse_novels(tmp_path)


def test_parse_novels_accepts_a_small_user_supplied_fixture(tmp_path: Path) -> None:
    novel = tmp_path / "01示例.txt"
    novel.write_text("第一章 开始\n第一段。\n第二段。\n", encoding="gb18030")
    chapters = parse_novels(tmp_path)
    assert len(chapters) == 1
    assert [item.text for item in chapters[0].paragraphs] == ["第一段。", "第二段。"]
