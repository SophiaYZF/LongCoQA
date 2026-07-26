from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from longcoqa.utils.hashing import sha256_file, stable_json_hash
from longcoqa.utils.io import atomic_write_json, atomic_write_jsonl

CHAPTER_PATTERN = re.compile(r"^第(?P<number>[一二三四五六七八九十百〇零0-9]+)章(?:\s+(?P<title>.*))?$")


@dataclass(frozen=True)
class Paragraph:
    paragraph_id: str
    book_id: int
    book_title: str
    chapter_id: str
    chapter_number: str
    chapter_title: str
    index_in_chapter: int
    text: str


@dataclass(frozen=True)
class Chapter:
    chapter_id: str
    book_id: int
    book_title: str
    chapter_number: str
    chapter_title: str
    paragraphs: tuple[Paragraph, ...]


def _book_title(path: Path) -> str:
    return re.sub(r"^\d+", "", path.stem)


def _parse_novel(path: Path, book_id: int, encoding: str) -> list[Chapter]:
    chapters: list[Chapter] = []
    book_title = _book_title(path)
    chapter_number = "0"
    chapter_title = "前言"
    chapter_index = 0
    paragraphs: list[Paragraph] = []

    def flush() -> None:
        nonlocal paragraphs
        if not paragraphs:
            return
        chapter_id = f"b{book_id:02d}-c{chapter_index:03d}"
        chapters.append(
            Chapter(
                chapter_id=chapter_id,
                book_id=book_id,
                book_title=book_title,
                chapter_number=chapter_number,
                chapter_title=chapter_title,
                paragraphs=tuple(paragraphs),
            )
        )
        paragraphs = []

    for raw_line in path.read_text(encoding=encoding).splitlines():
        text = raw_line.strip()
        if not text:
            continue
        match = CHAPTER_PATTERN.fullmatch(text)
        if match:
            flush()
            chapter_index += 1
            chapter_number = match.group("number")
            chapter_title = (match.group("title") or "").strip()
            continue
        if chapter_index == 0:
            chapter_index = 1
        chapter_id = f"b{book_id:02d}-c{chapter_index:03d}"
        paragraph_index = len(paragraphs) + 1
        paragraphs.append(
            Paragraph(
                paragraph_id=f"{chapter_id}-p{paragraph_index:04d}",
                book_id=book_id,
                book_title=book_title,
                chapter_id=chapter_id,
                chapter_number=chapter_number,
                chapter_title=chapter_title,
                index_in_chapter=paragraph_index,
                text=text,
            )
        )
    flush()
    return chapters


def discover_novel_paths(novel_dir: Path) -> list[Path]:
    novel_dir = Path(novel_dir)
    paths = sorted(novel_dir.glob("*.txt"))
    if not paths:
        raise FileNotFoundError(
            f"No .txt novels found in {novel_dir}. LongCoQA does not bundle the "
            "source novels; provide user-supplied lawful source texts."
        )
    return paths


def parse_novels(novel_dir: Path, encoding: str = "gb18030") -> list[Chapter]:
    chapters: list[Chapter] = []
    for book_id, path in enumerate(discover_novel_paths(novel_dir), start=1):
        chapters.extend(_parse_novel(path, book_id, encoding))
    return chapters


def _pack_paragraphs(
    paragraphs: tuple[Paragraph, ...],
    *,
    target_chars: int,
    max_chars: int,
) -> list[list[Paragraph]]:
    if target_chars <= 0 or max_chars < target_chars:
        raise ValueError("Chunk sizes must satisfy 0 < target_chars <= max_chars")
    groups: list[list[Paragraph]] = []
    current: list[Paragraph] = []
    current_chars = 0
    for paragraph in paragraphs:
        separator_chars = 1 if current else 0
        projected = current_chars + separator_chars + len(paragraph.text)
        if current and projected > max_chars:
            groups.append(current)
            current = []
            current_chars = 0
        current.append(paragraph)
        current_chars += (1 if current_chars else 0) + len(paragraph.text)
        if current_chars >= target_chars:
            groups.append(current)
            current = []
            current_chars = 0
    if current:
        groups.append(current)
    return groups


def build_hierarchical_corpus(
    chapters: list[Chapter],
    *,
    parent_target_chars: int,
    parent_max_chars: int,
    child_target_chars: int,
    child_max_chars: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    parents: list[dict[str, Any]] = []
    children: list[dict[str, Any]] = []
    for chapter in chapters:
        parent_groups = _pack_paragraphs(
            chapter.paragraphs,
            target_chars=parent_target_chars,
            max_chars=parent_max_chars,
        )
        for parent_index, parent_group in enumerate(parent_groups, start=1):
            parent_id = f"{chapter.chapter_id}-parent{parent_index:03d}"
            parent_text = "\n".join(item.text for item in parent_group)
            parent = {
                "parent_id": parent_id,
                "book_id": chapter.book_id,
                "book_title": chapter.book_title,
                "chapter_id": chapter.chapter_id,
                "chapter_number": chapter.chapter_number,
                "chapter_title": chapter.chapter_title,
                "paragraph_ids": [item.paragraph_id for item in parent_group],
                "text": parent_text,
                "characters": len(parent_text),
            }
            parents.append(parent)

            child_groups = _pack_paragraphs(
                tuple(parent_group),
                target_chars=child_target_chars,
                max_chars=child_max_chars,
            )
            for child_index, child_group in enumerate(child_groups, start=1):
                child_id = f"{parent_id}-child{child_index:02d}"
                child_text = "\n".join(item.text for item in child_group)
                children.append(
                    {
                        "child_id": child_id,
                        "parent_id": parent_id,
                        "book_id": chapter.book_id,
                        "book_title": chapter.book_title,
                        "chapter_id": chapter.chapter_id,
                        "chapter_number": chapter.chapter_number,
                        "chapter_title": chapter.chapter_title,
                        "paragraph_ids": [item.paragraph_id for item in child_group],
                        "text": child_text,
                        "characters": len(child_text),
                    }
                )
    return parents, children


def build_legacy_units(
    chapters: list[Chapter],
    *,
    short_paragraph_chars: int = 50,
) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    unit_groups: list[list[Paragraph]] = []
    for chapter in chapters:
        for paragraph in chapter.paragraphs:
            if unit_groups and len(paragraph.text) < short_paragraph_chars:
                unit_groups[-1].append(paragraph)
            else:
                unit_groups.append([paragraph])
    for group in unit_groups:
        first = group[0]
        text = "\n".join(item.text for item in group)
        units.append(
            {
                "candidate_id": len(units) + 1,
                "book_id": first.book_id,
                "book_title": first.book_title,
                "chapter_id": first.chapter_id,
                "chapter_number": first.chapter_number,
                "chapter_title": first.chapter_title,
                "paragraph_ids": [item.paragraph_id for item in group],
                "text": text,
                "characters": len(text),
            }
        )
    return units


def write_corpus(
    novel_dir: Path,
    output_dir: Path,
    *,
    encoding: str = "gb18030",
    legacy_short_paragraph_chars: int = 50,
    parent_target_chars: int = 1200,
    parent_max_chars: int = 1600,
    child_target_chars: int = 350,
    child_max_chars: int = 500,
) -> dict[str, Any]:
    chapters = parse_novels(novel_dir, encoding=encoding)
    parents, children = build_hierarchical_corpus(
        chapters,
        parent_target_chars=parent_target_chars,
        parent_max_chars=parent_max_chars,
        child_target_chars=child_target_chars,
        child_max_chars=child_max_chars,
    )
    legacy_units = build_legacy_units(
        chapters,
        short_paragraph_chars=legacy_short_paragraph_chars,
    )
    paragraphs = [asdict(paragraph) for chapter in chapters for paragraph in chapter.paragraphs]
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "paragraphs": output_dir / "paragraphs.jsonl",
        "parents": output_dir / "parents.jsonl",
        "children": output_dir / "children.jsonl",
        "legacy_units": output_dir / "legacy_units.jsonl",
    }
    atomic_write_jsonl(paths["paragraphs"], paragraphs)
    atomic_write_jsonl(paths["parents"], parents)
    atomic_write_jsonl(paths["children"], children)
    atomic_write_jsonl(paths["legacy_units"], legacy_units)

    metadata = {
        "schema_version": "longcoqa-corpus-v1",
        "parameters": {
            "encoding": encoding,
            "legacy_short_paragraph_chars": legacy_short_paragraph_chars,
            "parent_target_chars": parent_target_chars,
            "parent_max_chars": parent_max_chars,
            "child_target_chars": child_target_chars,
            "child_max_chars": child_max_chars,
        },
        "counts": {
            "books": len({chapter.book_id for chapter in chapters}),
            "chapters": len(chapters),
            "paragraphs": len(paragraphs),
            "parents": len(parents),
            "children": len(children),
            "legacy_units": len(legacy_units),
        },
        "novel_checksums": {path.name: sha256_file(path) for path in sorted(Path(novel_dir).glob("*.txt"))},
        "content_hashes": {
            key: stable_json_hash(
                [
                    {
                        "id": item.get(f"{key[:-1]}_id", item.get("candidate_id")),
                        "text": item["text"],
                    }
                    for item in (
                        paragraphs
                        if key == "paragraphs"
                        else parents
                        if key == "parents"
                        else children
                        if key == "children"
                        else legacy_units
                    )
                ]
            )
            for key in paths
        },
    }
    metadata["artifact_checksums"] = {key: sha256_file(path) for key, path in paths.items()}
    atomic_write_json(output_dir / "metadata.json", metadata)
    return metadata
