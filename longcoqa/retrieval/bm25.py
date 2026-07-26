from __future__ import annotations

import gzip
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from longcoqa.utils.hashing import sha256_file, stable_json_hash
from longcoqa.utils.io import atomic_write_json

ASCII_WORD = re.compile(r"[A-Za-z0-9]+")
CHINESE_RUN = re.compile(r"[\u4e00-\u9fff]+")


def lexical_tokens(text: str) -> list[str]:
    text = str(text).lower()
    tokens = ASCII_WORD.findall(text)
    for run in CHINESE_RUN.findall(text):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    return tokens


class BM25Index:
    def __init__(
        self,
        *,
        document_ids: list[str],
        document_lengths: list[int],
        postings: dict[str, list[tuple[int, int]]],
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        if len(document_ids) != len(document_lengths):
            raise ValueError("document_ids and document_lengths differ in size")
        self.document_ids = document_ids
        self.document_lengths = document_lengths
        self.postings = postings
        self.k1 = k1
        self.b = b
        self.average_document_length = (
            sum(document_lengths) / len(document_lengths) if document_lengths else 0.0
        )

    @classmethod
    def build(
        cls,
        documents: list[dict[str, Any]],
        *,
        id_field: str = "child_id",
        text_field: str = "text",
        k1: float = 1.2,
        b: float = 0.75,
    ) -> BM25Index:
        document_ids: list[str] = []
        document_lengths: list[int] = []
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for document_index, document in enumerate(documents):
            document_ids.append(str(document[id_field]))
            counts = Counter(lexical_tokens(str(document[text_field])))
            document_lengths.append(sum(counts.values()))
            for token, frequency in counts.items():
                postings[token].append((document_index, frequency))
        return cls(
            document_ids=document_ids,
            document_lengths=document_lengths,
            postings=dict(postings),
            k1=k1,
            b=b,
        )

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        if not self.document_ids or top_k <= 0:
            return []
        query_counts = Counter(lexical_tokens(query))
        scores: dict[int, float] = defaultdict(float)
        document_count = len(self.document_ids)
        for token, query_frequency in query_counts.items():
            token_postings = self.postings.get(token)
            if not token_postings:
                continue
            document_frequency = len(token_postings)
            inverse_document_frequency = math.log(
                1.0 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            for document_index, term_frequency in token_postings:
                length_normalization = (
                    1.0
                    - self.b
                    + self.b * (self.document_lengths[document_index] / self.average_document_length)
                )
                term_score = (
                    inverse_document_frequency
                    * term_frequency
                    * (self.k1 + 1.0)
                    / (term_frequency + self.k1 * length_normalization)
                )
                scores[document_index] += term_score * min(query_frequency, 3)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], self.document_ids[item[0]]))
        return [(self.document_ids[document_index], score) for document_index, score in ranked[:top_k]]

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "longcoqa-bm25-v1",
            "document_ids": self.document_ids,
            "document_lengths": self.document_lengths,
            "postings": {
                token: [[document_index, frequency] for document_index, frequency in values]
                for token, values in self.postings.items()
            },
            "parameters": {"k1": self.k1, "b": self.b},
        }

    def save(self, path: Path) -> dict[str, Any]:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_payload()
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as raw_handle:
                with gzip.GzipFile(fileobj=raw_handle, mode="wb", mtime=0) as compressed:
                    compressed.write(
                        json.dumps(
                            payload,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    )
                raw_handle.flush()
                os.fsync(raw_handle.fileno())
            os.chmod(temporary, 0o644)
            os.replace(temporary, path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
        metadata = {
            "schema_version": payload["schema_version"],
            "documents": len(self.document_ids),
            "terms": len(self.postings),
            "average_document_length": self.average_document_length,
            "parameters": payload["parameters"],
            "document_id_hash": stable_json_hash(self.document_ids),
            "artifact_checksum": sha256_file(path),
        }
        atomic_write_json(path.with_suffix(path.suffix + ".metadata.json"), metadata)
        return metadata

    @classmethod
    def load(cls, path: Path) -> BM25Index:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("schema_version") != "longcoqa-bm25-v1":
            raise ValueError(f"Unsupported BM25 index schema in {path}")
        return cls(
            document_ids=[str(value) for value in payload["document_ids"]],
            document_lengths=[int(value) for value in payload["document_lengths"]],
            postings={
                token: [(int(document_index), int(frequency)) for document_index, frequency in values]
                for token, values in payload["postings"].items()
            },
            k1=float(payload["parameters"]["k1"]),
            b=float(payload["parameters"]["b"]),
        )
