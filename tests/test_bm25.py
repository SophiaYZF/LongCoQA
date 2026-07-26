from __future__ import annotations

from pathlib import Path

from longcoqa.retrieval.bm25 import BM25Index, lexical_tokens
from longcoqa.retrieval.dense import reciprocal_rank_fusion


def test_chinese_bigram_tokenization_and_search_roundtrip(tmp_path: Path) -> None:
    documents = [
        {"child_id": "one", "text": "哈利拿到了魔法石。"},
        {"child_id": "two", "text": "罗恩正在下巫师棋。"},
        {"child_id": "three", "text": "赫敏查阅了图书馆的书。"},
    ]
    index = BM25Index.build(documents)

    assert "魔法" in lexical_tokens(documents[0]["text"])
    assert index.search("谁拿到了魔法石？", top_k=1)[0][0] == "one"

    path = tmp_path / "bm25.json.gz"
    metadata = index.save(path)
    restored = BM25Index.load(path)
    assert metadata["documents"] == 3
    assert restored.search("巫师棋", top_k=1)[0][0] == "two"


def test_reciprocal_rank_fusion_combines_sparse_and_dense_ranks() -> None:
    dense = [("a", 0.9), ("b", 0.8)]
    sparse = [("b", 10.0), ("c", 9.0)]

    fused = reciprocal_rank_fusion(dense, sparse, top_k=3, rrf_k=60)

    assert fused[0][0] == "b"
    assert {document_id for document_id, _ in fused} == {"a", "b", "c"}
