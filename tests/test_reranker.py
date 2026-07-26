from longcoqa.retrieval.reranker import rerank_candidates, select_balanced_queries


def test_select_balanced_queries() -> None:
    rows = [
        {"sample_id": f"factoid:1:{index}"} for index in range(4)
    ] + [{"sample_id": f"non_factoid:1:{index}"} for index in range(4)]
    selected = select_balanced_queries(rows, 4)
    assert [row["sample_id"].split(":")[0] for row in selected] == [
        "factoid",
        "factoid",
        "non_factoid",
        "non_factoid",
    ]


def test_rerank_candidates_is_deterministic() -> None:
    ranked = rerank_candidates(["b", "a", "c"], [0.5, 0.5, 0.1], output_k=2)
    assert ranked == [("a", 0.5), ("b", 0.5)]
