from longcoqa.retrieval.contextual import build_context_messages
from longcoqa.retrieval.evaluate import (
    build_all_ranking_results,
    build_all_retrieval_queries,
    project_rankings_for_evaluation,
)
from longcoqa.retrieval.graph_progressive import (
    build_position_neighbors,
    progressive_graph_ranking,
)
from longcoqa.retrieval.progressive import parse_subqueries


def test_contextual_prompt_contains_parent_and_child() -> None:
    parent = {
        "parent_id": "p1",
        "book_title": "书",
        "chapter_title": "章",
        "text": "完整父段落",
    }
    child = {"parent_id": "p1", "text": "目标块"}
    messages = build_context_messages([child], [parent])
    assert "完整父段落" in messages[0][1]["content"]
    assert "目标块" in messages[0][1]["content"]


def test_progressive_graph_adds_semantic_and_position_nodes() -> None:
    ranked = progressive_graph_ranking(
        [("a", 1.0), ("b", 0.5)],
        semantic_neighbors={"a": [("c", 0.9)]},
        position_neighbors={"a": {"d"}},
        seed_k=1,
        steps=1,
        top_k=10,
    )
    ids = {row[0] for row in ranked}
    assert {"a", "b", "c", "d"} <= ids


def test_position_neighbors_do_not_cross_chapters() -> None:
    rows = [
        {"chapter_id": "c1", "child_id": "a"},
        {"chapter_id": "c1", "child_id": "b"},
        {"chapter_id": "c2", "child_id": "c"},
    ]
    neighbors = build_position_neighbors(rows, window=3)
    assert neighbors["a"] == {"b"}
    assert "c" not in neighbors["a"]


def test_parse_progressive_subqueries_with_fallback() -> None:
    assert parse_subqueries('prefix {"queries":["甲","乙","甲"]}', "原问题") == ["甲", "乙"]
    assert parse_subqueries("not json", "原问题") == ["原问题"]


class _Dataset:
    def iter_samples(self):
        yield {
            "sample_id": "factoid:1:1",
            "question_type": "factoid",
            "dialogue_id": 1,
            "turn": 1,
            "question": "第一问",
            "evidence_ids": [1],
        }
        yield {
            "sample_id": "factoid:1:2",
            "question_type": "factoid",
            "dialogue_id": 1,
            "turn": 2,
            "question": "无标注问题",
            "evidence_ids": [],
        }


def test_all_retrieval_queries_keep_samples_without_gold() -> None:
    rows = build_all_retrieval_queries(_Dataset(), query_mode="history_questions")
    assert [row["sample_id"] for row in rows] == ["factoid:1:1", "factoid:1:2"]
    assert rows[1]["query"] == "第一问\n无标注问题"
    assert "gold_child_ids" not in rows[1]


def test_all_retrieval_queries_use_complete_rewrites() -> None:
    rows = build_all_retrieval_queries(
        _Dataset(),
        query_mode="rewritten_dialogue",
        rewrites={"factoid:1:1": "改写一", "factoid:1:2": "改写二"},
    )
    assert [row["query"] for row in rows] == ["改写一", "改写二"]


def test_all_ranking_results_project_to_gold_query_order() -> None:
    queries = build_all_retrieval_queries(_Dataset(), query_mode="current_question")
    rankings = [[("c1", 1.0)], [("c2", 0.5)]]
    rows = build_all_ranking_results(queries, rankings, method="dense")
    eval_queries = [
        {
            "sample_id": "factoid:1:1",
            "query": "第一问",
            "query_mode": "current_question",
            "gold_child_ids": ["c1"],
        }
    ]
    projected = project_rankings_for_evaluation(rows, eval_queries)
    assert projected == [[("c1", 1.0)]]
    assert [row["sample_id"] for row in rows] == ["factoid:1:1", "factoid:1:2"]
