from doc_rag.store import (
    collection_name,
    hit_at_k,
    reciprocal_rank,
    reciprocal_rank_fusion,
)


def test_rrf_rewards_a_document_ranked_by_both_lists() -> None:
    tied = reciprocal_rank_fusion([["a", "b"], ["b", "a"]], rrf_k=60)
    assert tied["a"] == tied["b"]
    scores = reciprocal_rank_fusion([["a", "b"], ["a", "c"]], rrf_k=60)
    assert scores["a"] > scores["b"]
    assert scores["a"] > scores["c"]


def test_hit_at_k_and_mrr_use_page_overlap() -> None:
    results = [
        {"page_start": 1, "page_end": 2},
        {"page_start": 12, "page_end": 13},
    ]
    assert hit_at_k(results, [13]) == 1.0
    assert reciprocal_rank(results, [13]) == 0.5
    assert hit_at_k(results, [9]) == 0.0
    assert reciprocal_rank(results, [9]) == 0.0
    assert reciprocal_rank(results, [1]) == 1.0


def test_collection_names_stay_chroma_safe() -> None:
    assert collection_name("PER-5100") == "PER-5100"
    assert collection_name("a/b") == "a-b"
    long_name = collection_name("A" * 80)
    assert 3 <= len(long_name) <= 63
    assert long_name[0].isalnum() and long_name[-1].isalnum()
    ipv4 = collection_name("1.2.3.4")
    assert ipv4.startswith("doc-")
    assert collection_name("..") == "doc"
