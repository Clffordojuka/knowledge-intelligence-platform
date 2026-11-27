from uuid import UUID, uuid4

import pytest

from app.retrieval import hybrid_search
from app.retrieval.bm25_search import (
    BM25RetrievalError,
    BM25SearchResult,
)
from app.retrieval.fusion import HybridSearchResult
from app.retrieval.hybrid_search import (
    HybridRetrievalError,
    search_hybrid_chunks,
)
from app.retrieval.reranking import RerankingError
from app.retrieval.semantic_search import (
    RetrievalError,
    SearchResult,
)


def make_semantic_result(
    chunk_id: UUID | None = None,
    document_id: UUID | None = None,
    chunk_index: int = 0,
    content: str = "Sample policy content.",
    score: float = 0.90,
    section_title: str | None = None,
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id or uuid4(),
        document_id=document_id or uuid4(),
        chunk_index=chunk_index,
        content=content,
        score=score,
        filename="credit_policy.txt",
        page_start=1,
        page_end=2,
        section_title=section_title,
    )


def make_bm25_result(
    chunk_id: UUID | None = None,
    document_id: UUID | None = None,
    chunk_index: int = 0,
    content: str = "Sample policy content.",
    score: float = 4.5,
    section_title: str | None = None,
) -> BM25SearchResult:
    return BM25SearchResult(
        chunk_id=chunk_id or uuid4(),
        document_id=document_id or uuid4(),
        chunk_index=chunk_index,
        content=content,
        score=score,
        filename="credit_policy.txt",
        page_start=1,
        page_end=2,
        section_title=section_title,
    )


def make_hybrid_result(
    chunk_id: UUID | None = None,
    document_id: UUID | None = None,
    content: str = "Sample policy content.",
    rrf_score: float = 0.03,
    semantic_score: float | None = 0.90,
    bm25_score: float | None = 4.5,
    section_title: str | None = None,
) -> HybridSearchResult:
    return HybridSearchResult(
        chunk_id=chunk_id or uuid4(),
        document_id=document_id or uuid4(),
        chunk_index=0,
        content=content,
        filename="credit_policy.txt",
        page_start=1,
        page_end=2,
        rrf_score=rrf_score,
        semantic_score=semantic_score,
        bm25_score=bm25_score,
        section_title=section_title,
    )


def test_hybrid_search_combines_semantic_and_bm25_results(monkeypatch):
    chunk_id = uuid4()
    document_id = uuid4()

    semantic_result = make_semantic_result(
        chunk_id=chunk_id,
        document_id=document_id,
    )
    bm25_result = make_bm25_result(
        chunk_id=chunk_id,
        document_id=document_id,
    )

    monkeypatch.setattr(
        hybrid_search,
        "search_similar_chunks",
        lambda **kwargs: [semantic_result],
    )
    monkeypatch.setattr(
        hybrid_search,
        "search_bm25",
        lambda **kwargs: [bm25_result],
    )

    results = search_hybrid_chunks(
        db=object(),
        query="credit policy requirements",
    )

    assert len(results) == 1
    assert results[0].chunk_id == chunk_id
    assert results[0].document_id == document_id
    assert results[0].semantic_score == pytest.approx(0.90)
    assert results[0].bm25_score == pytest.approx(4.5)
    assert results[0].rrf_score == pytest.approx(2 / 61)


def test_hybrid_search_forwards_query_instruction_to_semantic_only(
    monkeypatch,
):
    instruction = (
        "Represent this sentence for searching relevant passages: "
    )
    captured = {}

    def fake_semantic_search(**kwargs):
        captured["semantic_instruction"] = kwargs["query_instruction"]
        return []

    def fake_bm25_search(**kwargs):
        captured["bm25_query"] = kwargs["query"]
        return []

    monkeypatch.setattr(
        hybrid_search,
        "search_similar_chunks",
        fake_semantic_search,
    )
    monkeypatch.setattr(
        hybrid_search,
        "search_bm25",
        fake_bm25_search,
    )

    search_hybrid_chunks(
        db=object(),
        query="What are the credit approval requirements?",
        query_instruction=instruction,
    )

    assert captured["semantic_instruction"] == instruction
    assert captured["bm25_query"] == (
        "What are the credit approval requirements?"
    )


def test_hybrid_search_passes_none_instruction_by_default(monkeypatch):
    captured = {}

    def fake_semantic_search(**kwargs):
        captured["query_instruction"] = kwargs["query_instruction"]
        return []

    monkeypatch.setattr(
        hybrid_search,
        "search_similar_chunks",
        fake_semantic_search,
    )
    monkeypatch.setattr(
        hybrid_search,
        "search_bm25",
        lambda **kwargs: [],
    )

    search_hybrid_chunks(
        db=object(),
        query="credit policy",
    )

    assert captured["query_instruction"] is None


@pytest.mark.parametrize(
    "instruction",
    [
        "",
        "   ",
        123,
        True,
        [],
    ],
)
def test_hybrid_search_rejects_invalid_query_instructions(instruction):
    with pytest.raises(ValueError, match="Query instruction"):
        search_hybrid_chunks(
            db=object(),
            query="credit policy",
            query_instruction=instruction,
        )


def test_hybrid_search_uses_default_candidate_limit(monkeypatch):
    captured = {}

    def fake_semantic_search(**kwargs):
        captured["semantic_limit"] = kwargs["limit"]
        return []

    def fake_bm25_search(**kwargs):
        captured["bm25_limit"] = kwargs["limit"]
        return []

    monkeypatch.setattr(
        hybrid_search,
        "search_similar_chunks",
        fake_semantic_search,
    )
    monkeypatch.setattr(
        hybrid_search,
        "search_bm25",
        fake_bm25_search,
    )

    search_hybrid_chunks(
        db=object(),
        query="credit policy",
        limit=3,
    )

    assert captured["semantic_limit"] == 10
    assert captured["bm25_limit"] == 10


def test_hybrid_search_uses_custom_candidate_limit(monkeypatch):
    captured = {}

    def fake_semantic_search(**kwargs):
        captured["semantic_limit"] = kwargs["limit"]
        return []

    def fake_bm25_search(**kwargs):
        captured["bm25_limit"] = kwargs["limit"]
        return []

    monkeypatch.setattr(
        hybrid_search,
        "search_similar_chunks",
        fake_semantic_search,
    )
    monkeypatch.setattr(
        hybrid_search,
        "search_bm25",
        fake_bm25_search,
    )

    search_hybrid_chunks(
        db=object(),
        query="credit policy",
        limit=5,
        candidate_limit=20,
    )

    assert captured["semantic_limit"] == 20
    assert captured["bm25_limit"] == 20


def test_hybrid_search_forwards_document_filter(monkeypatch):
    document_id = uuid4()
    captured = {}

    def fake_semantic_search(**kwargs):
        captured["semantic_document_id"] = kwargs["document_id"]
        return []

    def fake_bm25_search(**kwargs):
        captured["bm25_document_id"] = kwargs["document_id"]
        return []

    monkeypatch.setattr(
        hybrid_search,
        "search_similar_chunks",
        fake_semantic_search,
    )
    monkeypatch.setattr(
        hybrid_search,
        "search_bm25",
        fake_bm25_search,
    )

    search_hybrid_chunks(
        db=object(),
        query="credit policy",
        document_id=document_id,
    )

    assert captured["semantic_document_id"] == document_id
    assert captured["bm25_document_id"] == document_id


def test_hybrid_search_forwards_multiple_document_filters(monkeypatch):
    document_ids = [uuid4(), uuid4()]
    captured = {}

    def fake_semantic_search(**kwargs):
        captured["semantic_document_ids"] = kwargs["document_ids"]
        return []

    def fake_bm25_search(**kwargs):
        captured["bm25_document_ids"] = kwargs["document_ids"]
        return []

    monkeypatch.setattr(
        hybrid_search,
        "search_similar_chunks",
        fake_semantic_search,
    )
    monkeypatch.setattr(
        hybrid_search,
        "search_bm25",
        fake_bm25_search,
    )

    search_hybrid_chunks(
        db=object(),
        query="credit policy",
        document_ids=document_ids,
    )

    assert captured["semantic_document_ids"] == document_ids
    assert captured["bm25_document_ids"] == document_ids


def test_hybrid_search_returns_empty_list_when_no_results(monkeypatch):
    monkeypatch.setattr(
        hybrid_search,
        "search_similar_chunks",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        hybrid_search,
        "search_bm25",
        lambda **kwargs: [],
    )

    results = search_hybrid_chunks(
        db=object(),
        query="nonexistent policy",
    )

    assert results == []


def test_hybrid_search_returns_semantic_only_results(monkeypatch):
    semantic_result = make_semantic_result()

    monkeypatch.setattr(
        hybrid_search,
        "search_similar_chunks",
        lambda **kwargs: [semantic_result],
    )
    monkeypatch.setattr(
        hybrid_search,
        "search_bm25",
        lambda **kwargs: [],
    )

    results = search_hybrid_chunks(
        db=object(),
        query="credit policy",
    )

    assert len(results) == 1
    assert results[0].chunk_id == semantic_result.chunk_id
    assert results[0].semantic_score == semantic_result.score
    assert results[0].bm25_score is None


def test_hybrid_search_returns_bm25_only_results(monkeypatch):
    bm25_result = make_bm25_result()

    monkeypatch.setattr(
        hybrid_search,
        "search_similar_chunks",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        hybrid_search,
        "search_bm25",
        lambda **kwargs: [bm25_result],
    )

    results = search_hybrid_chunks(
        db=object(),
        query="credit policy",
    )

    assert len(results) == 1
    assert results[0].chunk_id == bm25_result.chunk_id
    assert results[0].semantic_score is None
    assert results[0].bm25_score == bm25_result.score


def test_hybrid_search_wraps_semantic_retrieval_errors(monkeypatch):
    def failing_semantic_search(**kwargs):
        raise RetrievalError("Qdrant unavailable.")

    monkeypatch.setattr(
        hybrid_search,
        "search_similar_chunks",
        failing_semantic_search,
    )
    monkeypatch.setattr(
        hybrid_search,
        "search_bm25",
        lambda **kwargs: [],
    )

    with pytest.raises(HybridRetrievalError) as exc_info:
        search_hybrid_chunks(
            db=object(),
            query="credit policy",
        )

    assert isinstance(exc_info.value.__cause__, RetrievalError)


def test_hybrid_search_wraps_bm25_retrieval_errors(monkeypatch):
    def failing_bm25_search(**kwargs):
        raise BM25RetrievalError("Database unavailable.")

    monkeypatch.setattr(
        hybrid_search,
        "search_similar_chunks",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        hybrid_search,
        "search_bm25",
        failing_bm25_search,
    )

    with pytest.raises(HybridRetrievalError) as exc_info:
        search_hybrid_chunks(
            db=object(),
            query="credit policy",
        )

    assert isinstance(exc_info.value.__cause__, BM25RetrievalError)


def test_hybrid_search_wraps_unexpected_errors(monkeypatch):
    def failing_semantic_search(**kwargs):
        raise RuntimeError("Unexpected failure.")

    monkeypatch.setattr(
        hybrid_search,
        "search_similar_chunks",
        failing_semantic_search,
    )

    with pytest.raises(HybridRetrievalError) as exc_info:
        search_hybrid_chunks(
            db=object(),
            query="credit policy",
        )

    assert isinstance(exc_info.value.__cause__, RuntimeError)


@pytest.mark.parametrize("query", ["", "   ", None, 123])
def test_hybrid_search_rejects_invalid_queries(query):
    with pytest.raises(ValueError, match="non-empty string"):
        search_hybrid_chunks(
            db=object(),
            query=query,
        )


@pytest.mark.parametrize("limit", [0, -1, 1.5, True])
def test_hybrid_search_rejects_invalid_result_limits(limit):
    with pytest.raises(ValueError, match="positive integer"):
        search_hybrid_chunks(
            db=object(),
            query="credit policy",
            limit=limit,
        )


@pytest.mark.parametrize("candidate_limit", [0, -1, 1.5, True])
def test_hybrid_search_rejects_invalid_candidate_limits(candidate_limit):
    with pytest.raises(ValueError, match="positive integer"):
        search_hybrid_chunks(
            db=object(),
            query="credit policy",
            candidate_limit=candidate_limit,
        )


def test_hybrid_search_rejects_candidate_limit_smaller_than_result_limit():
    with pytest.raises(ValueError, match="cannot be smaller"):
        search_hybrid_chunks(
            db=object(),
            query="credit policy",
            limit=10,
            candidate_limit=5,
        )


def test_hybrid_search_invokes_reranker_when_enabled(monkeypatch):
    candidates = [
        make_hybrid_result(content="First candidate."),
        make_hybrid_result(content="Second candidate."),
    ]
    captured = {}

    def fake_fusion(**kwargs):
        captured["fusion_limit"] = kwargs["limit"]
        return candidates

    def fake_reranking(**kwargs):
        captured["query"] = kwargs["query"]
        captured["candidates"] = kwargs["candidates"]
        captured["top_k"] = kwargs["top_k"]
        captured["reranker"] = kwargs["reranker"]
        return list(reversed(kwargs["candidates"]))[:kwargs["top_k"]]

    fake_model = object()

    monkeypatch.setattr(
        hybrid_search,
        "search_similar_chunks",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        hybrid_search,
        "search_bm25",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        hybrid_search,
        "reciprocal_rank_fusion",
        fake_fusion,
    )
    monkeypatch.setattr(
        hybrid_search,
        "rerank_candidates",
        fake_reranking,
    )

    results = search_hybrid_chunks(
        db=object(),
        query="credit policy requirements",
        limit=1,
        candidate_limit=5,
        rerank=True,
        reranker=fake_model,
    )

    assert captured["fusion_limit"] == 5
    assert captured["query"] == "credit policy requirements"
    assert captured["candidates"] == candidates
    assert captured["top_k"] == 1
    assert captured["reranker"] is fake_model
    assert len(results) == 1
    assert results[0] == candidates[1]


def test_hybrid_search_uses_default_candidate_pool_when_reranking(
    monkeypatch,
):
    captured = {}
    candidates = [make_hybrid_result()]

    def fake_fusion(**kwargs):
        captured["fusion_limit"] = kwargs["limit"]
        return candidates

    monkeypatch.setattr(
        hybrid_search,
        "search_similar_chunks",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        hybrid_search,
        "search_bm25",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        hybrid_search,
        "reciprocal_rank_fusion",
        fake_fusion,
    )
    monkeypatch.setattr(
        hybrid_search,
        "rerank_candidates",
        lambda **kwargs: kwargs["candidates"],
    )

    search_hybrid_chunks(
        db=object(),
        query="credit policy",
        limit=3,
        rerank=True,
    )

    assert captured["fusion_limit"] == 10


def test_hybrid_search_does_not_rerank_when_disabled(monkeypatch):
    candidates = [make_hybrid_result()]
    captured = {}

    def fake_fusion(**kwargs):
        captured["fusion_limit"] = kwargs["limit"]
        return candidates

    def unexpected_reranking(**kwargs):
        pytest.fail("Reranking should not be called when disabled.")

    monkeypatch.setattr(
        hybrid_search,
        "search_similar_chunks",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        hybrid_search,
        "search_bm25",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        hybrid_search,
        "reciprocal_rank_fusion",
        fake_fusion,
    )
    monkeypatch.setattr(
        hybrid_search,
        "rerank_candidates",
        unexpected_reranking,
    )

    results = search_hybrid_chunks(
        db=object(),
        query="credit policy",
        limit=3,
        candidate_limit=10,
    )

    assert captured["fusion_limit"] == 3
    assert results == candidates


def test_hybrid_search_wraps_reranking_errors(monkeypatch):
    def failing_reranking(**kwargs):
        raise RerankingError("Cross-encoder inference failed.")

    monkeypatch.setattr(
        hybrid_search,
        "search_similar_chunks",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        hybrid_search,
        "search_bm25",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        hybrid_search,
        "rerank_candidates",
        failing_reranking,
    )

    with pytest.raises(HybridRetrievalError) as exc_info:
        search_hybrid_chunks(
            db=object(),
            query="credit policy",
            rerank=True,
        )

    assert isinstance(exc_info.value.__cause__, RerankingError)


@pytest.mark.parametrize("rerank", ["true", 1, None, [], object()])
def test_hybrid_search_rejects_invalid_rerank_values(rerank):
    with pytest.raises(ValueError, match="boolean"):
        search_hybrid_chunks(
            db=object(),
            query="credit policy",
            rerank=rerank,
        )


def test_hybrid_search_returns_empty_results_when_reranking_empty_candidates(
    monkeypatch,
):
    monkeypatch.setattr(
        hybrid_search,
        "search_similar_chunks",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        hybrid_search,
        "search_bm25",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        hybrid_search,
        "rerank_candidates",
        lambda **kwargs: [],
    )

    results = search_hybrid_chunks(
        db=object(),
        query="credit policy",
        rerank=True,
    )

    assert results == []

def test_hybrid_search_preserves_section_metadata(monkeypatch):
    chunk_id = uuid4()
    document_id = uuid4()
    section_title = "2. Credit Assessment"

    semantic_result = make_semantic_result(
        chunk_id=chunk_id,
        document_id=document_id,
        section_title=section_title,
    )

    bm25_result = make_bm25_result(
        chunk_id=chunk_id,
        document_id=document_id,
        section_title=section_title,
    )

    monkeypatch.setattr(
        hybrid_search,
        "search_similar_chunks",
        lambda **kwargs: [semantic_result],
    )
    monkeypatch.setattr(
        hybrid_search,
        "search_bm25",
        lambda **kwargs: [bm25_result],
    )

    results = search_hybrid_chunks(
        db=object(),
        query="credit assessment requirements",
    )

    assert len(results) == 1
    assert results[0].section_title == section_title