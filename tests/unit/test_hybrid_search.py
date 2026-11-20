from uuid import UUID, uuid4

import pytest

from app.retrieval import hybrid_search
from app.retrieval.bm25_search import (
    BM25RetrievalError,
    BM25SearchResult,
)
from app.retrieval.hybrid_search import (
    HybridRetrievalError,
    search_hybrid_chunks,
)
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
    )


def make_bm25_result(
    chunk_id: UUID | None = None,
    document_id: UUID | None = None,
    chunk_index: int = 0,
    content: str = "Sample policy content.",
    score: float = 4.5,
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
