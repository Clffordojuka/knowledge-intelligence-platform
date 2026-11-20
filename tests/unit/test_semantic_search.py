from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.core.config import settings
from app.retrieval.semantic_search import (
    RetrievalError,
    SearchResult,
    search_similar_chunks,
)


class FakeEmbeddingService:
    """Provide deterministic query embeddings for unit tests."""

    def __init__(self, dimension=None, should_fail=False):
        self.vector_dimension = dimension or settings.embedding_dimension
        self.should_fail = should_fail
        self.received_query = None

    def embed_query(self, query):
        self.received_query = query

        if self.should_fail:
            raise RuntimeError("Embedding generation failed.")

        return [0.1] * self.vector_dimension


class FakeQdrantClient:
    """Capture Qdrant search requests and return controlled results."""

    def __init__(self, points=None, should_fail=False):
        self.points = points or []
        self.should_fail = should_fail
        self.received_arguments = None

    def query_points(self, **kwargs):
        self.received_arguments = kwargs

        if self.should_fail:
            raise RuntimeError("Qdrant connection failed.")

        return SimpleNamespace(points=self.points)


def make_point(
    document_id=None,
    chunk_id=None,
    chunk_index=0,
    content="A sample document chunk.",
    score=0.85,
):
    """Create a Qdrant-like scored point."""

    document_id = document_id or uuid4()
    chunk_id = chunk_id or uuid4()

    return SimpleNamespace(
        id=str(chunk_id),
        score=score,
        payload={
            "chunk_id": str(chunk_id),
            "document_id": str(document_id),
            "chunk_index": chunk_index,
            "content": content,
            "filename": "credit_policy.txt",
            "page_start": 1,
            "page_end": 2,
        },
    )


@pytest.fixture
def embedding_service():
    return FakeEmbeddingService()


@pytest.fixture
def qdrant_client():
    return FakeQdrantClient()


def test_successful_semantic_search(embedding_service, qdrant_client):
    document_id = uuid4()
    chunk_id = uuid4()

    qdrant_client.points = [
        make_point(
            document_id=document_id,
            chunk_id=chunk_id,
            chunk_index=2,
            content="Credit approval requirements.",
            score=0.91,
        )
    ]

    results = search_similar_chunks(
        query="What are the credit approval requirements?",
        limit=5,
        client=qdrant_client,
        embedding_service=embedding_service,
    )

    assert len(results) == 1

    result = results[0]

    assert isinstance(result, SearchResult)
    assert result.document_id == document_id
    assert result.chunk_id == chunk_id
    assert result.chunk_index == 2
    assert result.content == "Credit approval requirements."
    assert result.score == pytest.approx(0.91)
    assert result.filename == "credit_policy.txt"
    assert result.page_start == 1
    assert result.page_end == 2

    assert embedding_service.received_query == (
        "What are the credit approval requirements?"
    )

    assert qdrant_client.received_arguments["limit"] == 5
    assert qdrant_client.received_arguments["with_payload"] is True
    assert qdrant_client.received_arguments["with_vectors"] is False


def test_semantic_search_returns_empty_list(
    embedding_service,
    qdrant_client,
):
    results = search_similar_chunks(
        query="Unknown policy topic",
        client=qdrant_client,
        embedding_service=embedding_service,
    )

    assert results == []


def test_semantic_search_filters_by_document(
    embedding_service,
    qdrant_client,
):
    document_id = uuid4()

    search_similar_chunks(
        query="Find policy information",
        document_id=document_id,
        client=qdrant_client,
        embedding_service=embedding_service,
    )

    query_filter = qdrant_client.received_arguments["query_filter"]

    assert query_filter is not None
    assert len(query_filter.must) == 1
    assert query_filter.must[0].key == "document_id"
    assert query_filter.must[0].match.value == str(document_id)


def test_semantic_search_without_document_filter(
    embedding_service,
    qdrant_client,
):
    search_similar_chunks(
        query="Find policy information",
        client=qdrant_client,
        embedding_service=embedding_service,
    )

    assert qdrant_client.received_arguments["query_filter"] is None


def test_semantic_search_passes_score_threshold(
    embedding_service,
    qdrant_client,
):
    search_similar_chunks(
        query="Find relevant information",
        score_threshold=0.65,
        client=qdrant_client,
        embedding_service=embedding_service,
    )

    assert qdrant_client.received_arguments["score_threshold"] == 0.65


@pytest.mark.parametrize(
    "query",
    [
        "",
        "   ",
        None,
        123,
    ],
)
def test_semantic_search_rejects_invalid_query(query):
    with pytest.raises(ValueError, match="non-empty"):
        search_similar_chunks(query=query)


@pytest.mark.parametrize(
    "limit",
    [
        0,
        -1,
        1.5,
        True,
        "5",
    ],
)
def test_semantic_search_rejects_invalid_limit(limit):
    with pytest.raises(ValueError, match="positive integer"):
        search_similar_chunks(query="Find policy information", limit=limit)


@pytest.mark.parametrize(
    "threshold",
    [
        -1.1,
        1.1,
        float("nan"),
        float("inf"),
        True,
        "0.5",
    ],
)
def test_semantic_search_rejects_invalid_score_threshold(threshold):
    with pytest.raises(ValueError, match="Score threshold"):
        search_similar_chunks(
            query="Find policy information",
            score_threshold=threshold,
        )


def test_semantic_search_rejects_malformed_payload(
    embedding_service,
    qdrant_client,
):
    qdrant_client.points = [
        SimpleNamespace(
            id="invalid-point",
            score=0.85,
            payload={
                "chunk_id": "not-a-valid-uuid",
                "document_id": str(uuid4()),
                "chunk_index": 0,
                "content": "Some content.",
            },
        )
    ]

    with pytest.raises(RetrievalError, match="Invalid Qdrant result payload"):
        search_similar_chunks(
            query="Find policy information",
            client=qdrant_client,
            embedding_service=embedding_service,
        )


def test_semantic_search_rejects_empty_chunk_content(
    embedding_service,
    qdrant_client,
):
    point = make_point(content="   ")
    qdrant_client.points = [point]

    with pytest.raises(RetrievalError, match="Invalid Qdrant result payload"):
        search_similar_chunks(
            query="Find policy information",
            client=qdrant_client,
            embedding_service=embedding_service,
        )


def test_semantic_search_rejects_invalid_chunk_index(
    embedding_service,
    qdrant_client,
):
    point = make_point(chunk_index=-1)
    qdrant_client.points = [point]

    with pytest.raises(RetrievalError, match="Invalid Qdrant result payload"):
        search_similar_chunks(
            query="Find policy information",
            client=qdrant_client,
            embedding_service=embedding_service,
        )


def test_semantic_search_rejects_non_finite_score(
    embedding_service,
    qdrant_client,
):
    point = make_point(score=float("nan"))
    qdrant_client.points = [point]

    with pytest.raises(RetrievalError, match="Invalid Qdrant result payload"):
        search_similar_chunks(
            query="Find policy information",
            client=qdrant_client,
            embedding_service=embedding_service,
        )


def test_semantic_search_handles_embedding_failure(qdrant_client):
    failing_embedder = FakeEmbeddingService(should_fail=True)

    with pytest.raises(RetrievalError, match="Semantic search failed"):
        search_similar_chunks(
            query="Find policy information",
            client=qdrant_client,
            embedding_service=failing_embedder,
        )


def test_semantic_search_handles_qdrant_failure(embedding_service):
    failing_client = FakeQdrantClient(should_fail=True)

    with pytest.raises(RetrievalError, match="Semantic search failed"):
        search_similar_chunks(
            query="Find policy information",
            client=failing_client,
            embedding_service=embedding_service,
        )


def test_semantic_search_rejects_wrong_embedding_dimension(
    qdrant_client,
):
    wrong_dimension = FakeEmbeddingService(dimension=settings.embedding_dimension + 1)

    with pytest.raises(RetrievalError, match="dimension"):
        search_similar_chunks(
            query="Find policy information",
            client=qdrant_client,
            embedding_service=wrong_dimension,
        )

    assert qdrant_client.received_arguments is None
