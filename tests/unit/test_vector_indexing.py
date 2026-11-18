from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from app.core.config import settings
from app.vector_store import indexing
from app.vector_store.indexing import (
    DocumentNotFoundError,
    IndexingError,
    index_document,
)


@pytest.fixture
def document():
    return SimpleNamespace(
        id=uuid4(),
        filename="credit_policy.txt",
        status="completed",
        indexing_status="pending",
    )


@pytest.fixture
def chunks(document):
    return [
        SimpleNamespace(
            id=uuid4(),
            document_id=document.id,
            chunk_index=0,
            content="Credit applications require identity verification.",
            page_start=1,
            page_end=1,
        ),
        SimpleNamespace(
            id=uuid4(),
            document_id=document.id,
            chunk_index=1,
            content="Applicants must satisfy the minimum income criteria.",
            page_start=1,
            page_end=2,
        ),
    ]


@pytest.fixture
def db(document, chunks):
    session = MagicMock()
    session.get.return_value = document
    session.scalars.return_value.all.return_value = chunks
    return session


@pytest.fixture
def qdrant_client():
    return MagicMock()


@pytest.fixture
def embedding_service(chunks):
    return SimpleNamespace(
        vector_dimension=settings.embedding_dimension,
        embed_texts=MagicMock(
            return_value=[[0.1] * settings.embedding_dimension for _ in chunks]
        ),
    )


@pytest.fixture
def setup_indexing(monkeypatch, embedding_service):
    ensure_collection = MagicMock()

    monkeypatch.setattr(
        indexing,
        "get_embedding_service",
        lambda: embedding_service,
    )
    monkeypatch.setattr(
        indexing,
        "ensure_collection",
        ensure_collection,
    )

    return ensure_collection


def test_index_document_success(
    db,
    document,
    chunks,
    qdrant_client,
    embedding_service,
    setup_indexing,
):
    result = index_document(
        db,
        document.id,
        qdrant_client,
    )

    assert result.document_id == document.id
    assert result.indexed_chunks == 2
    assert result.collection_name == settings.qdrant_collection_name

    assert document.indexing_status == "completed"

    embedding_service.embed_texts.assert_called_once_with(
        [chunk.content for chunk in chunks]
    )

    setup_indexing.assert_called_once_with(qdrant_client)

    qdrant_client.upsert.assert_called_once()

    call = qdrant_client.upsert.call_args

    assert call.kwargs["collection_name"] == settings.qdrant_collection_name
    assert call.kwargs["wait"] is True

    points = call.kwargs["points"]

    assert len(points) == 2
    assert [point.id for point in points] == [str(chunk.id) for chunk in chunks]

    assert points[0].payload["document_id"] == str(document.id)
    assert points[0].payload["content"] == chunks[0].content

    assert db.commit.call_count == 2


def test_index_document_not_found(db, qdrant_client):
    db.get.return_value = None

    with pytest.raises(DocumentNotFoundError):
        index_document(db, uuid4(), qdrant_client)

    db.commit.assert_not_called()
    qdrant_client.upsert.assert_not_called()


def test_index_document_requires_completed_processing(
    db,
    document,
    qdrant_client,
):
    document.status = "processing"

    with pytest.raises(IndexingError, match="successfully processed"):
        index_document(db, document.id, qdrant_client)

    db.commit.assert_not_called()
    qdrant_client.upsert.assert_not_called()


def test_index_document_rejects_empty_chunks(
    db,
    document,
    qdrant_client,
):
    db.scalars.return_value.all.return_value = []

    with pytest.raises(IndexingError, match="without extracted chunks"):
        index_document(db, document.id, qdrant_client)

    db.commit.assert_not_called()
    qdrant_client.upsert.assert_not_called()


def test_index_document_rejects_already_processing(
    db,
    document,
    qdrant_client,
):
    document.indexing_status = "processing"

    with pytest.raises(IndexingError, match="already being indexed"):
        index_document(db, document.id, qdrant_client)

    db.commit.assert_not_called()
    qdrant_client.upsert.assert_not_called()


def test_index_document_embedding_failure(
    db,
    document,
    qdrant_client,
    embedding_service,
    setup_indexing,
):
    embedding_service.embed_texts.side_effect = RuntimeError(
        "Embedding model unavailable."
    )

    with pytest.raises(IndexingError, match="Failed to index"):
        index_document(db, document.id, qdrant_client)

    assert document.indexing_status == "failed"
    db.rollback.assert_called_once()
    assert db.commit.call_count == 2
    qdrant_client.upsert.assert_not_called()


def test_index_document_qdrant_failure(
    db,
    document,
    qdrant_client,
    setup_indexing,
):
    qdrant_client.upsert.side_effect = RuntimeError("Qdrant unavailable.")

    with pytest.raises(IndexingError, match="Failed to index"):
        index_document(db, document.id, qdrant_client)

    assert document.indexing_status == "failed"
    db.rollback.assert_called_once()
    assert db.commit.call_count == 2


def test_index_document_rejects_wrong_vector_dimension(
    db,
    document,
    qdrant_client,
    embedding_service,
    setup_indexing,
):
    embedding_service.vector_dimension = settings.embedding_dimension + 1

    with pytest.raises(IndexingError, match="dimension"):
        index_document(db, document.id, qdrant_client)

    assert document.indexing_status == "failed"
    qdrant_client.upsert.assert_not_called()


def test_index_document_rejects_embedding_count_mismatch(
    db,
    document,
    chunks,
    qdrant_client,
    embedding_service,
    setup_indexing,
):
    embedding_service.embed_texts.return_value = [[0.1] * settings.embedding_dimension]

    with pytest.raises(IndexingError, match="Embedding count"):
        index_document(db, document.id, qdrant_client)

    assert document.indexing_status == "failed"
    qdrant_client.upsert.assert_not_called()


def test_index_document_is_repeatable(
    db,
    document,
    chunks,
    qdrant_client,
    setup_indexing,
):
    first_result = index_document(
        db,
        document.id,
        qdrant_client,
    )

    first_points = qdrant_client.upsert.call_args.kwargs["points"]
    first_ids = [point.id for point in first_points]

    second_result = index_document(
        db,
        document.id,
        qdrant_client,
    )

    second_points = qdrant_client.upsert.call_args.kwargs["points"]
    second_ids = [point.id for point in second_points]

    assert first_result.indexed_chunks == second_result.indexed_chunks
    assert first_ids == second_ids
    assert first_ids == [str(chunk.id) for chunk in chunks]

    assert qdrant_client.upsert.call_count == 2
