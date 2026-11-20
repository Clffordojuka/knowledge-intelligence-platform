from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.database.base import Base
from app.database.models import Chunk, Document
from app.retrieval.bm25_search import (
    BM25SearchResult,
    search_bm25,
    tokenize,
)


@pytest.fixture
def db_session():
    """Create an isolated in-memory database for BM25 tests."""

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    Base.metadata.create_all(engine)

    with Session(engine) as session:
        yield session

    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture
def document_with_chunks(db_session):
    """Create a completed document with searchable chunks."""

    document = Document(
        id=uuid4(),
        filename="credit_policy.txt",
        content_type="text/plain",
        file_size_bytes=500,
        sha256=uuid4().hex,
        storage_path="test/credit_policy.txt",
        status="completed",
        indexing_status="completed",
    )

    db_session.add(document)
    db_session.flush()

    contents = [
        (
            "Credit approval requires an income assessment "
            "and verification of employment."
        ),
        ("The loan repayment period must not exceed the approved repayment schedule."),
        (
            "Credit risk assessment evaluates borrower "
            "income, repayment capacity, and existing debt."
        ),
    ]

    chunks = []

    for index, content in enumerate(contents):
        chunk = Chunk(
            id=uuid4(),
            document_id=document.id,
            chunk_index=index,
            content=content,
            page_start=index + 1,
            page_end=index + 1,
        )
        chunks.append(chunk)
        db_session.add(chunk)

    db_session.commit()

    return document, chunks


def test_tokenize_normalizes_text():
    tokens = tokenize("Credit POLICY No. 123 - Loan_Approval!")

    assert tokens == [
        "credit",
        "policy",
        "no",
        "123",
        "loan_approval",
    ]


def test_tokenize_preserves_hyphenated_terms():
    tokens = tokenize("credit-policy risk-assessment")

    assert tokens == [
        "credit-policy",
        "risk-assessment",
    ]


def test_tokenize_empty_text():
    assert tokenize("") == []
    assert tokenize("   ") == []


def test_successful_bm25_search(db_session, document_with_chunks):
    document, chunks = document_with_chunks

    results = search_bm25(
        db=db_session,
        query="credit approval income",
        limit=3,
    )

    assert len(results) > 0
    assert all(isinstance(result, BM25SearchResult) for result in results)

    assert results[0].document_id == document.id
    assert results[0].chunk_id == chunks[0].id
    assert results[0].score > 0


def test_bm25_returns_results_in_descending_score_order(
    db_session,
    document_with_chunks,
):
    results = search_bm25(
        db=db_session,
        query="credit income assessment",
        limit=3,
    )

    scores = [result.score for result in results]

    assert scores == sorted(scores, reverse=True)


def test_bm25_respects_result_limit(
    db_session,
    document_with_chunks,
):
    results = search_bm25(
        db=db_session,
        query="credit income",
        limit=1,
    )

    assert len(results) == 1


def test_bm25_filters_by_document(
    db_session,
    document_with_chunks,
):
    document, _ = document_with_chunks

    other_document = Document(
        id=uuid4(),
        filename="other_policy.txt",
        content_type="text/plain",
        file_size_bytes=200,
        sha256=uuid4().hex,
        storage_path="test/other_policy.txt",
        status="completed",
        indexing_status="completed",
    )

    db_session.add(other_document)
    db_session.flush()

    other_chunk = Chunk(
        id=uuid4(),
        document_id=other_document.id,
        chunk_index=0,
        content="Credit approval requires identity verification.",
        page_start=1,
        page_end=1,
    )

    db_session.add(other_chunk)
    db_session.commit()

    results = search_bm25(
        db=db_session,
        query="credit approval",
        document_id=document.id,
        limit=10,
    )

    assert results
    assert all(result.document_id == document.id for result in results)


def test_bm25_excludes_unprocessed_documents(db_session):
    document = Document(
        id=uuid4(),
        filename="pending_policy.txt",
        content_type="text/plain",
        file_size_bytes=200,
        sha256=uuid4().hex,
        storage_path="test/pending_policy.txt",
        status="processing",
        indexing_status="pending",
    )

    db_session.add(document)
    db_session.flush()

    chunk = Chunk(
        id=uuid4(),
        document_id=document.id,
        chunk_index=0,
        content="Credit approval and income assessment.",
        page_start=1,
        page_end=1,
    )

    db_session.add(chunk)
    db_session.commit()

    results = search_bm25(
        db=db_session,
        query="credit approval",
    )

    assert results == []


def test_bm25_returns_empty_list_for_unmatched_query(
    db_session,
    document_with_chunks,
):
    results = search_bm25(
        db=db_session,
        query="quantum astrophysics",
    )

    assert results == []


def test_bm25_returns_empty_list_for_empty_corpus(db_session):
    results = search_bm25(
        db=db_session,
        query="credit approval",
    )

    assert results == []


def test_bm25_returns_empty_list_for_tokenless_query(
    db_session,
    document_with_chunks,
):
    results = search_bm25(
        db=db_session,
        query="!!! ... ???",
    )

    assert results == []


@pytest.mark.parametrize(
    "query",
    [
        "",
        "   ",
        None,
        123,
    ],
)
def test_bm25_rejects_invalid_query(db_session, query):
    with pytest.raises(ValueError, match="non-empty"):
        search_bm25(db=db_session, query=query)


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
def test_bm25_rejects_invalid_limit(db_session, limit):
    with pytest.raises(ValueError, match="positive integer"):
        search_bm25(
            db=db_session,
            query="credit approval",
            limit=limit,
        )


def test_bm25_result_preserves_chunk_provenance(
    db_session,
    document_with_chunks,
):
    document, chunks = document_with_chunks

    results = search_bm25(
        db=db_session,
        query="credit approval income",
    )

    result = next(result for result in results if result.chunk_id == chunks[0].id)

    assert result.document_id == document.id
    assert result.chunk_id == chunks[0].id
    assert result.chunk_index == 0
    assert result.filename == "credit_policy.txt"
    assert result.page_start == 1
    assert result.page_end == 1
    assert result.content == chunks[0].content
