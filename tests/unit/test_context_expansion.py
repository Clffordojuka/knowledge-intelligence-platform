
import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.database.base import Base
from app.database.models import Chunk, Document
from app.retrieval.context_expansion import (
    ContextExpansionError,
    expand_retrieved_context,
)
from app.retrieval.fusion import HybridSearchResult


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        yield session

    Base.metadata.drop_all(engine)
    engine.dispose()


def create_document(db, *, status="completed"):
    document = Document(
        filename="test_policy.txt",
        content_type="text/plain",
        file_size_bytes=100,
        sha256=uuid.uuid4().hex,
        storage_path="test/test_policy.txt",
        status=status,
        indexing_status="completed",
    )
    db.add(document)
    db.flush()
    return document


def create_chunk(
    db,
    document,
    index,
    content,
    section_title="1. Credit Assessment",
    char_start=None,
    char_end=None,
):
    chunk = Chunk(
        document_id=document.id,
        chunk_index=index,
        content=content,
        section_title=section_title,
        char_start=char_start,
        char_end=char_end,
    )
    db.add(chunk)
    db.flush()
    return chunk


def make_result(chunk, document):
    return HybridSearchResult(
        chunk_id=chunk.id,
        document_id=document.id,
        chunk_index=chunk.chunk_index,
        content=chunk.content,
        filename=document.filename,
        page_start=1,
        page_end=1,
        rrf_score=0.03,
        semantic_score=0.8,
        bm25_score=3.0,
        section_title=chunk.section_title,
    )


def test_expands_with_adjacent_chunks_and_removes_overlap(db):
    document = create_document(db)

    previous = create_chunk(
        db,
        document,
        0,
        "Alpha ",
        char_start=0,
        char_end=6,
    )
    original = create_chunk(
        db,
        document,
        1,
        "Bravo ",
        char_start=6,
        char_end=12,
    )
    following = create_chunk(
        db,
        document,
        2,
        "Charlie",
        char_start=12,
        char_end=19,
    )

    result = make_result(original, document)

    expanded = expand_retrieved_context(db, result)

    assert expanded.content == "Alpha Bravo Charlie"
    assert expanded.chunk_ids == (
        previous.id,
        original.id,
        following.id,
    )
    assert expanded.was_expanded is True
    assert expanded.retrieval_result is result
    assert expanded.section_title == original.section_title


def test_does_not_expand_across_section_boundary(db):
    document = create_document(db)

    create_chunk(
        db,
        document,
        0,
        "Previous section",
        "1. Introduction",
    )
    original = create_chunk(
        db,
        document,
        1,
        "Target passage",
        "2. Assessment",
    )
    create_chunk(
        db,
        document,
        2,
        "Next section",
        "3. Repayment",
    )

    expanded = expand_retrieved_context(
        db,
        make_result(original, document),
    )

    assert expanded.content == "Target passage"
    assert expanded.chunk_ids == (original.id,)
    assert expanded.was_expanded is False


def test_does_not_jump_to_repeated_section_title_after_a_gap(db):
    document = create_document(db)

    original = create_chunk(
        db,
        document,
        0,
        "First occurrence",
    )
    create_chunk(
        db,
        document,
        1,
        "Different section",
        "2. Other",
    )
    create_chunk(
        db,
        document,
        2,
        "Repeated heading",
        "1. Credit Assessment",
    )

    expanded = expand_retrieved_context(
        db,
        make_result(original, document),
        before=0,
        after=2,
    )

    assert expanded.content == "First occurrence"
    assert expanded.chunk_ids == (original.id,)


def test_respects_context_character_budget(db):
    document = create_document(db)

    create_chunk(db, document, 0, "Alpha ")
    original = create_chunk(db, document, 1, "Bravo ")
    create_chunk(db, document, 2, "Charlie")

    expanded = expand_retrieved_context(
        db,
        make_result(original, document),
        max_context_chars=14,
    )

    assert expanded.content == "Alpha \n\nBravo "
    assert len(expanded.content) <= 14
    assert original.id in expanded.chunk_ids


def test_preserves_original_chunk_when_it_exceeds_budget(db):
    document = create_document(db)

    original = create_chunk(
        db,
        document,
        0,
        "This original passage exceeds the configured budget.",
    )
    create_chunk(db, document, 1, "Neighboring content")

    expanded = expand_retrieved_context(
        db,
        make_result(original, document),
        max_context_chars=10,
    )

    assert expanded.content == original.content
    assert expanded.chunk_ids == (original.id,)
    assert expanded.was_expanded is False


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"before": -1}, "before"),
        ({"after": -1}, "after"),
        ({"max_context_chars": 0}, "Maximum context"),
    ],
)
def test_rejects_invalid_expansion_parameters(db, kwargs, message):
    document = create_document(db)
    original = create_chunk(db, document, 0, "Target passage")

    with pytest.raises(ValueError, match=message):
        expand_retrieved_context(
            db,
            make_result(original, document),
            **kwargs,
        )


def test_rejects_retrieval_result_missing_from_database(db):
    document = create_document(db)
    original = create_chunk(db, document, 0, "Target passage")

    result = make_result(original, document)

    db.delete(original)
    db.flush()

    with pytest.raises(ContextExpansionError):
        expand_retrieved_context(db, result)


def test_does_not_expand_from_incomplete_document(db):
    document = create_document(db, status="processing")
    original = create_chunk(db, document, 0, "Target passage")
    create_chunk(db, document, 1, "Neighboring content")

    with pytest.raises(ContextExpansionError):
        expand_retrieved_context(
            db,
            make_result(original, document),
        )


def test_supports_zero_neighbor_expansion(db):
    document = create_document(db)

    original = create_chunk(db, document, 0, "Target passage")
    create_chunk(db, document, 1, "Neighboring content")

    expanded = expand_retrieved_context(
        db,
        make_result(original, document),
        before=0,
        after=0,
    )

    assert expanded.content == "Target passage"
    assert expanded.chunk_ids == (original.id,)
    assert expanded.was_expanded is False
