
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from app.retrieval.context_assembly import (
    ContextAssemblyError,
    assemble_expanded_contexts,
)


def make_chunk(
    *,
    chunk_id=None,
    document_id=None,
    chunk_index=0,
    content="sample content",
    filename="policy.pdf",
    section_title="1. Credit Policy",
    page_start=1,
    page_end=1,
):
    chunk_id = chunk_id or uuid4()
    document_id = document_id or uuid4()

    chunk = SimpleNamespace(
        id=chunk_id,
        document_id=document_id,
        chunk_index=chunk_index,
        content=content,
        section_title=section_title,
        page_start=page_start,
        page_end=page_end,
    )

    return chunk, filename


def make_expanded_context(primary_chunk_id, chunk_ids):
    return SimpleNamespace(
        retrieval_result=SimpleNamespace(chunk_id=primary_chunk_id),
        chunk_ids=tuple(chunk_ids),
        content="Expanded context content",
        section_title="1. Credit Policy",
    )


def make_db(*rows):
    db = MagicMock()
    db.execute.return_value.all.return_value = list(rows)
    return db


def test_empty_context_returns_empty_assembly():
    db = MagicMock()

    result = assemble_expanded_contexts(
        db,
        [],
        max_context_chars=1000,
    )

    assert result.content == ""
    assert result.sources == ()
    assert result.unique_chunk_count == 0
    assert result.included_chunk_count == 0
    assert result.truncated is False
    db.execute.assert_not_called()


def test_assembles_chunks_in_retrieval_priority_order():
    document_id = uuid4()

    first, filename = make_chunk(
        document_id=document_id,
        chunk_index=0,
        content="First passage.",
    )
    second, _ = make_chunk(
        document_id=document_id,
        chunk_index=1,
        content="Second passage.",
    )

    db = make_db((first, filename), (second, filename))

    contexts = [
        make_expanded_context(first.id, [first.id, second.id]),
    ]

    result = assemble_expanded_contexts(
        db,
        contexts,
        max_context_chars=1000,
    )

    assert result.sources[0].chunk_id == first.id
    assert result.sources[1].chunk_id == second.id
    assert result.content.index("First passage.") < result.content.index(
        "Second passage."
    )


def test_deduplicates_chunk_ids_across_expanded_results():
    document_id = uuid4()

    first, filename = make_chunk(
        document_id=document_id,
        chunk_index=0,
        content="First passage.",
    )
    second, _ = make_chunk(
        document_id=document_id,
        chunk_index=1,
        content="Second passage.",
    )
    third, _ = make_chunk(
        document_id=document_id,
        chunk_index=2,
        content="Third passage.",
    )

    db = make_db(
        (first, filename),
        (second, filename),
        (third, filename),
    )

    contexts = [
        make_expanded_context(first.id, [first.id, second.id]),
        make_expanded_context(second.id, [first.id, second.id, third.id]),
    ]

    result = assemble_expanded_contexts(
        db,
        contexts,
        max_context_chars=1000,
    )

    assert result.unique_chunk_count == 3
    assert result.duplicate_chunk_count == 2
    assert len(result.sources) == 3
    assert result.content.count("First passage.") == 1
    assert result.content.count("Second passage.") == 1
    assert result.content.count("Third passage.") == 1


def test_preserves_retrieval_rank_and_primary_rank_provenance():
    document_id = uuid4()

    first, filename = make_chunk(
        document_id=document_id,
        chunk_index=0,
        content="First passage.",
    )
    second, _ = make_chunk(
        document_id=document_id,
        chunk_index=1,
        content="Second passage.",
    )

    db = make_db((first, filename), (second, filename))

    contexts = [
        make_expanded_context(first.id, [first.id, second.id]),
        make_expanded_context(second.id, [first.id, second.id]),
    ]

    result = assemble_expanded_contexts(
        db,
        contexts,
        max_context_chars=1000,
    )

    first_source = next(
        source for source in result.sources
        if source.chunk_id == first.id
    )
    second_source = next(
        source for source in result.sources
        if source.chunk_id == second.id
    )

    assert first_source.retrieval_ranks == (1, 2)
    assert first_source.primary_retrieval_ranks == (1,)
    assert second_source.retrieval_ranks == (1, 2)
    assert second_source.primary_retrieval_ranks == (2,)


def test_removes_exact_overlap_between_consecutive_chunks():
    document_id = uuid4()

    first, filename = make_chunk(
        document_id=document_id,
        chunk_index=0,
        content="The borrower must provide verified income.",
    )
    second, _ = make_chunk(
        document_id=document_id,
        chunk_index=1,
        content="verified income. Repayment capacity must be assessed.",
    )

    db = make_db((first, filename), (second, filename))

    contexts = [
        make_expanded_context(first.id, [first.id, second.id]),
    ]

    result = assemble_expanded_contexts(
        db,
        contexts,
        max_context_chars=1000,
    )

    assert result.removed_overlap_characters == len("verified income.")
    assert result.content.count("verified income.") == 1
    assert "Repayment capacity must be assessed." in result.content


def test_does_not_remove_overlap_across_different_sections():
    document_id = uuid4()

    first, filename = make_chunk(
        document_id=document_id,
        chunk_index=0,
        content="The borrower must provide verified income.",
        section_title="1. Eligibility",
    )
    second, _ = make_chunk(
        document_id=document_id,
        chunk_index=1,
        content="verified income. Repayment capacity must be assessed.",
        section_title="2. Repayment Assessment",
    )

    db = make_db((first, filename), (second, filename))

    contexts = [
        make_expanded_context(first.id, [first.id, second.id]),
    ]

    result = assemble_expanded_contexts(
        db,
        contexts,
        max_context_chars=1000,
    )

    assert result.removed_overlap_characters == 0
    assert result.content.count("verified income.") == 2


def test_global_character_budget_includes_labels_and_separators():
    document_id = uuid4()

    first, filename = make_chunk(
        document_id=document_id,
        chunk_index=0,
        content="A" * 100,
    )
    second, _ = make_chunk(
        document_id=document_id,
        chunk_index=1,
        content="B" * 100,
    )

    db = make_db((first, filename), (second, filename))

    contexts = [
        make_expanded_context(first.id, [first.id, second.id]),
    ]

    result = assemble_expanded_contexts(
        db,
        contexts,
        max_context_chars=100,
    )

    assert len(result.content) <= 100
    assert result.truncated is True
    assert result.included_chunk_count >= 1
    assert result.separator_and_label_characters > 0


def test_preserves_document_and_section_provenance():
    first_document_id = uuid4()
    second_document_id = uuid4()

    first, first_filename = make_chunk(
        document_id=first_document_id,
        chunk_index=0,
        content="Credit eligibility requirements.",
        filename="credit_policy.pdf",
        section_title="1. Eligibility",
        page_start=2,
        page_end=2,
    )
    second, second_filename = make_chunk(
        document_id=second_document_id,
        chunk_index=0,
        content="Suspicious activity must be escalated.",
        filename="compliance_policy.pdf",
        section_title="4. Escalation",
        page_start=5,
        page_end=6,
    )

    db = make_db(
        (first, first_filename),
        (second, second_filename),
    )

    contexts = [
        make_expanded_context(first.id, [first.id]),
        make_expanded_context(second.id, [second.id]),
    ]

    result = assemble_expanded_contexts(
        db,
        contexts,
        max_context_chars=1000,
    )

    assert result.sources[0].filename == "credit_policy.pdf"
    assert result.sources[0].section_title == "1. Eligibility"
    assert result.sources[0].page_start == 2
    assert result.sources[1].filename == "compliance_policy.pdf"
    assert result.sources[1].section_title == "4. Escalation"
    assert result.sources[1].page_end == 6

    assert "Document: credit_policy.pdf" in result.content
    assert "Document: compliance_policy.pdf" in result.content


def test_raises_when_an_expanded_chunk_is_missing():
    missing_chunk_id = uuid4()

    db = make_db()
    contexts = [
        make_expanded_context(missing_chunk_id, [missing_chunk_id]),
    ]

    with pytest.raises(ContextAssemblyError, match="no longer exist"):
        assemble_expanded_contexts(
            db,
            contexts,
            max_context_chars=1000,
        )


@pytest.mark.parametrize(
    ("budget", "overlap_limit"),
    [
        (0, 100),
        (-1, 100),
        (100, 0),
        (100, -1),
    ],
)
def test_rejects_nonpositive_limits(budget, overlap_limit):
    with pytest.raises(ValueError):
        assemble_expanded_contexts(
            MagicMock(),
            [],
            max_context_chars=budget,
            max_overlap_chars=overlap_limit,
        )