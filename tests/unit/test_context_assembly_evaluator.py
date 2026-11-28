
from uuid import uuid4

import pytest

from app.evaluation import chunking_invariant_evaluator as base
from app.evaluation import context_assembly_evaluator as evaluator
from app.evaluation.chunking_invariant_evaluator import (
    EvidenceTarget,
    IndexedChunk,
    SourceSpan,
    normalize_text,
)
from app.retrieval.context_assembly import AssembledContext, ContextSource


def make_chunk(
    *,
    content: str = "The borrower must provide identification.",
    normalized_content: str | None = None,
    source_start: int = 0,
    source_end: int | None = None,
    filename: str = "credit_policy.txt",
    chunk_index: int = 0,
) -> IndexedChunk:
    if normalized_content is None:
        normalized_content = normalize_text(content)

    if source_end is None:
        source_end = source_start + len(normalized_content)

    return IndexedChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        filename=filename,
        chunk_index=chunk_index,
        content=content,
        normalized_content=normalized_content,
        source_start=source_start,
        source_end=source_end,
    )


def make_assembled(
    chunk: IndexedChunk,
    *,
    included_characters: int,
    removed_overlap_characters: int = 0,
    truncated: bool = False,
) -> AssembledContext:
    source = ContextSource(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        filename=chunk.filename,
        section_title=None,
        chunk_index=chunk.chunk_index,
        page_start=1,
        page_end=1,
        retrieval_ranks=(1,),
        primary_retrieval_ranks=(1,),
        included_characters=included_characters,
        removed_overlap_characters=removed_overlap_characters,
        truncated=truncated,
    )

    return AssembledContext(
        content=chunk.content[
            removed_overlap_characters:
            removed_overlap_characters + included_characters
        ],
        sources=(source,),
        max_context_chars=12000,
        total_source_characters=len(chunk.content),
        included_content_characters=included_characters,
        separator_and_label_characters=0,
        duplicate_chunk_count=0,
        removed_overlap_characters=removed_overlap_characters,
        unique_chunk_count=1,
        included_chunk_count=1 if included_characters else 0,
        truncated=truncated,
    )


def make_evidence_target(
    chunk: IndexedChunk,
    phrase: str,
) -> EvidenceTarget:
    normalized_phrase = normalize_text(phrase)
    phrase_start = chunk.normalized_content.index(normalized_phrase)

    return EvidenceTarget(
        evidence_id="evidence-1",
        filename=chunk.filename,
        phrase=phrase,
        spans=(
            SourceSpan(
                filename=chunk.filename,
                start=chunk.source_start + phrase_start,
                end=chunk.source_start + phrase_start + len(normalized_phrase),
            ),
        ),
    )


def test_maps_complete_included_chunk_to_source_span():
    chunk = make_chunk()

    assembled = make_assembled(
        chunk,
        included_characters=len(chunk.content),
    )

    spans = evaluator.build_assembled_source_spans(
        assembled,
        {chunk.chunk_id: chunk},
    )

    assert len(spans) == 1
    assert spans[0] == SourceSpan(
        filename=chunk.filename,
        start=chunk.source_start,
        end=chunk.source_end,
    )


def test_maps_truncated_chunk_to_only_included_prefix():
    chunk = make_chunk()
    included = 18

    assembled = make_assembled(
        chunk,
        included_characters=included,
        truncated=True,
    )

    spans = evaluator.build_assembled_source_spans(
        assembled,
        {chunk.chunk_id: chunk},
    )

    expected_normalized_prefix = normalize_text(
        chunk.content[:included]
    )

    assert len(spans) == 1
    assert spans[0].start == chunk.source_start
    assert spans[0].end == (
        chunk.source_start + len(expected_normalized_prefix)
    )


def test_maps_chunk_after_removed_overlap():
    chunk = make_chunk(
        content="repeated text followed by new evidence",
        source_start=100,
    )

    removed_overlap = len("repeated text ")

    assembled = make_assembled(
        chunk,
        included_characters=len(chunk.content) - removed_overlap,
        removed_overlap_characters=removed_overlap,
    )

    spans = evaluator.build_assembled_source_spans(
        assembled,
        {chunk.chunk_id: chunk},
    )

    expected_included = normalize_text(
        chunk.content[removed_overlap:]
    )

    expected_start = chunk.normalized_content.find(expected_included)

    assert expected_start >= 0
    assert len(spans) == 1
    assert spans[0].start == chunk.source_start + expected_start
    assert spans[0].end == (
        chunk.source_start + expected_start + len(expected_included)
    )


def test_ignores_sources_with_no_included_content():
    chunk = make_chunk()

    assembled = make_assembled(
        chunk,
        included_characters=0,
    )

    spans = evaluator.build_assembled_source_spans(
        assembled,
        {chunk.chunk_id: chunk},
    )

    assert spans == []


def test_rejects_unknown_assembled_chunk():
    chunk = make_chunk()

    assembled = make_assembled(
        chunk,
        included_characters=len(chunk.content),
    )

    with pytest.raises(
        evaluator.ContextAssemblyEvaluationError,
        match="Unknown assembled chunk",
    ):
        evaluator.build_assembled_source_spans(assembled, {})


def test_rejects_inconsistent_document_metadata():
    chunk = make_chunk()

    assembled = make_assembled(
        chunk,
        included_characters=len(chunk.content),
    )

    source = assembled.sources[0]

    inconsistent_source = ContextSource(
        chunk_id=source.chunk_id,
        document_id=uuid4(),
        filename=source.filename,
        section_title=source.section_title,
        chunk_index=source.chunk_index,
        page_start=source.page_start,
        page_end=source.page_end,
        retrieval_ranks=source.retrieval_ranks,
        primary_retrieval_ranks=source.primary_retrieval_ranks,
        included_characters=source.included_characters,
        removed_overlap_characters=source.removed_overlap_characters,
        truncated=source.truncated,
    )

    inconsistent_assembled = AssembledContext(
        content=assembled.content,
        sources=(inconsistent_source,),
        max_context_chars=assembled.max_context_chars,
        total_source_characters=assembled.total_source_characters,
        included_content_characters=assembled.included_content_characters,
        separator_and_label_characters=assembled.separator_and_label_characters,
        duplicate_chunk_count=assembled.duplicate_chunk_count,
        removed_overlap_characters=assembled.removed_overlap_characters,
        unique_chunk_count=assembled.unique_chunk_count,
        included_chunk_count=assembled.included_chunk_count,
        truncated=assembled.truncated,
    )

    with pytest.raises(
        evaluator.ContextAssemblyEvaluationError,
        match="metadata",
    ):
        evaluator.build_assembled_source_spans(
            inconsistent_assembled,
            {chunk.chunk_id: chunk},
        )


def test_evidence_coverage_uses_included_text_only():
    chunk = make_chunk(
        content="The borrower must provide identification.",
    )

    included = len("The borrower must")

    assembled = make_assembled(
        chunk,
        included_characters=included,
        truncated=True,
    )

    spans = evaluator.build_assembled_source_spans(
        assembled,
        {chunk.chunk_id: chunk},
    )

    evidence = make_evidence_target(
        chunk,
        "provide identification",
    )

    assert not base.is_evidence_covered(evidence, spans)


def test_evidence_coverage_detects_complete_included_phrase():
    chunk = make_chunk(
        content="The borrower must provide identification.",
    )

    assembled = make_assembled(
        chunk,
        included_characters=len(chunk.content),
    )

    spans = evaluator.build_assembled_source_spans(
        assembled,
        {chunk.chunk_id: chunk},
    )

    evidence = make_evidence_target(
        chunk,
        "provide identification",
    )

    assert base.is_evidence_covered(evidence, spans)


def test_rejects_empty_evidence_targets():
    chunk = make_chunk()

    assembled = make_assembled(
        chunk,
        included_characters=len(chunk.content),
    )

    with pytest.raises(
        evaluator.ContextAssemblyEvaluationError,
        match="Evidence targets must not be empty",
    ):
        evaluator.calculate_assembled_evidence_metrics(
            assembled=assembled,
            chunk_index={chunk.chunk_id: chunk},
            evidence_targets=[],
        )
