
from uuid import uuid4

import pytest

from app.evaluation import chunking_invariant_evaluator as base
from app.evaluation import context_expansion_evaluator as evaluator
from app.retrieval.context_expansion import ExpandedContext
from app.retrieval.fusion import HybridSearchResult


def make_indexed_chunk(
    *,
    start=0,
    end=10,
    index=0,
    filename="policy.txt",
    document_id=None,
    chunk_id=None,
):
    return base.IndexedChunk(
        chunk_id=chunk_id or uuid4(),
        document_id=document_id or uuid4(),
        filename=filename,
        chunk_index=index,
        content="x" * (end - start),
        normalized_content="x" * (end - start),
        source_start=start,
        source_end=end,
    )


def make_hybrid_result(
    *,
    chunk_id=None,
    document_id=None,
    chunk_index=0,
    section_title="1. Credit Assessment",
):
    return HybridSearchResult(
        chunk_id=chunk_id or uuid4(),
        document_id=document_id or uuid4(),
        chunk_index=chunk_index,
        content="Original passage",
        filename="policy.txt",
        page_start=1,
        page_end=1,
        rrf_score=0.03,
        semantic_score=0.8,
        bm25_score=3.0,
        section_title=section_title,
    )


def make_target(start, end, evidence_id="e001"):
    return base.EvidenceTarget(
        evidence_id=evidence_id,
        filename="policy.txt",
        phrase="relevant evidence",
        spans=(
            base.SourceSpan(
                filename="policy.txt",
                start=start,
                end=end,
            ),
        ),
    )


def test_to_hybrid_result_preserves_existing_hybrid_result():
    result = make_hybrid_result()

    adapted = evaluator._to_hybrid_result(result)

    assert adapted is result


def test_to_hybrid_result_adapts_semantic_or_bm25_result():
    chunk_id = uuid4()
    document_id = uuid4()

    result = type(
        "RetrievalResult",
        (),
        {
            "chunk_id": chunk_id,
            "document_id": document_id,
            "chunk_index": 2,
            "content": "Original passage",
            "filename": "policy.txt",
            "page_start": 1,
            "page_end": 1,
            "score": 0.75,
            "section_title": "2. Repayment",
        },
    )()

    adapted = evaluator._to_hybrid_result(result)

    assert adapted.chunk_id == chunk_id
    assert adapted.document_id == document_id
    assert adapted.chunk_index == 2
    assert adapted.semantic_score == 0.75
    assert adapted.bm25_score == 0.75
    assert adapted.section_title == "2. Repayment"


def test_build_baseline_unit_preserves_original_source_span():
    chunk = make_indexed_chunk(start=10, end=25)

    unit = evaluator._build_baseline_unit(chunk)

    assert unit.chunk_id == chunk.chunk_id
    assert unit.source_spans == (
        base.SourceSpan("policy.txt", 10, 25),
    )
    assert unit.normalized_characters == 15
    assert unit.original_characters == 15
    assert unit.included_chunk_count == 1


def test_build_expanded_unit_merges_adjacent_source_spans():
    first = make_indexed_chunk(
        start=0,
        end=10,
        index=0,
    )
    second = make_indexed_chunk(
        start=10,
        end=20,
        index=1,
        document_id=first.document_id,
    )

    result = make_hybrid_result(
        chunk_id=first.chunk_id,
        document_id=first.document_id,
        chunk_index=0,
    )

    expanded = ExpandedContext(
        retrieval_result=result,
        content="x" * 20,
        chunk_ids=(first.chunk_id, second.chunk_id),
        section_title=result.section_title,
    )

    unit = evaluator._build_expanded_unit(
        expanded=expanded,
        chunk_index={
            first.chunk_id: first,
            second.chunk_id: second,
        },
    )

    assert unit.chunk_id == first.chunk_id
    assert unit.source_spans == (
        base.SourceSpan("policy.txt", 0, 20),
    )
    assert unit.normalized_characters == 20
    assert unit.original_characters == len(
        base.normalize_text(result.content)
    )
    assert unit.included_chunk_count == 2


def test_build_expanded_unit_rejects_unknown_chunk():
    first = make_indexed_chunk(start=0, end=10)

    result = make_hybrid_result(
        chunk_id=first.chunk_id,
        document_id=first.document_id,
    )

    expanded = ExpandedContext(
        retrieval_result=result,
        content="expanded content",
        chunk_ids=(first.chunk_id, uuid4()),
        section_title=result.section_title,
    )

    with pytest.raises(
        evaluator.ContextExpansionEvaluationError,
        match="unknown chunk",
    ):
        evaluator._build_expanded_unit(
            expanded=expanded,
            chunk_index={first.chunk_id: first},
        )


def test_take_prefix_spans_maps_partial_context_to_source():
    spans = (
        base.SourceSpan("policy.txt", 0, 5),
        base.SourceSpan("policy.txt", 10, 15),
    )

    selected = evaluator._take_prefix_spans(
        spans=spans,
        character_count=8,
    )

    assert selected == [
        base.SourceSpan("policy.txt", 0, 5),
        base.SourceSpan("policy.txt", 10, 12),
    ]


def test_pack_context_units_respects_character_budget():
    first = make_indexed_chunk(start=0, end=5)
    second = make_indexed_chunk(
        start=10,
        end=15,
        index=1,
    )

    units = [
        evaluator._build_baseline_unit(first),
        evaluator._build_baseline_unit(second),
    ]

    spans, consumed = evaluator._pack_context_units(
        units=units,
        character_budget=8,
    )

    assert consumed == 8
    assert spans == [
        base.SourceSpan("policy.txt", 0, 5),
        base.SourceSpan("policy.txt", 10, 13),
    ]


def test_context_metrics_accumulate_evidence_across_ranked_units():
    first = evaluator.ContextUnit(
        chunk_id=uuid4(),
        source_spans=(
            base.SourceSpan("policy.txt", 0, 10),
        ),
        normalized_characters=10,
        original_characters=10,
        included_chunk_count=1,
    )

    second = evaluator.ContextUnit(
        chunk_id=uuid4(),
        source_spans=(
            base.SourceSpan("policy.txt", 10, 20),
        ),
        normalized_characters=10,
        original_characters=10,
        included_chunk_count=1,
    )

    result = evaluator.calculate_context_metrics(
        units=[first, second],
        evidence_targets=[make_target(8, 15)],
    )

    assert result["metrics"]["evidence_recall_at_5"] == 1.0
    assert result["metrics"]["evidence_mean_reciprocal_rank"] == 0.5
    assert result["evidence_ranks"][0]["first_covering_rank"] == 2


def test_expanded_context_can_cover_evidence_in_one_ranked_item():
    expanded_unit = evaluator.ContextUnit(
        chunk_id=uuid4(),
        source_spans=(
            base.SourceSpan("policy.txt", 0, 20),
        ),
        normalized_characters=20,
        original_characters=10,
        included_chunk_count=2,
    )

    result = evaluator.calculate_context_metrics(
        units=[expanded_unit],
        evidence_targets=[make_target(12, 18)],
    )

    assert result["metrics"]["evidence_recall_at_5"] == 1.0
    assert result["metrics"]["evidence_mean_reciprocal_rank"] == 1.0
    assert result["evidence_ranks"][0]["first_covering_rank"] == 1


def test_context_budget_only_counts_evidence_inside_included_prefix():
    unit = evaluator.ContextUnit(
        chunk_id=uuid4(),
        source_spans=(
            base.SourceSpan("policy.txt", 0, 2000),
        ),
        normalized_characters=2000,
        original_characters=2000,
        included_chunk_count=1,
    )

    result = evaluator.calculate_context_metrics(
        units=[unit],
        evidence_targets=[make_target(1500, 1510)],
    )

    assert (
        result["context_budgets"]["1000"]["evidence_recall"]
        == 0.0
    )
    assert (
        result["context_budgets"]["2000"]["evidence_recall"]
        == 1.0
    )


def test_context_metrics_reject_empty_evidence_targets():
    with pytest.raises(
        evaluator.ContextExpansionEvaluationError,
        match="must not be empty",
    ):
        evaluator.calculate_context_metrics(
            units=[],
            evidence_targets=[],
        )


def test_pack_context_units_rejects_nonpositive_budget():
    with pytest.raises(ValueError, match="must be positive"):
        evaluator._pack_context_units(
            units=[],
            character_budget=0,
        )