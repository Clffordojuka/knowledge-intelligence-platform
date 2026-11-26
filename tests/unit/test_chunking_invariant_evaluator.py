
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.evaluation import chunking_invariant_evaluator as evaluator


def make_chunk(content: str, index: int):
    return SimpleNamespace(
        id=uuid4(),
        document_id=uuid4(),
        chunk_index=index,
        content=content,
    )


def test_normalize_text_collapses_case_and_whitespace():
    assert evaluator.normalize_text(
        "  CREDIT   POLICY\n APPROVAL "
    ) == "credit policy approval"


def test_find_all_occurrences_returns_all_positions():
    occurrences = evaluator._find_all_occurrences(
        "credit approval and credit review",
        "credit",
    )

    assert occurrences == [(0, 6), (20, 26)]


def test_build_chunk_index_maps_overlapping_chunks():
    source = "policy section one section two"

    first = make_chunk("policy section one", 0)
    second = make_chunk("section one section two", 1)

    index = evaluator.build_chunk_index(
        filename="policy.txt",
        source_text=source,
        chunks=[first, second],
    )

    normalized_source = evaluator.normalize_text(source)
    expected_second_start = normalized_source.find("section one")

    assert index[first.id].source_start == 0
    assert index[second.id].source_start == expected_second_start
    assert index[first.id].source_end > index[second.id].source_start


def test_build_chunk_index_rejects_content_not_in_source():
    chunk = make_chunk("unrelated content", 0)

    with pytest.raises(
        evaluator.ChunkingInvariantEvaluationError,
        match="map a persisted chunk",
    ):
        evaluator.build_chunk_index(
            filename="policy.txt",
            source_text="the original policy",
            chunks=[chunk],
        )


def test_build_evidence_targets_maps_phrase_to_source_span():
    source = "Customer identity must be verified before approval."

    targets = evaluator.build_evidence_targets(
        query_item={
            "id": "q001",
            "relevant_evidence": [
                {
                    "filename": "policy.txt",
                    "phrase": "identity must be verified",
                }
            ],
        },
        source_texts={"policy.txt": source},
    )

    assert len(targets) == 1
    assert targets[0].evidence_id == "e001"
    assert targets[0].filename == "policy.txt"

    normalized_source = evaluator.normalize_text(source)
    start = normalized_source.find("identity must be verified")

    assert targets[0].spans[0].start == start
    assert targets[0].spans[0].end == (
        start + len("identity must be verified")
    )


def test_build_evidence_targets_rejects_missing_phrase():
    with pytest.raises(
        evaluator.ChunkingInvariantEvaluationError,
        match="does not occur",
    ):
        evaluator.build_evidence_targets(
            query_item={
                "id": "q001",
                "relevant_evidence": [
                    {
                        "filename": "policy.txt",
                        "phrase": "missing evidence",
                    }
                ],
            },
            source_texts={"policy.txt": "original policy text"},
        )


def test_merge_source_spans_merges_overlapping_and_adjacent_ranges():
    merged = evaluator.merge_source_spans(
        [
            evaluator.SourceSpan("policy.txt", 0, 10),
            evaluator.SourceSpan("policy.txt", 8, 15),
            evaluator.SourceSpan("policy.txt", 15, 20),
            evaluator.SourceSpan("other.txt", 0, 5),
        ]
    )

    assert merged == [
        evaluator.SourceSpan("policy.txt", 0, 20),
        evaluator.SourceSpan("other.txt", 0, 5),
    ]


def test_evidence_can_be_covered_by_multiple_adjacent_chunks():
    source = "customer identity must be verified before approval"

    first = make_chunk("customer identity must", 0)
    second = make_chunk("must be verified before approval", 1)

    index = evaluator.build_chunk_index(
        filename="policy.txt",
        source_text=source,
        chunks=[first, second],
    )

    targets = evaluator.build_evidence_targets(
        query_item={
            "id": "q001",
            "relevant_evidence": [
                {
                    "filename": "policy.txt",
                    "phrase": "identity must be verified",
                }
            ],
        },
        source_texts={"policy.txt": source},
    )

    result = evaluator.calculate_evidence_metrics(
        ranked_chunks=[
            index[first.id],
            index[second.id],
        ],
        evidence_targets=targets,
    )

    assert result["metrics"]["evidence_recall_at_5"] == 1.0
    assert result["metrics"]["evidence_recall_at_10"] == 1.0
    assert result["metrics"]["evidence_mean_reciprocal_rank"] == 0.5
    assert result["evidence_ranks"][0]["first_covering_rank"] == 2


def test_partial_phrase_coverage_does_not_count_as_retrieved():
    source = "customer identity must be verified before approval"

    first = make_chunk("customer identity must", 0)
    second = make_chunk("must be verified before approval", 1)

    index = evaluator.build_chunk_index(
        filename="policy.txt",
        source_text=source,
        chunks=[first, second],
    )

    targets = evaluator.build_evidence_targets(
        query_item={
            "id": "q001",
            "relevant_evidence": [
                {
                    "filename": "policy.txt",
                    "phrase": "identity must be verified",
                }
            ],
        },
        source_texts={"policy.txt": source},
    )

    result = evaluator.calculate_evidence_metrics(
        ranked_chunks=[index[first.id]],
        evidence_targets=targets,
    )

    assert result["metrics"]["evidence_recall_at_5"] == 0.0
    assert result["metrics"]["evidence_mean_reciprocal_rank"] == 0.0
    assert result["evidence_ranks"][0]["first_covering_rank"] is None


def test_context_budget_counts_only_the_included_chunk_prefix():
    chunk = make_chunk("customer identity must be verified", 0)

    index = evaluator.build_chunk_index(
        filename="policy.txt",
        source_text="customer identity must be verified",
        chunks=[chunk],
    )

    indexed_chunk = index[chunk.id]

    spans, consumed = evaluator._context_spans(
        ranked_chunks=[indexed_chunk],
        character_budget=10,
    )

    assert consumed == 10
    assert spans == [
        evaluator.SourceSpan(
            filename="policy.txt",
            start=0,
            end=10,
        )
    ]


def test_context_budget_counts_overlapping_chunk_text_again():
    first = make_chunk("policy section one", 0)
    second = make_chunk("section one section two", 1)

    index = evaluator.build_chunk_index(
        filename="policy.txt",
        source_text="policy section one section two",
        chunks=[first, second],
    )

    spans, consumed = evaluator._context_spans(
        ranked_chunks=[
            index[first.id],
            index[second.id],
        ],
        character_budget=100,
    )

    expected_characters = (
        len(evaluator.normalize_text(first.content))
        + len(evaluator.normalize_text(second.content))
    )

    assert consumed == expected_characters
    assert len(spans) == 2


def test_evidence_metrics_reject_empty_targets():
    with pytest.raises(
        evaluator.ChunkingInvariantEvaluationError,
        match="must not be empty",
    ):
        evaluator.calculate_evidence_metrics(
            ranked_chunks=[],
            evidence_targets=[],
        )