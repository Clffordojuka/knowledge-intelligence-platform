from uuid import uuid4

import pytest

from app.retrieval.bm25_search import BM25SearchResult
from app.retrieval.fusion import (
    FusionError,
    HybridSearchResult,
    reciprocal_rank_fusion,
)
from app.retrieval.semantic_search import SearchResult


def make_semantic_result(
    chunk_id=None,
    document_id=None,
    chunk_index=0,
    content="Sample document content.",
    score=0.90,
    section_title=None,
):
    """Create a semantic retrieval result."""

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
    chunk_id=None,
    document_id=None,
    chunk_index=0,
    content="Sample document content.",
    score=4.5,
    section_title=None,
):
    """Create a BM25 retrieval result."""

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


def test_fusion_combines_matching_chunks():
    document_id = uuid4()
    chunk_id = uuid4()

    semantic = [
        make_semantic_result(
            chunk_id=chunk_id,
            document_id=document_id,
            score=0.91,
        )
    ]

    bm25 = [
        make_bm25_result(
            chunk_id=chunk_id,
            document_id=document_id,
            score=8.2,
        )
    ]

    results = reciprocal_rank_fusion(semantic, bm25)

    assert len(results) == 1

    result = results[0]

    assert isinstance(result, HybridSearchResult)
    assert result.chunk_id == chunk_id
    assert result.document_id == document_id
    assert result.semantic_score == pytest.approx(0.91)
    assert result.bm25_score == pytest.approx(8.2)


def test_fusion_calculates_correct_rrf_score():
    document_id = uuid4()
    chunk_id = uuid4()

    semantic = [
        make_semantic_result(
            chunk_id=chunk_id,
            document_id=document_id,
        )
    ]

    bm25 = [
        make_bm25_result(
            chunk_id=chunk_id,
            document_id=document_id,
        )
    ]

    results = reciprocal_rank_fusion(
        semantic,
        bm25,
        k=60,
    )

    expected_score = (1 / 61) + (1 / 61)

    assert results[0].rrf_score == pytest.approx(expected_score)


def test_fusion_rewards_chunks_ranked_high_by_both_methods():
    shared_chunk_id = uuid4()
    shared_document_id = uuid4()

    semantic = [
        make_semantic_result(
            chunk_id=shared_chunk_id,
            document_id=shared_document_id,
        ),
        make_semantic_result(),
    ]

    bm25 = [
        make_bm25_result(),
        make_bm25_result(
            chunk_id=shared_chunk_id,
            document_id=shared_document_id,
        ),
    ]

    results = reciprocal_rank_fusion(semantic, bm25)

    shared_result = next(
        result for result in results if result.chunk_id == shared_chunk_id
    )

    assert shared_result.rrf_score == pytest.approx((1 / 61) + (1 / 62))

    assert shared_result.semantic_score is not None
    assert shared_result.bm25_score is not None


def test_fusion_preserves_semantic_only_results():
    semantic_result = make_semantic_result()

    results = reciprocal_rank_fusion(
        semantic_results=[semantic_result],
        bm25_results=[],
    )

    assert len(results) == 1
    assert results[0].chunk_id == semantic_result.chunk_id
    assert results[0].semantic_score == semantic_result.score
    assert results[0].bm25_score is None


def test_fusion_preserves_bm25_only_results():
    bm25_result = make_bm25_result()

    results = reciprocal_rank_fusion(
        semantic_results=[],
        bm25_results=[bm25_result],
    )

    assert len(results) == 1
    assert results[0].chunk_id == bm25_result.chunk_id
    assert results[0].semantic_score is None
    assert results[0].bm25_score == bm25_result.score


def test_fusion_returns_empty_list_for_empty_rankings():
    results = reciprocal_rank_fusion([], [])

    assert results == []


def test_fusion_returns_results_in_descending_rrf_order():
    document_id = uuid4()

    first_id = uuid4()
    second_id = uuid4()
    third_id = uuid4()

    semantic = [
        make_semantic_result(
            chunk_id=first_id,
            document_id=document_id,
        ),
        make_semantic_result(
            chunk_id=second_id,
            document_id=document_id,
        ),
        make_semantic_result(
            chunk_id=third_id,
            document_id=document_id,
        ),
    ]

    bm25 = [
        make_bm25_result(
            chunk_id=third_id,
            document_id=document_id,
        ),
        make_bm25_result(
            chunk_id=second_id,
            document_id=document_id,
        ),
        make_bm25_result(
            chunk_id=first_id,
            document_id=document_id,
        ),
    ]

    results = reciprocal_rank_fusion(semantic, bm25)

    scores = [result.rrf_score for result in results]

    assert scores == sorted(scores, reverse=True)


def test_fusion_respects_result_limit():
    semantic = [
        make_semantic_result(),
        make_semantic_result(),
        make_semantic_result(),
    ]

    results = reciprocal_rank_fusion(
        semantic_results=semantic,
        bm25_results=[],
        limit=2,
    )

    assert len(results) == 2


@pytest.mark.parametrize(
    "k",
    [
        0,
        -1,
        1.5,
        True,
        "60",
    ],
)
def test_fusion_rejects_invalid_k(k):
    with pytest.raises(ValueError, match="RRF k"):
        reciprocal_rank_fusion([], [], k=k)


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
def test_fusion_rejects_invalid_limit(limit):
    with pytest.raises(ValueError, match="Result limit"):
        reciprocal_rank_fusion([], [], limit=limit)


def test_fusion_rejects_non_list_semantic_results():
    with pytest.raises(ValueError, match="Semantic results"):
        reciprocal_rank_fusion(None, [])


def test_fusion_rejects_non_list_bm25_results():
    with pytest.raises(ValueError, match="BM25 results"):
        reciprocal_rank_fusion([], None)


def test_fusion_rejects_unsupported_result_type():
    with pytest.raises(FusionError, match="Unsupported"):
        reciprocal_rank_fusion(
            semantic_results=[object()],
            bm25_results=[],
        )


def test_fusion_rejects_duplicate_chunk_ids_in_semantic_ranking():
    result = make_semantic_result()

    with pytest.raises(FusionError, match="duplicate"):
        reciprocal_rank_fusion(
            semantic_results=[result, result],
            bm25_results=[],
        )


def test_fusion_rejects_duplicate_chunk_ids_in_bm25_ranking():
    result = make_bm25_result()

    with pytest.raises(FusionError, match="duplicate"):
        reciprocal_rank_fusion(
            semantic_results=[],
            bm25_results=[result, result],
        )


def test_fusion_rejects_conflicting_metadata():
    chunk_id = uuid4()

    semantic = [
        make_semantic_result(
            chunk_id=chunk_id,
            content="Original content.",
        )
    ]

    bm25 = [
        make_bm25_result(
            chunk_id=chunk_id,
            content="Conflicting content.",
        )
    ]

    with pytest.raises(FusionError, match="Conflicting metadata"):
        reciprocal_rank_fusion(semantic, bm25)


@pytest.mark.parametrize(
    "score",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_fusion_rejects_non_finite_semantic_scores(score):
    semantic = [make_semantic_result(score=score)]

    with pytest.raises(FusionError, match="finite"):
        reciprocal_rank_fusion(semantic, [])


@pytest.mark.parametrize(
    "score",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_fusion_rejects_non_finite_bm25_scores(score):
    bm25 = [make_bm25_result(score=score)]

    with pytest.raises(FusionError, match="finite"):
        reciprocal_rank_fusion([], bm25)


def test_fusion_uses_deterministic_tie_breaking():
    document_id = uuid4()

    first_id = uuid4()
    second_id = uuid4()

    first_semantic = make_semantic_result(
        chunk_id=first_id,
        document_id=document_id,
    )

    second_semantic = make_semantic_result(
        chunk_id=second_id,
        document_id=document_id,
    )

    second_bm25 = make_bm25_result(
        chunk_id=second_id,
        document_id=document_id,
    )

    first_bm25 = make_bm25_result(
        chunk_id=first_id,
        document_id=document_id,
    )

    results = reciprocal_rank_fusion(
        semantic_results=[
            first_semantic,
            second_semantic,
        ],
        bm25_results=[
            second_bm25,
            first_bm25,
        ],
        k=60,
    )

    expected = sorted(
        [first_id, second_id],
        key=str,
    )

    assert [result.chunk_id for result in results] == expected

    assert results[0].rrf_score == pytest.approx(results[1].rrf_score)


def test_fusion_preserves_result_provenance():
    document_id = uuid4()
    chunk_id = uuid4()

    semantic = [
        make_semantic_result(
            chunk_id=chunk_id,
            document_id=document_id,
            chunk_index=3,
            content="Loan repayment requirements.",
        )
    ]

    results = reciprocal_rank_fusion(semantic, [])

    result = results[0]

    assert result.document_id == document_id
    assert result.chunk_id == chunk_id
    assert result.chunk_index == 3
    assert result.content == "Loan repayment requirements."
    assert result.filename == "credit_policy.txt"
    assert result.page_start == 1
    assert result.page_end == 2

def test_fusion_preserves_section_title():
    document_id = uuid4()
    chunk_id = uuid4()
    section_title = "2. Credit Assessment"

    semantic = [
        make_semantic_result(
            chunk_id=chunk_id,
            document_id=document_id,
            section_title=section_title,
        )
    ]

    bm25 = [
        make_bm25_result(
            chunk_id=chunk_id,
            document_id=document_id,
            section_title=section_title,
        )
    ]

    results = reciprocal_rank_fusion(semantic, bm25)

    assert len(results) == 1
    assert results[0].section_title == section_title


def test_fusion_preserves_section_title_from_semantic_only_result():
    result = make_semantic_result(
        section_title="3. Repayment Capacity"
    )

    fused = reciprocal_rank_fusion([result], [])

    assert fused[0].section_title == "3. Repayment Capacity"


def test_fusion_preserves_section_title_from_bm25_only_result():
    result = make_bm25_result(
        section_title="4. Approval Process"
    )

    fused = reciprocal_rank_fusion([], [result])

    assert fused[0].section_title == "4. Approval Process"


def test_fusion_rejects_conflicting_section_titles():
    chunk_id = uuid4()
    document_id = uuid4()

    semantic = [
        make_semantic_result(
            chunk_id=chunk_id,
            document_id=document_id,
            section_title="2. Credit Assessment",
        )
    ]

    bm25 = [
        make_bm25_result(
            chunk_id=chunk_id,
            document_id=document_id,
            section_title="3. Repayment Capacity",
        )
    ]

    with pytest.raises(FusionError, match="Conflicting section titles"):
        reciprocal_rank_fusion(semantic, bm25)