from dataclasses import replace
from uuid import UUID

import pytest

from app.core.config import settings
from app.retrieval.fusion import HybridSearchResult
from app.retrieval.reranking import (
    RerankingError,
    get_reranker,
    rerank_candidates,
)


class FakeCrossEncoder:
    """A lightweight cross-encoder substitute for unit tests."""

    def __init__(self, scores=None, error=None):
        self.scores = scores or []
        self.error = error
        self.received_pairs = None
        self.received_batch_size = None

    def predict(
        self,
        pairs,
        batch_size,
        show_progress_bar,
    ):
        self.received_pairs = pairs
        self.received_batch_size = batch_size

        if self.error:
            raise self.error

        return self.scores


@pytest.fixture
def candidates():
    document_id = UUID("10000000-0000-0000-0000-000000000001")

    return [
        HybridSearchResult(
            chunk_id=UUID("00000000-0000-0000-0000-000000000001"),
            document_id=document_id,
            chunk_index=0,
            content="Identity verification is required.",
            filename="policy.txt",
            page_start=1,
            page_end=1,
            rrf_score=0.02,
            semantic_score=0.85,
            bm25_score=1.2,
        ),
        HybridSearchResult(
            chunk_id=UUID("00000000-0000-0000-0000-000000000002"),
            document_id=document_id,
            chunk_index=1,
            content="Repayment capacity must be assessed.",
            filename="policy.txt",
            page_start=2,
            page_end=2,
            rrf_score=0.03,
            semantic_score=0.80,
            bm25_score=1.5,
        ),
        HybridSearchResult(
            chunk_id=UUID("00000000-0000-0000-0000-000000000003"),
            document_id=document_id,
            chunk_index=2,
            content="Credit history must be reviewed.",
            filename="policy.txt",
            page_start=3,
            page_end=3,
            rrf_score=0.01,
            semantic_score=0.75,
            bm25_score=1.0,
        ),
    ]


def test_reranks_candidates_by_cross_encoder_score(candidates):
    model = FakeCrossEncoder(scores=[0.2, 0.9, 0.5])

    results = rerank_candidates(
        "How is repayment capacity assessed?",
        candidates,
        reranker=model,
    )

    assert [result.chunk_index for result in results] == [1, 2, 0]


def test_preserves_original_candidate_objects(candidates):
    model = FakeCrossEncoder(scores=[0.2, 0.9, 0.5])

    results = rerank_candidates(
        "How is repayment capacity assessed?",
        candidates,
        reranker=model,
    )

    assert results[0] is candidates[1]
    assert results[1] is candidates[2]
    assert results[2] is candidates[0]


def test_preserves_original_retrieval_scores(candidates):
    model = FakeCrossEncoder(scores=[0.2, 0.9, 0.5])

    results = rerank_candidates(
        "How is repayment capacity assessed?",
        candidates,
        reranker=model,
    )

    reranked_candidate = results[0]

    assert reranked_candidate.rrf_score == 0.03
    assert reranked_candidate.semantic_score == 0.80
    assert reranked_candidate.bm25_score == 1.5


def test_applies_top_k_limit(candidates):
    model = FakeCrossEncoder(scores=[0.2, 0.9, 0.5])

    results = rerank_candidates(
        "How is repayment capacity assessed?",
        candidates,
        top_k=2,
        reranker=model,
    )

    assert len(results) == 2
    assert results[0].chunk_index == 1
    assert results[1].chunk_index == 2


def test_returns_empty_list_without_model_inference():
    model = FakeCrossEncoder()

    results = rerank_candidates(
        "Any query",
        [],
        reranker=model,
    )

    assert results == []
    assert model.received_pairs is None


def test_strips_query_before_building_pairs(candidates):
    model = FakeCrossEncoder(scores=[0.2, 0.9, 0.5])

    rerank_candidates(
        "  repayment capacity  ",
        candidates,
        reranker=model,
    )

    assert model.received_pairs == [
        ("repayment capacity", candidate.content) for candidate in candidates
    ]


def test_uses_configured_batch_size(candidates):
    model = FakeCrossEncoder(scores=[0.2, 0.9, 0.5])

    rerank_candidates(
        "policy requirements",
        candidates,
        reranker=model,
    )

    assert model.received_batch_size == settings.reranker_batch_size


@pytest.mark.parametrize(
    "query",
    [
        "",
        "   ",
        None,
        123,
    ],
)
def test_rejects_invalid_queries(candidates, query):
    model = FakeCrossEncoder(scores=[0.2, 0.9, 0.5])

    with pytest.raises(RerankingError):
        rerank_candidates(
            query,
            candidates,
            reranker=model,
        )


@pytest.mark.parametrize(
    "top_k",
    [
        0,
        -1,
        1.5,
        True,
        "2",
    ],
)
def test_rejects_invalid_top_k(candidates, top_k):
    model = FakeCrossEncoder(scores=[0.2, 0.9, 0.5])

    with pytest.raises(RerankingError):
        rerank_candidates(
            "policy requirements",
            candidates,
            top_k=top_k,
            reranker=model,
        )


def test_rejects_non_list_candidates():
    with pytest.raises(RerankingError):
        rerank_candidates(
            "policy requirements",
            None,
            reranker=FakeCrossEncoder(),
        )


def test_rejects_invalid_candidate_type():
    with pytest.raises(RerankingError):
        rerank_candidates(
            "policy requirements",
            ["not a hybrid result"],
            reranker=FakeCrossEncoder(),
        )


def test_rejects_non_uuid_chunk_id(candidates):
    invalid_candidate = replace(
        candidates[0],
        chunk_id="not-a-uuid",
    )

    with pytest.raises(RerankingError):
        rerank_candidates(
            "policy requirements",
            [invalid_candidate],
            reranker=FakeCrossEncoder(scores=[0.5]),
        )


def test_rejects_duplicate_chunk_ids(candidates):
    duplicate_candidate = replace(
        candidates[1],
        chunk_id=candidates[0].chunk_id,
    )

    with pytest.raises(RerankingError):
        rerank_candidates(
            "policy requirements",
            [candidates[0], duplicate_candidate],
            reranker=FakeCrossEncoder(scores=[0.5, 0.8]),
        )


def test_rejects_empty_candidate_content(candidates):
    invalid_candidate = replace(
        candidates[0],
        content="   ",
    )

    with pytest.raises(RerankingError):
        rerank_candidates(
            "policy requirements",
            [invalid_candidate],
            reranker=FakeCrossEncoder(scores=[0.5]),
        )


def test_rejects_unexpected_number_of_scores(candidates):
    model = FakeCrossEncoder(scores=[0.5, 0.8])

    with pytest.raises(RerankingError):
        rerank_candidates(
            "policy requirements",
            candidates,
            reranker=model,
        )


@pytest.mark.parametrize(
    "scores",
    [
        [float("nan"), 0.8, 0.5],
        [float("inf"), 0.8, 0.5],
        [float("-inf"), 0.8, 0.5],
    ],
)
def test_rejects_non_finite_scores(candidates, scores):
    model = FakeCrossEncoder(scores=scores)

    with pytest.raises(RerankingError):
        rerank_candidates(
            "policy requirements",
            candidates,
            reranker=model,
        )


def test_wraps_model_inference_errors(candidates):
    model = FakeCrossEncoder(error=RuntimeError("Model inference failed."))

    with pytest.raises(
        RerankingError,
        match="Cross-encoder inference failed",
    ):
        rerank_candidates(
            "policy requirements",
            candidates,
            reranker=model,
        )


def test_uses_chunk_id_for_deterministic_tie_breaking(candidates):
    model = FakeCrossEncoder(scores=[0.5, 0.5, 0.5])

    results = rerank_candidates(
        "policy requirements",
        candidates,
        reranker=model,
    )

    expected = sorted(
        candidates,
        key=lambda candidate: candidate.chunk_id,
    )

    assert results == expected


def test_get_reranker_caches_model(monkeypatch):
    created_models = []

    def fake_cross_encoder(model_name, max_length):
        model = object()
        created_models.append((model_name, max_length, model))
        return model

    monkeypatch.setattr(
        "app.retrieval.reranking.CrossEncoder",
        fake_cross_encoder,
    )

    get_reranker.cache_clear()

    try:
        first_model = get_reranker()
        second_model = get_reranker()

        assert first_model is second_model
        assert len(created_models) == 1
        assert created_models[0][0] == settings.reranker_model_name
        assert created_models[0][1] == settings.reranker_max_length

    finally:
        get_reranker.cache_clear()
