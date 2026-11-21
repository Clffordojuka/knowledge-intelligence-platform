import pytest

from app.evaluation.retrieval_metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_precision_at_k():
    assert precision_at_k(
        ["A", "C", "B", "D"],
        ["B", "C"],
        k=3,
    ) == pytest.approx(2 / 3)


def test_recall_at_k():
    assert recall_at_k(
        ["A", "C", "B", "D"],
        ["B", "C"],
        k=3,
    ) == pytest.approx(1.0)


def test_reciprocal_rank():
    assert reciprocal_rank(
        ["A", "C", "B", "D"],
        ["B", "C"],
    ) == pytest.approx(0.5)


def test_reciprocal_rank_returns_zero_without_relevant_results():
    assert (
        reciprocal_rank(
            ["A", "B", "C"],
            ["D"],
        )
        == 0.0
    )


def test_ndcg_at_k_for_perfect_ranking():
    assert ndcg_at_k(
        ["B", "C", "A"],
        ["B", "C"],
        k=3,
    ) == pytest.approx(1.0)


def test_ndcg_at_k_rewards_higher_relevant_rankings():
    good_ranking = ndcg_at_k(
        ["B", "C", "A"],
        ["B", "C"],
        k=3,
    )

    poor_ranking = ndcg_at_k(
        ["A", "B", "C"],
        ["B", "C"],
        k=3,
    )

    assert good_ranking > poor_ranking


def test_precision_at_k_returns_zero_for_empty_results():
    assert precision_at_k([], ["A"], k=5) == 0.0


def test_recall_at_k_returns_zero_for_empty_results():
    assert recall_at_k([], ["A"], k=5) == 0.0


def test_ndcg_at_k_returns_zero_for_empty_results():
    assert ndcg_at_k([], ["A"], k=5) == 0.0


@pytest.mark.parametrize("k", [0, -1, 1.5, True])
def test_metrics_reject_invalid_k(k):
    with pytest.raises(ValueError, match="positive integer"):
        precision_at_k(["A"], ["A"], k=k)


def test_metrics_reject_empty_relevance_set():
    with pytest.raises(ValueError, match="At least one relevant"):
        recall_at_k(["A"], [], k=5)


def test_metrics_reject_duplicate_retrieved_ids():
    with pytest.raises(ValueError, match="duplicates"):
        precision_at_k(
            ["A", "A"],
            ["A"],
            k=2,
        )


def test_metrics_accept_uuid_identifiers():
    from uuid import uuid4

    relevant_id = uuid4()

    assert precision_at_k(
        [relevant_id],
        [str(relevant_id)],
        k=1,
    ) == pytest.approx(1.0)
