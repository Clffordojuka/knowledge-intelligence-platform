from collections.abc import Sequence
from math import log2
from uuid import UUID

ChunkID = str | UUID


def _validate_k(k: int) -> None:
    """Validate a positive integer cutoff."""

    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("K must be a positive integer.")


def _normalize_ranked_ids(
    retrieved_ids: Sequence[ChunkID],
) -> list[str]:
    """Normalize retrieved chunk identifiers and reject duplicates."""

    if isinstance(retrieved_ids, (str, bytes)):
        raise ValueError("Retrieved IDs must be a sequence of identifiers.")

    normalized = [str(chunk_id) for chunk_id in retrieved_ids]

    if any(not chunk_id.strip() for chunk_id in normalized):
        raise ValueError("Retrieved IDs must not be empty.")

    if len(normalized) != len(set(normalized)):
        raise ValueError("Retrieved IDs must not contain duplicates.")

    return normalized


def _normalize_relevant_ids(
    relevant_ids: Sequence[ChunkID],
) -> set[str]:
    """Normalize relevant chunk identifiers."""

    if isinstance(relevant_ids, (str, bytes)):
        raise ValueError("Relevant IDs must be a sequence of identifiers.")

    normalized = [str(chunk_id) for chunk_id in relevant_ids]

    if any(not chunk_id.strip() for chunk_id in normalized):
        raise ValueError("Relevant IDs must not be empty.")

    relevant = set(normalized)

    if not relevant:
        raise ValueError("At least one relevant chunk ID is required.")

    return relevant


def precision_at_k(
    retrieved_ids: Sequence[ChunkID],
    relevant_ids: Sequence[ChunkID],
    k: int = 5,
) -> float:
    """Calculate Precision@K."""

    _validate_k(k)

    retrieved = _normalize_ranked_ids(retrieved_ids)
    relevant = _normalize_relevant_ids(relevant_ids)

    top_k = retrieved[:k]
    relevant_count = sum(chunk_id in relevant for chunk_id in top_k)

    return relevant_count / k


def recall_at_k(
    retrieved_ids: Sequence[ChunkID],
    relevant_ids: Sequence[ChunkID],
    k: int = 5,
) -> float:
    """Calculate Recall@K."""

    _validate_k(k)

    retrieved = _normalize_ranked_ids(retrieved_ids)
    relevant = _normalize_relevant_ids(relevant_ids)

    top_k = retrieved[:k]
    relevant_count = sum(chunk_id in relevant for chunk_id in top_k)

    return relevant_count / len(relevant)


def reciprocal_rank(
    retrieved_ids: Sequence[ChunkID],
    relevant_ids: Sequence[ChunkID],
) -> float:
    """Calculate the reciprocal rank of the first relevant result."""

    retrieved = _normalize_ranked_ids(retrieved_ids)
    relevant = _normalize_relevant_ids(relevant_ids)

    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(
    retrieved_ids: Sequence[ChunkID],
    relevant_ids: Sequence[ChunkID],
    k: int = 5,
) -> float:
    """Calculate binary-relevance Normalized Discounted Cumulative Gain."""

    _validate_k(k)

    retrieved = _normalize_ranked_ids(retrieved_ids)
    relevant = _normalize_relevant_ids(relevant_ids)

    top_k = retrieved[:k]

    dcg = sum(
        1.0 / log2(rank + 1)
        for rank, chunk_id in enumerate(top_k, start=1)
        if chunk_id in relevant
    )

    ideal_relevant_count = min(len(relevant), k)

    idcg = sum(1.0 / log2(rank + 1) for rank in range(1, ideal_relevant_count + 1))

    return dcg / idcg if idcg else 0.0
