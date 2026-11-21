from functools import lru_cache
from math import isfinite
from uuid import UUID

from sentence_transformers import CrossEncoder

from app.core.config import settings
from app.retrieval.fusion import HybridSearchResult


class RerankingError(Exception):
    """Raised when cross-encoder reranking fails."""


def _validate_inputs(
    query: str,
    candidates: list[HybridSearchResult],
) -> None:
    """Validate the query and candidate collection."""

    if not isinstance(query, str) or not query.strip():
        raise RerankingError("Query must be a non-empty string.")

    if not isinstance(candidates, list):
        raise RerankingError("Candidates must be provided as a list.")

    seen_ids = set()

    for candidate in candidates:
        if not isinstance(candidate, HybridSearchResult):
            raise RerankingError("Every candidate must be a HybridSearchResult.")

        if not isinstance(candidate.chunk_id, UUID):
            raise RerankingError("Candidate chunk IDs must be UUID values.")

        if candidate.chunk_id in seen_ids:
            raise RerankingError("Duplicate candidate chunk IDs are not allowed.")

        seen_ids.add(candidate.chunk_id)

        if not isinstance(candidate.content, str) or not candidate.content.strip():
            raise RerankingError("Candidate content must be a non-empty string.")


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    """Load and cache the configured cross-encoder model."""

    try:
        return CrossEncoder(
            settings.reranker_model_name,
            max_length=settings.reranker_max_length,
        )
    except Exception as exc:
        raise RerankingError("Failed to load the cross-encoder model.") from exc


def rerank_candidates(
    query: str,
    candidates: list[HybridSearchResult],
    *,
    top_k: int | None = None,
    reranker: CrossEncoder | None = None,
) -> list[HybridSearchResult]:
    """
    Rerank retrieved candidates using a cross-encoder.

    The original retrieval scores are preserved. Cross-encoder
    scores are used only to determine the final ordering.
    """

    _validate_inputs(query, candidates)

    if top_k is not None:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise RerankingError("top_k must be a positive integer.")

    if not candidates:
        return []

    model = reranker if reranker is not None else get_reranker()

    pairs = [(query.strip(), candidate.content) for candidate in candidates]

    try:
        scores = model.predict(
            pairs,
            batch_size=settings.reranker_batch_size,
            show_progress_bar=False,
        )
    except Exception as exc:
        raise RerankingError("Cross-encoder inference failed.") from exc

    if len(scores) != len(candidates):
        raise RerankingError("Cross-encoder returned an unexpected number of scores.")

    scored_candidates = []

    for candidate, score in zip(candidates, scores):
        relevance_score = float(score)

        if not isfinite(relevance_score):
            raise RerankingError("Cross-encoder returned a non-finite relevance score.")

        scored_candidates.append((candidate, relevance_score))

    scored_candidates.sort(
        key=lambda item: (
            -item[1],
            item[0].chunk_id,
        )
    )

    reranked = [candidate for candidate, _ in scored_candidates]

    if top_k is not None:
        reranked = reranked[:top_k]

    return reranked
