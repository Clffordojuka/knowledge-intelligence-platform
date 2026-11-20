from uuid import UUID

from qdrant_client import QdrantClient

from app.embeddings.service import EmbeddingService
from app.retrieval.bm25_search import (
    BM25RetrievalError,
    search_bm25,
)
from app.retrieval.fusion import (
    FusionError,
    HybridSearchResult,
    reciprocal_rank_fusion,
)
from app.retrieval.semantic_search import (
    RetrievalError,
    search_similar_chunks,
)


class HybridRetrievalError(Exception):
    """Raised when hybrid retrieval cannot be completed."""


def search_hybrid_chunks(
    db,
    query: str,
    limit: int = 10,
    document_id: UUID | None = None,
    candidate_limit: int | None = None,
    client: QdrantClient | None = None,
    embedding_service: EmbeddingService | None = None,
) -> list[HybridSearchResult]:
    """Retrieve and fuse semantic and keyword search results."""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Search query must be a non-empty string.")

    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("Result limit must be a positive integer.")

    if candidate_limit is None:
        candidate_limit = max(limit * 2, 10)

    if (
        isinstance(candidate_limit, bool)
        or not isinstance(candidate_limit, int)
        or candidate_limit <= 0
    ):
        raise ValueError("Candidate limit must be a positive integer.")

    if candidate_limit < limit:
        raise ValueError("Candidate limit cannot be smaller than result limit.")

    try:
        semantic_results = search_similar_chunks(
            query=query,
            limit=candidate_limit,
            document_id=document_id,
            client=client,
            embedding_service=embedding_service,
        )

        bm25_results = search_bm25(
            db=db,
            query=query,
            limit=candidate_limit,
            document_id=document_id,
        )

        return reciprocal_rank_fusion(
            semantic_results=semantic_results,
            bm25_results=bm25_results,
            limit=limit,
        )

    except (
        RetrievalError,
        BM25RetrievalError,
        FusionError,
    ) as exc:
        raise HybridRetrievalError("Hybrid retrieval could not be completed.") from exc

    except Exception as exc:
        raise HybridRetrievalError(
            "An unexpected error occurred during hybrid retrieval."
        ) from exc
