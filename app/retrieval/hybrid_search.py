from __future__ import annotations

from typing import TYPE_CHECKING
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
from app.retrieval.reranking import (
    RerankingError,
    rerank_candidates,
)
from app.retrieval.semantic_search import (
    RetrievalError,
    search_similar_chunks,
)

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder


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
    rerank: bool = False,
    reranker: CrossEncoder | None = None,
    document_ids: list[UUID] | None = None,
) -> list[HybridSearchResult]:
    """Retrieve, fuse, and optionally rerank semantic and keyword results."""

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
        raise ValueError(
            "Candidate limit cannot be smaller than result limit."
        )

    if not isinstance(rerank, bool):
        raise ValueError("Rerank must be a boolean.")

    if document_id is not None and not isinstance(document_id, UUID):
        raise ValueError("Document ID must be a UUID.")

    if document_id is not None and document_ids is not None:
        raise ValueError(
            "Specify either document_id or document_ids, not both."
        )

    if document_ids is not None:
        if not isinstance(document_ids, list) or not document_ids:
            raise ValueError("Document IDs must be a non-empty list.")

        if any(not isinstance(item, UUID) for item in document_ids):
            raise ValueError("Every document ID must be a UUID.")

        if len(document_ids) != len(set(document_ids)):
            raise ValueError("Document IDs must not contain duplicates.")

    try:
        semantic_results = search_similar_chunks(
            query=query,
            limit=candidate_limit,
            document_id=document_id,
            client=client,
            embedding_service=embedding_service,
            document_ids=document_ids,
        )

        bm25_results = search_bm25(
            db=db,
            query=query,
            limit=candidate_limit,
            document_id=document_id,
            document_ids=document_ids,
        )

        fused_results = reciprocal_rank_fusion(
            semantic_results=semantic_results,
            bm25_results=bm25_results,
            limit=candidate_limit if rerank else limit,
        )

        if rerank:
            return rerank_candidates(
                query=query,
                candidates=fused_results,
                top_k=limit,
                reranker=reranker,
            )

        return fused_results

    except (
        RetrievalError,
        BM25RetrievalError,
        FusionError,
        RerankingError,
    ) as exc:
        raise HybridRetrievalError(
            "Hybrid retrieval could not be completed."
        ) from exc

    except Exception as exc:
        raise HybridRetrievalError(
            "An unexpected error occurred during hybrid retrieval."
        ) from exc