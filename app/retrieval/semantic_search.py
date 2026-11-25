from dataclasses import dataclass
from math import isfinite
from uuid import UUID

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    ScoredPoint,
)

from app.core.config import settings
from app.embeddings.service import (
    EmbeddingService,
    get_embedding_service,
)
from app.vector_store.client import get_qdrant_client


class RetrievalError(Exception):
    """Raised when semantic retrieval fails."""


@dataclass(frozen=True)
class SearchResult:
    """A retrieved document chunk with its provenance and score."""

    chunk_id: UUID
    document_id: UUID
    chunk_index: int
    content: str
    score: float
    filename: str | None
    page_start: int | None
    page_end: int | None


def _build_document_filter(
    document_id: UUID | None = None,
    document_ids: list[UUID] | None = None,
) -> Filter | None:
    """Restrict retrieval to one or more documents."""

    if document_id is not None:
        return Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=str(document_id)),
                )
            ]
        )

    if document_ids is not None:
        return Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchAny(
                        any=[str(item) for item in document_ids]
                    ),
                )
            ]
        )

    return None


def _parse_search_result(point: ScoredPoint) -> SearchResult:
    """Convert a Qdrant search point into our retrieval result."""

    payload = point.payload or {}

    try:
        chunk_id = UUID(str(payload["chunk_id"]))
        document_id = UUID(str(payload["document_id"]))
        chunk_index = payload["chunk_index"]
        content = payload["content"]

        if isinstance(chunk_index, bool) or not isinstance(chunk_index, int):
            raise ValueError("Chunk index must be an integer.")

        if chunk_index < 0:
            raise ValueError("Chunk index cannot be negative.")

        if not isinstance(content, str) or not content.strip():
            raise ValueError("Chunk content must be a non-empty string.")

        score = float(point.score)

        if not isfinite(score):
            raise ValueError("Similarity score must be finite.")

        return SearchResult(
            chunk_id=chunk_id,
            document_id=document_id,
            chunk_index=chunk_index,
            content=content,
            score=score,
            filename=payload.get("filename"),
            page_start=payload.get("page_start"),
            page_end=payload.get("page_end"),
        )

    except (KeyError, TypeError, ValueError) as exc:
        raise RetrievalError(
            f"Invalid Qdrant result payload for point '{point.id}'."
        ) from exc


def search_similar_chunks(
    query: str,
    limit: int = 5,
    document_id: UUID | None = None,
    score_threshold: float | None = None,
    client: QdrantClient | None = None,
    embedding_service: EmbeddingService | None = None,
    document_ids: list[UUID] | None = None,
    query_instruction: str | None = None,
) -> list[SearchResult]:
    """Search Qdrant for semantically similar document chunks."""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Search query must be a non-empty string.")

    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("Search limit must be a positive integer.")

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

    if query_instruction is not None:
        if (
            not isinstance(query_instruction, str)
            or not query_instruction.strip()
        ):
            raise ValueError(
                "Query instruction must be a non-empty string or None."
            )

    if score_threshold is not None:
        if (
            isinstance(score_threshold, bool)
            or not isinstance(score_threshold, (int, float))
            or not isfinite(score_threshold)
            or not -1.0 <= score_threshold <= 1.0
        ):
            raise ValueError(
                "Score threshold must be a finite cosine score "
                "between -1.0 and 1.0."
            )

    try:
        embedder = (
            get_embedding_service()
            if embedding_service is None
            else embedding_service
        )

        normalized_query = query.strip()

        if query_instruction is None:
            # Preserve the original embedding call by default.
            query_vector = embedder.embed_query(normalized_query)
        else:
            query_vector = embedder.embed_query(
                normalized_query,
                instruction=query_instruction,
            )

        if len(query_vector) != settings.embedding_dimension:
            raise RetrievalError(
                "Query embedding dimension does not match "
                "the configured Qdrant vector dimension."
            )

        qdrant_client = (
            get_qdrant_client() if client is None else client
        )

        query_filter = _build_document_filter(
            document_id=document_id,
            document_ids=document_ids,
        )

        response = qdrant_client.query_points(
            collection_name=settings.qdrant_collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
            with_vectors=False,
        )

        return [
            _parse_search_result(point)
            for point in response.points
        ]

    except RetrievalError:
        raise

    except Exception as exc:
        raise RetrievalError("Semantic search failed.") from exc