from dataclasses import dataclass
from uuid import UUID

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.database.models import Chunk, Document
from app.embeddings.service import get_embedding_service
from app.vector_store.client import get_qdrant_client
from app.vector_store.service import ensure_collection


class IndexingError(Exception):
    """Raised when document vector indexing fails."""


class DocumentNotFoundError(IndexingError):
    """Raised when the requested document does not exist."""


@dataclass(frozen=True)
class IndexingResult:
    """Summary of a successfully indexed document."""

    document_id: UUID
    indexed_chunks: int
    collection_name: str


def index_document(
    db: Session,
    document_id: UUID,
    client: QdrantClient | None = None,
) -> IndexingResult:
    """Generate embeddings and index a document's chunks in Qdrant."""

    document = db.get(Document, document_id)

    if document is None:
        raise DocumentNotFoundError(f"Document '{document_id}' was not found.")

    if document.status != "completed":
        raise IndexingError("Only successfully processed documents can be indexed.")

    if document.indexing_status == "processing":
        raise IndexingError("This document is already being indexed.")

    chunks = db.scalars(
        select(Chunk)
        .where(Chunk.document_id == document_id)
        .order_by(Chunk.chunk_index)
    ).all()

    if not chunks:
        raise IndexingError("Cannot index a document without extracted chunks.")

    # Commit the indexing state before performing external operations.
    document.indexing_status = "processing"
    db.commit()

    try:
        embedding_service = get_embedding_service()

        if embedding_service.vector_dimension != settings.embedding_dimension:
            raise IndexingError(
                "Embedding model dimension does not match "
                "the configured Qdrant vector dimension."
            )

        qdrant_client = get_qdrant_client() if client is None else client

        ensure_collection(qdrant_client)

        texts = [chunk.content for chunk in chunks]
        embeddings = embedding_service.embed_texts(texts)

        if len(embeddings) != len(chunks):
            raise IndexingError("Embedding count does not match the number of chunks.")

        points = []

        for chunk, embedding in zip(chunks, embeddings, strict=True):
            if len(embedding) != settings.embedding_dimension:
                raise IndexingError(
                    f"Unexpected embedding dimension for chunk {chunk.id}."
                )

            points.append(
                PointStruct(
                    id=str(chunk.id),
                    vector=embedding,
                    payload={
                        "chunk_id": str(chunk.id),
                        "document_id": str(document.id),
                        "chunk_index": chunk.chunk_index,
                        "filename": document.filename,
                        "content": chunk.content,
                        "page_start": chunk.page_start,
                        "page_end": chunk.page_end,
                    },
                )
            )

        qdrant_client.upsert(
            collection_name=settings.qdrant_collection_name,
            points=points,
            wait=True,
        )

        document.indexing_status = "completed"
        db.commit()

        return IndexingResult(
            document_id=document.id,
            indexed_chunks=len(points),
            collection_name=settings.qdrant_collection_name,
        )

    except Exception as exc:
        db.rollback()

        # Best-effort recording of the indexing failure.
        try:
            failed_document = db.get(Document, document_id)

            if failed_document is not None:
                failed_document.indexing_status = "failed"
                db.commit()

        except Exception:
            db.rollback()

        if isinstance(exc, IndexingError):
            raise

        raise IndexingError(f"Failed to index document '{document_id}'.") from exc
