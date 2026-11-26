
import logging
from pathlib import Path
from uuid import UUID

from sqlalchemy.orm import Session

from app.database.models import Chunk, Document
from app.ingestion.chunking.chunker import (
    ChunkingConfig,
    chunk_extraction_result,
)
from app.ingestion.extractors.base import (
    DocumentExtractionError,
    DocumentExtractor,
    ExtractionResult,
)
from app.ingestion.extractors.registry import get_extractor

logger = logging.getLogger(__name__)


class DocumentNotFoundError(Exception):
    """Raised when the requested document does not exist."""


class DocumentProcessingError(Exception):
    """Raised when document processing cannot be completed."""


def process_document(
    db: Session,
    document_id: UUID,
    extractor: DocumentExtractor | None = None,
    chunking_config: ChunkingConfig | None = None,
) -> list[Chunk]:
    """Extract, chunk, and persist a document's content."""

    document = db.get(Document, document_id)

    if document is None:
        raise DocumentNotFoundError(f"Document {document_id} was not found.")

    if document.status in {"processing", "completed"}:
        raise DocumentProcessingError(
            f"Cannot process a document with status '{document.status}'."
        )

    try:
        # Mark the document as processing before doing expensive work.
        document.status = "processing"
        document.error_message = None
        db.commit()

        selected_extractor = extractor or get_extractor(document.content_type)

        extraction: ExtractionResult = selected_extractor.extract(
            Path(document.storage_path)
        )

        chunks = chunk_extraction_result(
            extraction,
            config=chunking_config,
        )

        if not chunks:
            raise DocumentProcessingError(
                "Document processing produced no text chunks."
            )

        # Remove old chunks if this is a retry of a failed ingestion.
        db.query(Chunk).filter(Chunk.document_id == document.id).delete(
            synchronize_session=False
        )

        persisted_chunks = [
            Chunk(
                document_id=document.id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                section_title=chunk.section_title,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
            )
            for chunk in chunks
        ]

        db.add_all(persisted_chunks)

        document.page_count = extraction.page_count
        document.extracted_char_count = sum(
            len(page.text) for page in extraction.pages
        )
        document.status = "completed"
        document.error_message = None

        # Flush to catch constraint violations before committing.
        db.flush()
        db.commit()

        logger.info(
            "Document %s processed successfully: %s chunks",
            document.id,
            len(persisted_chunks),
        )

        return persisted_chunks

    except Exception as exc:
        db.rollback()

        # Reload the record because rollback expires ORM state.
        document = db.get(Document, document_id)

        if document is not None:
            document.status = "failed"
            document.error_message = str(exc)[:1000]

            try:
                db.commit()
            except Exception:
                db.rollback()
                logger.exception(
                    "Could not persist processing failure for document %s",
                    document_id,
                )

        if isinstance(
            exc,
            (DocumentExtractionError, DocumentProcessingError),
        ):
            raise

        logger.exception(
            "Unexpected document processing failure for %s",
            document_id,
        )

        raise DocumentProcessingError(
            "An unexpected error occurred during document processing."
        ) from exc