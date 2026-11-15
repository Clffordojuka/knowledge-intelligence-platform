import logging
from pathlib import Path
from uuid import UUID

from sqlalchemy.orm import Session

from app.database.models import Document
from app.ingestion.extractors.base import (
    DocumentExtractionError,
    DocumentExtractor,
    ExtractionResult,
)
from app.ingestion.extractors.registry import get_extractor

logger = logging.getLogger(__name__)


class DocumentNotFoundError(Exception):
    """Raised when the requested document does not exist."""


def extract_document(
    db: Session,
    document_id: UUID,
    extractor: DocumentExtractor | None = None,
) -> ExtractionResult:
    """Extract a document and update its processing metadata."""

    document = db.get(Document, document_id)

    if document is None:
        raise DocumentNotFoundError(f"Document {document_id} was not found.")

    if document.status == "completed":
        raise DocumentExtractionError("This document has already been processed.")

    try:
        selected_extractor = extractor or get_extractor(document.content_type)

        document.status = "processing"
        document.error_message = None
        db.commit()

        result = selected_extractor.extract(Path(document.storage_path))

        document.page_count = result.page_count
        document.extracted_char_count = sum(len(page.text) for page in result.pages)
        document.status = "completed"
        document.error_message = None

        db.commit()
        db.refresh(document)

        return result

    except Exception as exc:
        db.rollback()

        document = db.get(Document, document_id)

        if document is not None:
            document.status = "failed"
            document.error_message = str(exc)[:1000]

            try:
                db.commit()
            except Exception:
                db.rollback()
                logger.exception(
                    "Could not persist extraction failure for document %s",
                    document_id,
                )

        if isinstance(exc, DocumentExtractionError):
            raise

        logger.exception(
            "Unexpected extraction failure for document %s",
            document_id,
        )

        raise DocumentExtractionError(
            "An unexpected error occurred during document extraction."
        ) from exc
