import logging
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database.models import Document
from app.ingestion.services.storage import StoredFile

logger = logging.getLogger(__name__)


class DuplicateDocumentError(Exception):
    """Raised when a document with identical content already exists."""


def _remove_stored_file(path: Path) -> None:
    """Remove a stored file when its database operation cannot be completed."""

    try:
        path.unlink(missing_ok=True)
    except OSError:
        # Preserve the original database or duplicate error.
        logger.exception("Could not remove uploaded file: %s", path)


def persist_document(
    db: Session,
    original_filename: str,
    stored_file: StoredFile,
) -> Document:
    """Persist document metadata, rejecting duplicate file contents."""

    try:
        existing_id = db.scalar(
            select(Document.id).where(Document.sha256 == stored_file.sha256)
        )

        if existing_id is not None:
            raise DuplicateDocumentError(
                "A document with identical content already exists."
            )

        document = Document(
            filename=original_filename,
            content_type=stored_file.content_type,
            file_size_bytes=stored_file.size_bytes,
            sha256=stored_file.sha256,
            storage_path=str(stored_file.path),
            status="pending",
        )

        db.add(document)
        db.commit()

    except DuplicateDocumentError:
        db.rollback()
        _remove_stored_file(stored_file.path)
        raise

    except IntegrityError as exc:
        # Another upload may have inserted the same hash after our initial
        # duplicate check. The database's unique constraint handles that race.
        db.rollback()

        existing_id = db.scalar(
            select(Document.id).where(Document.sha256 == stored_file.sha256)
        )

        _remove_stored_file(stored_file.path)

        if existing_id is not None:
            raise DuplicateDocumentError(
                "A document with identical content already exists."
            ) from exc

        # This was a different integrity error; preserve the real cause.
        raise

    except Exception:
        db.rollback()
        _remove_stored_file(stored_file.path)
        raise

    db.refresh(document)
    return document
