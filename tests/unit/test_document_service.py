import hashlib
from pathlib import Path

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.database.base import Base
from app.database.models import Document
from app.ingestion.services.document_service import (
    DuplicateDocumentError,
    persist_document,
)
from app.ingestion.services.storage import StoredFile


@pytest.fixture
def db_session():
    """Provide an isolated in-memory database for each test."""

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    Base.metadata.create_all(engine)

    with Session(engine) as session:
        yield session

    Base.metadata.drop_all(engine)
    engine.dispose()


def create_stored_file(
    tmp_path: Path,
    filename: str,
    content: bytes,
) -> StoredFile:
    """Create a test file and its corresponding storage metadata."""

    path = tmp_path / filename
    path.write_bytes(content)

    return StoredFile(
        path=path,
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
        content_type="text/plain",
    )


def test_persist_new_document(db_session, tmp_path):
    content = b"Internal credit policy document."

    stored_file = create_stored_file(
        tmp_path,
        "stored-document.txt",
        content,
    )

    document = persist_document(
        db_session,
        original_filename="credit_policy.txt",
        stored_file=stored_file,
    )

    assert document.id is not None
    assert document.filename == "credit_policy.txt"
    assert document.content_type == "text/plain"
    assert document.file_size_bytes == len(content)
    assert document.sha256 == stored_file.sha256
    assert document.storage_path == str(stored_file.path)
    assert document.status == "pending"
    assert stored_file.path.exists()


def test_reject_duplicate_document(db_session, tmp_path):
    content = b"Identical document content."

    first_file = create_stored_file(
        tmp_path,
        "first-upload.txt",
        content,
    )

    second_file = create_stored_file(
        tmp_path,
        "second-upload.txt",
        content,
    )

    first_document = persist_document(
        db_session,
        original_filename="first.txt",
        stored_file=first_file,
    )

    with pytest.raises(DuplicateDocumentError):
        persist_document(
            db_session,
            original_filename="second.txt",
            stored_file=second_file,
        )

    document_count = db_session.scalar(select(func.count()).select_from(Document))

    assert document_count == 1
    assert first_file.path.exists()
    assert not second_file.path.exists()

    saved_document = db_session.scalar(
        select(Document).where(Document.id == first_document.id)
    )

    assert saved_document is not None
    assert saved_document.filename == "first.txt"
