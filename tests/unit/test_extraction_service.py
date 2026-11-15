from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.database.base import Base
from app.database.models import Document
from app.ingestion.extractors.base import (
    DocumentExtractionError,
    ExtractedPage,
    ExtractionResult,
)
from app.ingestion.services.extraction_service import (
    DocumentNotFoundError,
    extract_document,
)


@pytest.fixture
def db_session():
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


def create_document(db_session, tmp_path):
    file_path = tmp_path / "policy.txt"
    file_path.write_text(
        "Credit policy introduction.",
        encoding="utf-8",
    )

    document = Document(
        filename="policy.txt",
        content_type="text/plain",
        file_size_bytes=file_path.stat().st_size,
        sha256="a" * 64,
        storage_path=str(file_path),
        status="pending",
    )

    db_session.add(document)
    db_session.commit()
    db_session.refresh(document)

    return document


class FakeExtractor:
    def extract(self, file_path: Path) -> ExtractionResult:
        return ExtractionResult(
            pages=(
                ExtractedPage(
                    page_number=1,
                    text="Credit policy introduction.",
                ),
                ExtractedPage(
                    page_number=2,
                    text="Eligibility requirements.",
                ),
            ),
            source_format="txt",
        )


class FailingExtractor:
    def extract(self, file_path: Path) -> ExtractionResult:
        raise DocumentExtractionError("Unable to extract document text.")


def test_successful_extraction_updates_document_metadata(
    db_session,
    tmp_path,
):
    document = create_document(db_session, tmp_path)

    result = extract_document(
        db_session,
        document.id,
        extractor=FakeExtractor(),
    )

    db_session.refresh(document)

    assert result.page_count == 2
    assert result.source_format == "txt"
    assert document.status == "completed"
    assert document.page_count == 2
    assert document.extracted_char_count == (
        len("Credit policy introduction.") + len("Eligibility requirements.")
    )
    assert document.error_message is None


def test_failed_extraction_updates_document_status(
    db_session,
    tmp_path,
):
    document = create_document(db_session, tmp_path)

    with pytest.raises(
        DocumentExtractionError,
        match="Unable to extract",
    ):
        extract_document(
            db_session,
            document.id,
            extractor=FailingExtractor(),
        )

    db_session.refresh(document)

    assert document.status == "failed"
    assert document.error_message == "Unable to extract document text."


def test_extraction_rejects_missing_document(db_session):
    from uuid import uuid4

    with pytest.raises(DocumentNotFoundError):
        extract_document(db_session, uuid4())


def test_completed_document_cannot_be_extracted_again(
    db_session,
    tmp_path,
):
    document = create_document(db_session, tmp_path)

    document.status = "completed"
    db_session.commit()

    with pytest.raises(
        DocumentExtractionError,
        match="already been processed",
    ):
        extract_document(
            db_session,
            document.id,
            extractor=FakeExtractor(),
        )
