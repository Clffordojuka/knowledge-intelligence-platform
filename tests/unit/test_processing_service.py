from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.database.base import Base
from app.database.models import Chunk, Document
from app.ingestion.chunking.chunker import ChunkingConfig
from app.ingestion.extractors.base import (
    DocumentExtractionError,
    ExtractedPage,
    ExtractionResult,
)
from app.ingestion.services.processing_service import (
    DocumentNotFoundError,
    DocumentProcessingError,
    process_document,
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
        sha256="b" * 64,
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


def test_successful_processing_persists_chunks_and_metadata(
    db_session,
    tmp_path,
):
    document = create_document(db_session, tmp_path)

    chunks = process_document(
        db_session,
        document.id,
        extractor=FakeExtractor(),
        chunking_config=ChunkingConfig(
            chunk_size=200,
            chunk_overlap=20,
        ),
    )

    db_session.refresh(document)

    stored_chunks = (
        db_session.query(Chunk)
        .filter(Chunk.document_id == document.id)
        .order_by(Chunk.chunk_index)
        .all()
    )

    assert len(chunks) == 1
    assert len(stored_chunks) == 1

    assert stored_chunks[0].content == (
        "Credit policy introduction.\n\nEligibility requirements."
    )

    assert stored_chunks[0].chunk_index == 0
    assert stored_chunks[0].page_start == 1
    assert stored_chunks[0].page_end == 2

    assert document.status == "completed"
    assert document.page_count == 2
    assert document.extracted_char_count == (
        len("Credit policy introduction.") + len("Eligibility requirements.")
    )
    assert document.error_message is None


def test_failed_processing_updates_document_status(
    db_session,
    tmp_path,
):
    document = create_document(db_session, tmp_path)

    with pytest.raises(
        DocumentExtractionError,
        match="Unable to extract",
    ):
        process_document(
            db_session,
            document.id,
            extractor=FailingExtractor(),
        )

    db_session.refresh(document)

    assert document.status == "failed"
    assert document.error_message == "Unable to extract document text."

    assert db_session.query(Chunk).filter(Chunk.document_id == document.id).count() == 0


def test_missing_document_raises_error(db_session):
    with pytest.raises(DocumentNotFoundError):
        process_document(db_session, uuid4())


def test_completed_document_cannot_be_processed_again(
    db_session,
    tmp_path,
):
    document = create_document(db_session, tmp_path)

    document.status = "completed"
    db_session.commit()

    with pytest.raises(
        DocumentProcessingError,
        match="completed",
    ):
        process_document(
            db_session,
            document.id,
            extractor=FakeExtractor(),
        )


def test_failed_document_can_be_reprocessed(
    db_session,
    tmp_path,
):
    document = create_document(db_session, tmp_path)

    with pytest.raises(DocumentExtractionError):
        process_document(
            db_session,
            document.id,
            extractor=FailingExtractor(),
        )

    db_session.refresh(document)
    assert document.status == "failed"

    chunks = process_document(
        db_session,
        document.id,
        extractor=FakeExtractor(),
        chunking_config=ChunkingConfig(
            chunk_size=200,
            chunk_overlap=20,
        ),
    )

    db_session.refresh(document)

    assert document.status == "completed"
    assert document.error_message is None
    assert len(chunks) == 1

    assert db_session.query(Chunk).filter(Chunk.document_id == document.id).count() == 1
