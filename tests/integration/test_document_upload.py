from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.database.base import Base
from app.database.dependencies import get_db
from app.database.models import Document
from app.main import app


@pytest.fixture
def db_session():
    """Create an isolated in-memory database for API integration tests."""

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


@pytest.fixture
def client(db_session, tmp_path, monkeypatch):
    """Provide a test client with isolated database and file storage."""

    monkeypatch.setattr(settings, "upload_storage_dir", tmp_path)
    monkeypatch.setattr(settings, "max_upload_size_bytes", 1024)

    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.pop(get_db, None)


def test_upload_valid_txt_document(client, db_session, tmp_path):
    content = b"Internal credit policy document."

    response = client.post(
        "/documents",
        files={
            "file": (
                "credit_policy.txt",
                content,
                "text/plain",
            )
        },
    )

    assert response.status_code == 201

    data = response.json()

    assert data["filename"] == "credit_policy.txt"
    assert data["content_type"] == "text/plain"
    assert data["file_size_bytes"] == len(content)
    assert data["status"] == "pending"
    assert "id" in data
    assert "created_at" in data

    document = db_session.scalar(
        select(Document).where(Document.filename == "credit_policy.txt")
    )

    assert document is not None
    assert Path(document.storage_path).exists()
    assert Path(document.storage_path).read_bytes() == content


def test_upload_duplicate_document_returns_conflict(client, tmp_path):
    content = b"Same internal document content."

    first_response = client.post(
        "/documents",
        files={"file": ("first.txt", content, "text/plain")},
    )

    second_response = client.post(
        "/documents",
        files={"file": ("second.txt", content, "text/plain")},
    )

    assert first_response.status_code == 201
    assert second_response.status_code == 409

    stored_files = list(tmp_path.glob("*.txt"))

    assert len(stored_files) == 1
    assert stored_files[0].read_bytes() == content


def test_upload_invalid_file_returns_bad_request(client, tmp_path):
    response = client.post(
        "/documents",
        files={
            "file": (
                "malicious.exe",
                b"Some executable content",
                "application/octet-stream",
            )
        },
    )

    assert response.status_code == 400
    assert list(tmp_path.iterdir()) == []


def test_upload_oversized_file_returns_payload_too_large(
    client,
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(settings, "max_upload_size_bytes", 5)

    response = client.post(
        "/documents",
        files={
            "file": (
                "large.txt",
                b"1234567890",
                "text/plain",
            )
        },
    )

    assert response.status_code == 413
    assert list(tmp_path.iterdir()) == []
