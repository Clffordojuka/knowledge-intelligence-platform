from uuid import UUID, uuid4

from fastapi.testclient import TestClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from sqlalchemy import func, select

from app.core.config import settings
from app.database.models import Chunk, Document
from app.database.session import SessionLocal
from app.main import app
from app.vector_store.client import get_qdrant_client


def build_document_filter(document_id: UUID) -> Filter:
    """Build a Qdrant filter for one document's vectors."""

    return Filter(
        must=[
            FieldCondition(
                key="document_id",
                match=MatchValue(value=str(document_id)),
            )
        ]
    )


def count_document_vectors(document_id: UUID) -> int:
    """Return the exact number of Qdrant points for a document."""

    result = get_qdrant_client().count(
        collection_name=settings.qdrant_collection_name,
        count_filter=build_document_filter(document_id),
        exact=True,
    )

    return result.count


def get_document_vectors(document_id: UUID, expected_count: int):
    """Retrieve the indexed points for one document."""

    points, _ = get_qdrant_client().scroll(
        collection_name=settings.qdrant_collection_name,
        scroll_filter=build_document_filter(document_id),
        limit=expected_count + 1,
        with_payload=True,
        with_vectors=False,
    )

    return points


def build_test_content(run_id: str) -> bytes:
    """Generate unique, multi-chunk synthetic document content."""

    sections = []

    for section_number in range(1, 5):
        paragraph = (
            f"Verification run {run_id}, section {section_number}. "
            "This synthetic policy paragraph is used to verify "
            "document processing, chunk persistence, embedding "
            "generation, and vector indexing in the Knowledge "
            "Intelligence Platform. "
        )

        sections.append(paragraph * 12)

    content = (
        f"Knowledge Intelligence Platform Indexing Verification\n"
        f"Run ID: {run_id}\n\n" + "\n\n".join(sections)
    )

    return content.encode("utf-8")


def verify_database_records(document_id: UUID):
    """Verify the document and its chunks in PostgreSQL."""

    with SessionLocal() as db:
        document = db.get(Document, document_id)

        assert document is not None, "Document missing from PostgreSQL."
        assert document.status == "completed", (
            f"Unexpected processing status: {document.status}"
        )
        assert document.indexing_status == "completed", (
            f"Unexpected indexing status: {document.indexing_status}"
        )

        chunk_ids = db.scalars(
            select(Chunk.id)
            .where(Chunk.document_id == document_id)
            .order_by(Chunk.chunk_index)
        ).all()

        chunk_count = db.scalar(
            select(func.count())
            .select_from(Chunk)
            .where(Chunk.document_id == document_id)
        )

        assert chunk_count is not None and chunk_count > 0, (
            "No chunks were persisted in PostgreSQL."
        )

        return [str(chunk_id) for chunk_id in chunk_ids], chunk_count


def verify_vector_payloads(document_id: UUID, chunk_ids: list[str]) -> None:
    """Verify Qdrant point IDs and their document/chunk payloads."""

    points = get_document_vectors(document_id, len(chunk_ids))

    assert len(points) == len(chunk_ids), (
        f"Expected {len(chunk_ids)} Qdrant points, but retrieved {len(points)}."
    )

    retrieved_ids = {str(point.id) for point in points}

    assert retrieved_ids == set(chunk_ids), (
        "Qdrant point IDs do not match PostgreSQL chunk IDs."
    )

    for point in points:
        payload = point.payload or {}

        assert payload.get("document_id") == str(document_id)
        assert payload.get("chunk_id") == str(point.id)
        assert payload.get("content"), (
            f"Point {point.id} has no content in its payload."
        )
        assert isinstance(payload.get("chunk_index"), int), (
            f"Point {point.id} has an invalid chunk index."
        )


def verify_indexing() -> None:
    """Verify upload, indexing, payload integrity, and idempotency."""

    run_id = uuid4().hex
    filename = f"indexing_verification_{run_id}.txt"
    content = build_test_content(run_id)

    print("\n" + "=" * 65)
    print("KNOWLEDGE INTELLIGENCE PLATFORM - INDEXING VERIFICATION")
    print("=" * 65)

    print("\nSTEP 1: Uploading a synthetic document...")

    with TestClient(app) as client:
        upload_response = client.post(
            "/documents",
            files={
                "file": (
                    filename,
                    content,
                    "text/plain",
                )
            },
        )

        assert upload_response.status_code == 201, (
            "Document upload failed: "
            f"{upload_response.status_code} - {upload_response.text}"
        )

        upload_data = upload_response.json()
        document_id = UUID(upload_data["id"])

        print(f"Document ID: {document_id}")
        print(f"Filename: {upload_data['filename']}")
        print(f"Processing status: {upload_data['status']}")
        print(f"Indexing status: {upload_data['indexing_status']}")

        assert upload_data["status"] == "completed"
        assert upload_data["indexing_status"] == "completed"

        print("\nSTEP 2: Verifying PostgreSQL records...")

        chunk_ids, chunk_count = verify_database_records(document_id)

        print("Document found: Yes")
        print(f"Persisted chunks: {chunk_count}")

        print("\nSTEP 3: Verifying Qdrant vectors...")

        vector_count = count_document_vectors(document_id)

        print(f"Qdrant points: {vector_count}")

        assert vector_count == chunk_count, (
            f"PostgreSQL has {chunk_count} chunks, "
            f"but Qdrant has {vector_count} points."
        )

        verify_vector_payloads(document_id, chunk_ids)

        print("Point IDs match PostgreSQL chunk IDs: Yes")
        print("Document and chunk payloads: Valid")

        print("\nSTEP 4: Verifying idempotent re-indexing...")

        retry_response = client.post(f"/documents/{document_id}/index")

        assert retry_response.status_code == 200, (
            f"Re-indexing failed: {retry_response.status_code} - {retry_response.text}"
        )

        retry_data = retry_response.json()

        assert retry_data["indexing_status"] == "completed"

        vector_count_after_retry = count_document_vectors(document_id)

        print(f"Vectors before re-indexing: {vector_count}")
        print(f"Vectors after re-indexing: {vector_count_after_retry}")

        assert vector_count_after_retry == vector_count, (
            "Re-indexing created unexpected duplicate points."
        )

        verify_vector_payloads(document_id, chunk_ids)

    print("\n" + "=" * 65)
    print("ALL INDEXING VERIFICATIONS PASSED")
    print("=" * 65)
    print("Document ID: {document_id}")
    print(f"PostgreSQL chunks: {chunk_count}")
    print(f"Qdrant vectors: {vector_count_after_retry}")
    print("Payload integrity: Verified")
    print("Re-indexing idempotency: Verified")
    print("\nThe verification document and vectors have been retained.")


if __name__ == "__main__":
    verify_indexing()
