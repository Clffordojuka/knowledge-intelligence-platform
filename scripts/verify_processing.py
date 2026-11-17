
import argparse
from uuid import UUID

from sqlalchemy import select

from app.database.models import Chunk, Document
from app.database.session import SessionLocal
from app.ingestion.services.processing_service import process_document


def main(document_id: UUID) -> None:
    with SessionLocal() as db:
        document = db.get(Document, document_id)

        if document is None:
            raise SystemExit(
                f"Document {document_id} was not found."
            )

        print("\nDOCUMENT BEFORE PROCESSING")
        print(f"ID: {document.id}")
        print(f"Filename: {document.filename}")
        print(f"Content type: {document.content_type}")
        print(f"Status: {document.status}")

        chunks = process_document(db, document_id)

        db.refresh(document)

        stored_chunks = db.scalars(
            select(Chunk)
            .where(Chunk.document_id == document_id)
            .order_by(Chunk.chunk_index)
        ).all()

        print("\nDOCUMENT AFTER PROCESSING")
        print(f"Status: {document.status}")
        print(f"Pages: {document.page_count}")
        print(f"Extracted characters: {document.extracted_char_count}")
        print(f"Persisted chunks: {len(stored_chunks)}")

        print("\nCHUNK DETAILS")

        for chunk in stored_chunks:
            print(f"\nChunk index: {chunk.chunk_index}")
            print(f"Pages: {chunk.page_start}-{chunk.page_end}")
            print(f"Offsets: {chunk.char_start}-{chunk.char_end}")
            print(f"Content: {chunk.content[:250]}")

        assert document.status == "completed"
        assert len(chunks) == len(stored_chunks)
        assert len(stored_chunks) > 0

        print("\nPostgreSQL processing verification passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify document processing against PostgreSQL."
    )
    parser.add_argument(
        "document_id",
        type=UUID,
        help="UUID of the uploaded document to process.",
    )

    args = parser.parse_args()
    main(args.document_id)
