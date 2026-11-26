
import json
from pathlib import Path
from uuid import UUID

from sqlalchemy import select

from app.database.models import Chunk, Document
from app.database.session import SessionLocal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "section_aware_corpus_chunks.json"
)

EXPERIMENTAL_DOCUMENTS = {
    "credit_policy.txt": "1e6aae34-ee03-48f9-81e2-7ab34e7baace",
    "loan_operations_sop.txt": "b038d050-2ffe-4a16-9722-0d35d7230a46",
    "risk_management_policy.txt": "446ef462-f688-44ba-8732-4bc943524edf",
    "compliance_policy.txt": "39f24466-e953-4a02-af45-6c0b98fde37e",
    "general_office_procedures.txt": "4e17706a-531f-408c-94d0-3ee92507c70b",
}


def export_section_aware_chunks() -> dict:
    document_ids = {
        UUID(document_id)
        for document_id in EXPERIMENTAL_DOCUMENTS.values()
    }

    db = SessionLocal()

    try:
        documents = db.scalars(
            select(Document).where(Document.id.in_(document_ids))
        ).all()

        documents_by_id = {
            document.id: document
            for document in documents
        }

        missing_ids = document_ids - set(documents_by_id)

        if missing_ids:
            raise RuntimeError(
                "Could not find all experimental documents in PostgreSQL. "
                f"Missing IDs: {sorted(str(value) for value in missing_ids)}"
            )

        for filename, document_id in EXPERIMENTAL_DOCUMENTS.items():
            document = documents_by_id[UUID(document_id)]

            expected_filename = f"section_aware__{filename}"

            if document.filename != expected_filename:
                raise RuntimeError(
                   f"Document filename mismatch for {document_id}: "
                   f"expected {expected_filename!r}, got {document.filename!r}"
                )

            if document.status != "completed":
                raise RuntimeError(
                    f"{filename} has status {document.status!r}, "
                    "expected 'completed'."
                )

            if document.indexing_status != "completed":
                raise RuntimeError(
                    f"{filename} has indexing status "
                    f"{document.indexing_status!r}, expected 'completed'."
                )

        chunks = db.scalars(
            select(Chunk)
            .where(Chunk.document_id.in_(document_ids))
            .order_by(Chunk.document_id, Chunk.chunk_index)
        ).all()

        chunks_by_document = {
            document_id: []
            for document_id in document_ids
        }

        for chunk in chunks:
            chunks_by_document[chunk.document_id].append(
                {
                    "chunk_id": str(chunk.id),
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                }
            )

        exported_documents = []

        for filename, document_id in sorted(
            EXPERIMENTAL_DOCUMENTS.items()
        ):
            document = documents_by_id[UUID(document_id)]
            document_chunks = chunks_by_document[document.id]

            exported_documents.append(
                {
                    "document_id": str(document.id),
                    "filename": document.filename,
                    "status": document.status,
                    "indexing_status": document.indexing_status,
                    "page_count": document.page_count,
                    "extracted_char_count": document.extracted_char_count,
                    "chunk_count": len(document_chunks),
                    "chunks": document_chunks,
                }
            )

        total_chunks = sum(
            document["chunk_count"]
            for document in exported_documents
        )

        if total_chunks != 24:
            raise RuntimeError(
                "Unexpected section-aware chunk count: "
                f"expected 24, found {total_chunks}. "
                "Check the experimental documents before proceeding."
            )

        return {
            "export_type": "section_aware_retrieval_error_analysis_corpus",
            "document_count": len(exported_documents),
            "total_chunk_count": total_chunks,
            "documents": exported_documents,
        }

    finally:
        db.close()


def main() -> None:
    export_data = export_section_aware_chunks()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = OUTPUT_PATH.with_suffix(".json.tmp")

    temporary_path.write_text(
        json.dumps(export_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    temporary_path.replace(OUTPUT_PATH)

    print("Section-aware corpus export completed.")
    print(f"Documents: {export_data['document_count']}")
    print(f"Total chunks: {export_data['total_chunk_count']}")
    print(f"Output: {OUTPUT_PATH}")

    for document in export_data["documents"]:
        print(
            f"  {document['filename']}: "
            f"{document['chunk_count']} chunks"
        )


if __name__ == "__main__":
    main()