
import json
from pathlib import Path

from sqlalchemy import select

from app.database.models import Chunk, Document
from app.database.session import SessionLocal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = PROJECT_ROOT / "data" / "evaluation" / "corpus"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "retrieval_error_analysis_corpus.json"
)


def export_benchmark_chunks() -> None:
    expected_files = sorted(
        path.name for path in CORPUS_DIR.glob("*.txt")
    )

    if not expected_files:
        raise RuntimeError(
            f"No benchmark corpus files found in {CORPUS_DIR}"
        )

    session = SessionLocal()

    try:
        documents = session.scalars(
            select(Document).where(
                Document.filename.in_(expected_files)
            )
        ).all()

        documents_by_name = {}

        for document in documents:
            documents_by_name.setdefault(
                document.filename, []
            ).append(document)

        missing_files = [
            filename
            for filename in expected_files
            if filename not in documents_by_name
        ]

        duplicate_files = [
            filename
            for filename, matches in documents_by_name.items()
            if len(matches) > 1
        ]

        if missing_files:
            raise RuntimeError(
                f"Benchmark documents missing from PostgreSQL: "
                f"{missing_files}"
            )

        if duplicate_files:
            raise RuntimeError(
                f"Multiple database records found for: "
                f"{duplicate_files}"
            )

        document_ids = [
            document.id for document in documents
        ]

        chunks = session.scalars(
            select(Chunk)
            .where(Chunk.document_id.in_(document_ids))
            .order_by(
                Chunk.document_id,
                Chunk.chunk_index,
            )
        ).all()

        chunks_by_document = {}

        for chunk in chunks:
            chunks_by_document.setdefault(
                chunk.document_id, []
            ).append(chunk)

        export_documents = []

        for document in sorted(
            documents,
            key=lambda item: item.filename,
        ):
            document_chunks = chunks_by_document.get(
                document.id, []
            )

            export_documents.append(
                {
                    "document_id": str(document.id),
                    "filename": document.filename,
                    "status": document.status,
                    "indexing_status": document.indexing_status,
                    "page_count": document.page_count,
                    "extracted_char_count": (
                        document.extracted_char_count
                    ),
                    "chunk_count": len(document_chunks),
                    "chunks": [
                        {
                            "chunk_id": str(chunk.id),
                            "chunk_index": chunk.chunk_index,
                            "content": chunk.content,
                            "page_start": chunk.page_start,
                            "page_end": chunk.page_end,
                            "char_start": chunk.char_start,
                            "char_end": chunk.char_end,
                        }
                        for chunk in document_chunks
                    ],
                }
            )

        export_data = {
            "export_type": "retrieval_error_analysis_corpus",
            "document_count": len(export_documents),
            "total_chunk_count": len(chunks),
            "documents": export_documents,
        }

        OUTPUT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        OUTPUT_PATH.write_text(
            json.dumps(
                export_data,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print("=" * 70)
        print("RETRIEVAL ERROR ANALYSIS - DATABASE EXPORT")
        print("=" * 70)

        for document in export_documents:
            print(
                f"{document['filename']}: "
                f"{document['chunk_count']} chunks"
            )

        print("-" * 70)
        print(f"Documents exported: {len(export_documents)}")
        print(f"Chunks exported: {len(chunks)}")
        print(f"Output: {OUTPUT_PATH}")
        print("EXPORT COMPLETED SUCCESSFULLY")

    finally:
        session.close()


if __name__ == "__main__":
    export_benchmark_chunks()