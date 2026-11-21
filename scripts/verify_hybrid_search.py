from uuid import UUID, uuid4

from fastapi.testclient import TestClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from sqlalchemy import select

from app.core.config import settings
from app.database.models import Chunk, Document
from app.database.session import SessionLocal
from app.main import app
from app.retrieval.bm25_search import search_bm25
from app.retrieval.hybrid_search import search_hybrid_chunks
from app.retrieval.semantic_search import search_similar_chunks
from app.vector_store.client import get_qdrant_client

POLICY_IDENTIFIER = "KIP-LOAN-742"

KEYWORD_QUERY = POLICY_IDENTIFIER

SEMANTIC_QUERY = (
    "What borrower eligibility conditions must be satisfied "
    "before a loan can be approved?"
)

HYBRID_QUERY = "KIP-LOAN-742 borrower eligibility conditions before loan approval"


def build_document_filter(document_id: UUID) -> Filter:
    """Build a Qdrant filter for one document."""

    return Filter(
        must=[
            FieldCondition(
                key="document_id",
                match=MatchValue(value=str(document_id)),
            )
        ]
    )


def count_document_vectors(document_id: UUID) -> int:
    """Return the exact number of vectors for a document."""

    result = get_qdrant_client().count(
        collection_name=settings.qdrant_collection_name,
        count_filter=build_document_filter(document_id),
        exact=True,
    )

    return result.count


def build_test_content(run_id: str) -> bytes:
    """Generate synthetic policy content for hybrid retrieval testing."""

    sections = [
        (
            f"Credit Policy Verification Run {run_id}\n\n"
            f"Policy reference: {POLICY_IDENTIFIER}.\n"
            "This policy establishes the borrower eligibility "
            "requirements that must be checked before approving "
            "a credit facility. Applicants must demonstrate "
            "verified income, acceptable repayment capacity, "
            "a valid identification record, and a satisfactory "
            "credit history. Loan officers must confirm these "
            "conditions before recommending approval."
        ),
        (
            "Repayment Assessment\n\n"
            "The credit officer must evaluate the applicant's "
            "monthly income, existing financial obligations, "
            "and expected repayment amount. The assessment must "
            "establish that the applicant can meet the proposed "
            "repayment schedule without exceeding the institution's "
            "permitted debt-service threshold."
        ),
        (
            "Identity Verification\n\n"
            "Every applicant must provide valid identification "
            "documents. The institution must verify the applicant's "
            "identity and confirm that the submitted information "
            "matches the supporting records before processing "
            "the credit application."
        ),
        (
            "Credit History Review\n\n"
            "The institution must review available credit records "
            "and identify outstanding defaults, unresolved "
            "delinquencies, and other repayment risks. Any "
            "exceptions must be documented and escalated "
            "according to the applicable approval procedure."
        ),
        (
            "Approval Controls\n\n"
            "Loan approval must follow the institution's "
            "authorization limits. Applications requiring "
            "additional approval must be escalated to the "
            "appropriate credit authority. All decisions must "
            "retain an auditable record of the review."
        ),
    ]

    content = "\n\n".join(sections)

    return content.encode("utf-8")


def verify_database_records(document_id: UUID):
    """Verify the document and chunks persisted in PostgreSQL."""

    with SessionLocal() as db:
        document = db.get(Document, document_id)

        assert document is not None, "Document was not found in PostgreSQL."
        assert document.status == "completed"
        assert document.indexing_status == "completed"

        chunks = db.scalars(
            select(Chunk)
            .where(Chunk.document_id == document_id)
            .order_by(Chunk.chunk_index)
        ).all()

        assert chunks, "No chunks were persisted."

        target_chunk = next(
            (chunk for chunk in chunks if POLICY_IDENTIFIER in chunk.content),
            None,
        )

        assert target_chunk is not None, (
            "The target policy identifier was not found in any persisted chunk."
        )

        return len(chunks), target_chunk.id, target_chunk.content


def verify_bm25_retrieval(
    db,
    document_id: UUID,
    target_chunk_id: UUID,
) -> None:
    """Verify exact-term retrieval through BM25."""

    results = search_bm25(
        db=db,
        query=KEYWORD_QUERY,
        limit=10,
        document_id=document_id,
    )

    result_ids = {result.chunk_id for result in results}

    assert target_chunk_id in result_ids, (
        "BM25 did not retrieve the chunk containing the unique policy identifier."
    )

    print(f"BM25 results: {len(results)}")
    print("Target policy chunk retrieved: Yes")


def verify_semantic_retrieval(
    document_id: UUID,
    target_chunk_id: UUID,
) -> None:
    """Verify conceptual retrieval through vector search."""

    results = search_similar_chunks(
        query=SEMANTIC_QUERY,
        limit=10,
        document_id=document_id,
    )

    result_ids = {result.chunk_id for result in results}

    assert target_chunk_id in result_ids, (
        "Semantic retrieval did not return the target policy chunk."
    )

    print(f"Semantic results: {len(results)}")
    print("Target policy chunk retrieved: Yes")


def verify_hybrid_retrieval(
    db,
    document_id: UUID,
    target_chunk_id: UUID,
) -> None:
    """Verify fused retrieval and result provenance."""

    results = search_hybrid_chunks(
        db=db,
        query=HYBRID_QUERY,
        limit=10,
        document_id=document_id,
    )

    assert results, "Hybrid retrieval returned no results."

    result_ids = {result.chunk_id for result in results}

    assert target_chunk_id in result_ids, (
        "Hybrid retrieval did not return the target policy chunk."
    )

    target_result = next(
        result for result in results if result.chunk_id == target_chunk_id
    )

    assert target_result.semantic_score is not None, (
        "Target result has no semantic retrieval score."
    )
    assert target_result.bm25_score is not None, (
        "Target result has no BM25 retrieval score."
    )
    assert target_result.rrf_score > 0

    for result in results:
        assert result.document_id == document_id
        assert result.content.strip()
        assert result.rrf_score > 0

    print(f"Hybrid results: {len(results)}")
    print("Target policy chunk retrieved: Yes")
    print(f"Target semantic score: {target_result.semantic_score:.4f}")
    print(f"Target BM25 score: {target_result.bm25_score:.4f}")
    print(f"Target RRF score: {target_result.rrf_score:.6f}")
    print("Document provenance: Valid")


def verify_document_filtering(db, document_id: UUID) -> None:
    """Verify all hybrid results respect the document filter."""

    results = search_hybrid_chunks(
        db=db,
        query=HYBRID_QUERY,
        limit=10,
        document_id=document_id,
    )

    assert all(result.document_id == document_id for result in results), (
        "Hybrid retrieval returned a result from another document."
    )

    print("Document filtering: Verified")


def verify_hybrid_search() -> None:
    """Run real PostgreSQL, embedding, Qdrant, and hybrid verification."""

    run_id = uuid4().hex
    filename = f"hybrid_verification_{run_id}.txt"
    content = build_test_content(run_id)

    print("\n" + "=" * 70)
    print("KNOWLEDGE INTELLIGENCE PLATFORM - HYBRID SEARCH VERIFICATION")
    print("=" * 70)

    print("\nSTEP 1: Uploading synthetic policy document...")

    with TestClient(app) as client:
        response = client.post(
            "/documents",
            files={
                "file": (
                    filename,
                    content,
                    "text/plain",
                )
            },
        )

        assert response.status_code == 201, (
            f"Upload failed: {response.status_code} - {response.text}"
        )

        upload_data = response.json()
        document_id = UUID(upload_data["id"])

        assert upload_data["status"] == "completed"
        assert upload_data["indexing_status"] == "completed"

        print(f"Document ID: {document_id}")
        print(f"Filename: {upload_data['filename']}")
        print(f"Processing status: {upload_data['status']}")
        print(f"Indexing status: {upload_data['indexing_status']}")

        print("\nSTEP 2: Verifying PostgreSQL and Qdrant...")

        chunk_count, target_chunk_id, target_content = verify_database_records(
            document_id
        )

        vector_count = count_document_vectors(document_id)

        assert vector_count == chunk_count, (
            f"PostgreSQL chunks: {chunk_count}; Qdrant vectors: {vector_count}."
        )

        print(f"PostgreSQL chunks: {chunk_count}")
        print(f"Qdrant vectors: {vector_count}")
        print(f"Target chunk ID: {target_chunk_id}")
        print("Target policy content: Verified")

        print("\nSTEP 3: Verifying BM25 keyword retrieval...")

        with SessionLocal() as db:
            verify_bm25_retrieval(
                db=db,
                document_id=document_id,
                target_chunk_id=target_chunk_id,
            )

            print("\nSTEP 4: Verifying semantic retrieval...")

            verify_semantic_retrieval(
                document_id=document_id,
                target_chunk_id=target_chunk_id,
            )

            print("\nSTEP 5: Verifying unified hybrid retrieval...")

            verify_hybrid_retrieval(
                db=db,
                document_id=document_id,
                target_chunk_id=target_chunk_id,
            )

            print("\nSTEP 6: Verifying document filtering...")

            verify_document_filtering(
                db=db,
                document_id=document_id,
            )

    print("\n" + "=" * 70)
    print("ALL HYBRID RETRIEVAL VERIFICATIONS PASSED")
    print("=" * 70)
    print(f"Document ID: {document_id}")
    print(f"PostgreSQL chunks: {chunk_count}")
    print(f"Qdrant vectors: {vector_count}")
    print("BM25 keyword retrieval: Verified")
    print("Semantic retrieval: Verified")
    print("Reciprocal Rank Fusion: Verified")
    print("Document filtering: Verified")
    print("Result provenance: Verified")
    print("\nVerification document and vectors have been retained.")


if __name__ == "__main__":
    verify_hybrid_search()
