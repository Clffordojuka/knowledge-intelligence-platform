import hashlib
import json
from pathlib import Path
from statistics import mean
from uuid import UUID

from fastapi.testclient import TestClient
from sqlalchemy import select

from app.database.models import Chunk, Document
from app.database.session import SessionLocal
from app.evaluation.retrieval_metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from app.main import app
from app.retrieval.bm25_search import search_bm25
from app.retrieval.hybrid_search import search_hybrid_chunks
from app.retrieval.semantic_search import search_similar_chunks

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_DIR = PROJECT_ROOT / "data" / "evaluation"

CORPUS_PATH = EVALUATION_DIR / "benchmark_policy.txt"
DATASET_PATH = EVALUATION_DIR / "retrieval_dataset.json"
RESULTS_PATH = EVALUATION_DIR / "retrieval_results.json"

TOP_K = 5
HYBRID_CANDIDATE_LIMIT = 20


class RetrievalEvaluationError(Exception):
    """Raised when retrieval evaluation cannot be completed."""


def load_dataset() -> dict:
    """Load and validate the retrieval evaluation dataset."""

    try:
        dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RetrievalEvaluationError(
            "Unable to load the retrieval evaluation dataset."
        ) from exc

    queries = dataset.get("queries")

    if not isinstance(queries, list) or not queries:
        raise RetrievalEvaluationError("The evaluation dataset must contain queries.")

    query_ids = set()

    for item in queries:
        if not isinstance(item, dict):
            raise RetrievalEvaluationError("Every evaluation query must be an object.")

        query_id = item.get("query_id")
        query = item.get("query")
        evidence = item.get("relevant_evidence")

        if not isinstance(query_id, str) or not query_id.strip():
            raise RetrievalEvaluationError("Invalid query ID.")

        if query_id in query_ids:
            raise RetrievalEvaluationError(f"Duplicate query ID: {query_id}")

        if not isinstance(query, str) or not query.strip():
            raise RetrievalEvaluationError(f"Query {query_id} has invalid query text.")

        if (
            not isinstance(evidence, list)
            or not evidence
            or any(
                not isinstance(phrase, str) or not phrase.strip() for phrase in evidence
            )
        ):
            raise RetrievalEvaluationError(
                f"Query {query_id} has invalid relevance evidence."
            )

        query_ids.add(query_id)

    return dataset


def get_or_upload_benchmark_document() -> UUID:
    """Reuse the benchmark document or upload it if it does not exist."""

    try:
        content = CORPUS_PATH.read_bytes()
    except OSError as exc:
        raise RetrievalEvaluationError("Unable to read the benchmark corpus.") from exc

    content_hash = hashlib.sha256(content).hexdigest()

    with SessionLocal() as db:
        document = db.scalar(select(Document).where(Document.sha256 == content_hash))

        if document is not None:
            if (
                document.status != "completed"
                or document.indexing_status != "completed"
            ):
                raise RetrievalEvaluationError(
                    "The existing benchmark document is not fully processed "
                    "and indexed."
                )

            return document.id

    filename = CORPUS_PATH.name

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

    if response.status_code != 201:
        raise RetrievalEvaluationError(
            "Benchmark document upload failed: "
            f"{response.status_code} - {response.text}"
        )

    upload_data = response.json()

    if (
        upload_data.get("status") != "completed"
        or upload_data.get("indexing_status") != "completed"
    ):
        raise RetrievalEvaluationError("Benchmark document was not fully indexed.")

    return UUID(upload_data["id"])


def resolve_relevant_chunks(
    document_id: UUID,
    evidence_phrases: list[str],
) -> set[UUID]:
    """Resolve manually labelled evidence phrases to persisted chunk IDs."""

    with SessionLocal() as db:
        chunks = db.scalars(
            select(Chunk)
            .where(Chunk.document_id == document_id)
            .order_by(Chunk.chunk_index)
        ).all()

    relevant_ids = set()

    for phrase in evidence_phrases:
        matches = [
            chunk.id
            for chunk in chunks
            if phrase.casefold() in chunk.content.casefold()
        ]

        if not matches:
            raise RetrievalEvaluationError(
                "Could not resolve relevance evidence to a chunk: "
                f"{phrase!r}. Check chunk boundaries and the benchmark corpus."
            )

        relevant_ids.update(matches)

    return relevant_ids


def calculate_metrics(
    retrieved_ids: list[UUID],
    relevant_ids: set[UUID],
) -> dict[str, float]:
    """Calculate the four retrieval metrics for one result ranking."""

    return {
        "precision_at_5": precision_at_k(
            retrieved_ids,
            list(relevant_ids),
            k=TOP_K,
        ),
        "recall_at_5": recall_at_k(
            retrieved_ids,
            list(relevant_ids),
            k=TOP_K,
        ),
        "mrr": reciprocal_rank(
            retrieved_ids,
            list(relevant_ids),
        ),
        "ndcg_at_5": ndcg_at_k(
            retrieved_ids,
            list(relevant_ids),
            k=TOP_K,
        ),
    }


def evaluate_query(
    db,
    query_item: dict,
    document_id: UUID,
) -> dict:
    """Evaluate all three retrieval strategies for one query."""

    query_id = query_item["query_id"]
    query = query_item["query"]

    relevant_ids = resolve_relevant_chunks(
        document_id=document_id,
        evidence_phrases=query_item["relevant_evidence"],
    )

    semantic_results = search_similar_chunks(
        query=query,
        limit=TOP_K,
        document_id=document_id,
    )

    bm25_results = search_bm25(
        db=db,
        query=query,
        limit=TOP_K,
        document_id=document_id,
    )

    hybrid_results = search_hybrid_chunks(
        db=db,
        query=query,
        limit=TOP_K,
        candidate_limit=HYBRID_CANDIDATE_LIMIT,
        document_id=document_id,
    )

    rankings = {
        "semantic": [result.chunk_id for result in semantic_results],
        "bm25": [result.chunk_id for result in bm25_results],
        "hybrid": [result.chunk_id for result in hybrid_results],
    }

    return {
        "query_id": query_id,
        "query": query,
        "topic": query_item.get("topic"),
        "relevant_chunk_count": len(relevant_ids),
        "strategies": {
            strategy: {
                "retrieved_count": len(retrieved_ids),
                "retrieved_chunk_ids": [str(chunk_id) for chunk_id in retrieved_ids],
                "metrics": calculate_metrics(
                    retrieved_ids,
                    relevant_ids,
                ),
            }
            for strategy, retrieved_ids in rankings.items()
        },
    }


def calculate_aggregate_metrics(query_results: list[dict]) -> dict:
    """Calculate macro-average metrics across all evaluation queries."""

    strategies = ["semantic", "bm25", "hybrid"]

    aggregate = {}

    for strategy in strategies:
        metric_names = [
            "precision_at_5",
            "recall_at_5",
            "mrr",
            "ndcg_at_5",
        ]

        aggregate[strategy] = {
            metric: mean(
                result["strategies"][strategy]["metrics"][metric]
                for result in query_results
            )
            for metric in metric_names
        }

    return aggregate


def run_retrieval_evaluation() -> dict:
    """Run and save the complete retrieval evaluation."""

    dataset = load_dataset()

    document_id = get_or_upload_benchmark_document()

    print("\n" + "=" * 70)
    print("KNOWLEDGE INTELLIGENCE PLATFORM - RETRIEVAL EVALUATION")
    print("=" * 70)

    print(f"\nDataset: {dataset['dataset_name']}")
    print(f"Dataset version: {dataset['version']}")
    print(f"Benchmark document: {document_id}")
    print(f"Evaluation queries: {len(dataset['queries'])}")
    print(f"Evaluation cutoff: {TOP_K}")

    query_results = []

    with SessionLocal() as db:
        for index, query_item in enumerate(dataset["queries"], start=1):
            print(
                f"\nEvaluating query {index}/{len(dataset['queries'])}: "
                f"{query_item['query_id']}"
            )

            result = evaluate_query(
                db=db,
                query_item=query_item,
                document_id=document_id,
            )

            query_results.append(result)

            for strategy, strategy_result in result["strategies"].items():
                metrics = strategy_result["metrics"]

                print(
                    f"  {strategy:8} | "
                    f"P@5: {metrics['precision_at_5']:.3f} | "
                    f"R@5: {metrics['recall_at_5']:.3f} | "
                    f"MRR: {metrics['mrr']:.3f} | "
                    f"nDCG@5: {metrics['ndcg_at_5']:.3f}"
                )

    report = {
        "dataset_name": dataset["dataset_name"],
        "dataset_version": dataset["version"],
        "document_id": str(document_id),
        "evaluation_cutoff": TOP_K,
        "query_count": len(query_results),
        "aggregate_metrics": calculate_aggregate_metrics(query_results),
        "query_results": query_results,
    }

    RESULTS_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("AGGREGATE RETRIEVAL RESULTS")
    print("=" * 70)

    for strategy, metrics in report["aggregate_metrics"].items():
        print(f"\n{strategy.upper()}")

        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

    print("\n" + "=" * 70)
    print("RETRIEVAL EVALUATION COMPLETED")
    print("=" * 70)
    print(f"Results saved to: {RESULTS_PATH}")
    print(f"Evaluated queries: {len(query_results)}")

    return report
