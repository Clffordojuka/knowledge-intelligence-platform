import hashlib
import json
from pathlib import Path
from statistics import mean
from time import perf_counter
from uuid import UUID

from fastapi.testclient import TestClient
from sqlalchemy import select

from app.core.config import settings
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
from app.retrieval.reranking import RerankingError, get_reranker
from app.retrieval.semantic_search import search_similar_chunks

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_DIR = PROJECT_ROOT / "data" / "evaluation"

CORPUS_DIR = EVALUATION_DIR / "corpus"
DATASET_PATH = EVALUATION_DIR / "retrieval_dataset_v2.json"
RESULTS_PATH = EVALUATION_DIR / "retrieval_results_v2.json"

TOP_K = 5
HYBRID_CANDIDATE_LIMIT = 20

STRATEGIES = (
    "semantic",
    "bm25",
    "hybrid",
    "hybrid_reranked",
)

METRIC_NAMES = (
    "precision_at_5",
    "recall_at_5",
    "mrr",
    "ndcg_at_5",
)


class RetrievalEvaluationError(Exception):
    """Raised when retrieval evaluation cannot be completed."""


def normalize_text(text: str) -> str:
    """Normalize whitespace and capitalization for evidence matching."""

    return " ".join(text.casefold().split())


def load_dataset() -> dict:
    """Load and validate the multi-document retrieval dataset."""

    try:
        dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RetrievalEvaluationError(
            "Unable to load the retrieval evaluation dataset."
        ) from exc

    if not isinstance(dataset, dict):
        raise RetrievalEvaluationError(
            "The evaluation dataset must be a JSON object."
        )

    corpus_files = dataset.get("corpus_files")
    queries = dataset.get("queries")

    if (
        not isinstance(corpus_files, list)
        or not corpus_files
        or any(
            not isinstance(filename, str) or not filename.strip()
            for filename in corpus_files
        )
    ):
        raise RetrievalEvaluationError(
            "The dataset must contain valid corpus filenames."
        )

    if len(corpus_files) != len(set(corpus_files)):
        raise RetrievalEvaluationError("Duplicate corpus filenames.")

    for filename in corpus_files:
        if Path(filename).name != filename or "\\" in filename or "/" in filename:
            raise RetrievalEvaluationError(
                f"Invalid corpus filename: {filename}"
            )

        if not (CORPUS_DIR / filename).is_file():
            raise RetrievalEvaluationError(
                f"Corpus document does not exist: {filename}"
            )

    if not isinstance(queries, list) or not queries:
        raise RetrievalEvaluationError(
            "The evaluation dataset must contain queries."
        )

    query_ids = set()

    for item in queries:
        if not isinstance(item, dict):
            raise RetrievalEvaluationError(
                "Every evaluation query must be an object."
            )

        query_id = item.get("id")
        query = item.get("query")
        evidence = item.get("relevant_evidence")

        if not isinstance(query_id, str) or not query_id.strip():
            raise RetrievalEvaluationError("Invalid query ID.")

        if query_id in query_ids:
            raise RetrievalEvaluationError(
                f"Duplicate query ID: {query_id}"
            )

        if not isinstance(query, str) or not query.strip():
            raise RetrievalEvaluationError(
                f"Query {query_id} has invalid query text."
            )

        if not isinstance(evidence, list) or not evidence:
            raise RetrievalEvaluationError(
                f"Query {query_id} has no relevance evidence."
            )

        for entry in evidence:
            if not isinstance(entry, dict):
                raise RetrievalEvaluationError(
                    f"Query {query_id} has invalid evidence."
                )

            filename = entry.get("filename")
            phrase = entry.get("phrase")

            if filename not in corpus_files:
                raise RetrievalEvaluationError(
                    f"Query {query_id} references an unknown "
                    f"corpus document: {filename}"
                )

            if not isinstance(phrase, str) or not phrase.strip():
                raise RetrievalEvaluationError(
                    f"Query {query_id} has an invalid evidence phrase."
                )

        query_ids.add(query_id)

    return dataset


def get_or_upload_benchmark_documents(
    corpus_files: list[str],
) -> dict[str, UUID]:
    """Reuse or upload and index all benchmark corpus documents."""

    document_ids = {}
    files_to_upload = []

    for filename in corpus_files:
        corpus_path = CORPUS_DIR / filename

        try:
            content = corpus_path.read_bytes()
        except OSError as exc:
            raise RetrievalEvaluationError(
                f"Unable to read benchmark document: {filename}"
            ) from exc

        content_hash = hashlib.sha256(content).hexdigest()

        with SessionLocal() as db:
            document = db.scalar(
                select(Document).where(Document.sha256 == content_hash)
            )

            if document is not None:
                if (
                    document.status != "completed"
                    or document.indexing_status != "completed"
                ):
                    raise RetrievalEvaluationError(
                        f"Benchmark document {filename} exists but is "
                        "not fully processed and indexed."
                    )

                document_ids[filename] = document.id
            else:
                files_to_upload.append((filename, content))

    if files_to_upload:
        with TestClient(app) as client:
            for filename, content in files_to_upload:
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
                        f"Benchmark document upload failed for {filename}: "
                        f"{response.status_code} - {response.text}"
                    )

                upload_data = response.json()

                if (
                    upload_data.get("status") != "completed"
                    or upload_data.get("indexing_status") != "completed"
                ):
                    raise RetrievalEvaluationError(
                        f"Benchmark document {filename} "
                        "was not fully indexed."
                    )

                document_ids[filename] = UUID(upload_data["id"])

    if len(document_ids) != len(corpus_files):
        raise RetrievalEvaluationError(
            "Not all benchmark documents were registered."
        )

    return document_ids


def resolve_relevant_chunks(
    document_ids: dict[str, UUID],
    evidence_entries: list[dict],
) -> set[UUID]:
    """Resolve evidence phrases to chunks in their specified documents."""

    chunks_by_document = {}

    for entry in evidence_entries:
        filename = entry["filename"]

        if filename not in chunks_by_document:
            document_id = document_ids[filename]

            with SessionLocal() as db:
                chunks = db.scalars(
                    select(Chunk)
                    .where(Chunk.document_id == document_id)
                    .order_by(Chunk.chunk_index)
                ).all()

            chunks_by_document[filename] = chunks

    relevant_ids = set()

    for entry in evidence_entries:
        filename = entry["filename"]
        phrase = normalize_text(entry["phrase"])

        matching_chunks = [
            chunk.id
            for chunk in chunks_by_document[filename]
            if phrase in normalize_text(chunk.content)
        ]

        if not matching_chunks:
            raise RetrievalEvaluationError(
                "Could not resolve relevance evidence to a persisted chunk. "
                f"Query evidence: {entry['phrase']!r}; "
                f"document: {filename!r}. "
                "Check chunk boundaries and the benchmark corpus."
            )

        relevant_ids.update(matching_chunks)

    return relevant_ids


def calculate_metrics(
    retrieved_ids: list[UUID],
    relevant_ids: set[UUID],
) -> dict[str, float]:
    """Calculate the four retrieval metrics for one result ranking."""

    relevant_list = list(relevant_ids)

    return {
        "precision_at_5": precision_at_k(
            retrieved_ids,
            relevant_list,
            k=TOP_K,
        ),
        "recall_at_5": recall_at_k(
            retrieved_ids,
            relevant_list,
            k=TOP_K,
        ),
        "mrr": reciprocal_rank(
            retrieved_ids,
            relevant_list,
        ),
        "ndcg_at_5": ndcg_at_k(
            retrieved_ids,
            relevant_list,
            k=TOP_K,
        ),
    }


def evaluate_query(
    db,
    query_item: dict,
    document_ids: dict[str, UUID],
    reranker=None,
) -> dict:
    """Evaluate all four retrieval strategies within the benchmark corpus."""

    query_id = query_item["id"]
    query = query_item["query"]

    relevant_ids = resolve_relevant_chunks(
        document_ids=document_ids,
        evidence_entries=query_item["relevant_evidence"],
    )

    # Apply the benchmark corpus restriction inside each retrieval service.
    benchmark_document_ids = list(document_ids.values())

    latencies = {}

    start = perf_counter()
    semantic_results = search_similar_chunks(
        query=query,
        limit=TOP_K,
        document_ids=benchmark_document_ids,
    )
    latencies["semantic"] = (perf_counter() - start) * 1000

    start = perf_counter()
    bm25_results = search_bm25(
        db=db,
        query=query,
        limit=TOP_K,
        document_ids=benchmark_document_ids,
    )
    latencies["bm25"] = (perf_counter() - start) * 1000

    start = perf_counter()
    hybrid_results = search_hybrid_chunks(
        db=db,
        query=query,
        limit=TOP_K,
        candidate_limit=HYBRID_CANDIDATE_LIMIT,
        document_ids=benchmark_document_ids,
    )
    latencies["hybrid"] = (perf_counter() - start) * 1000

    start = perf_counter()
    hybrid_reranked_results = search_hybrid_chunks(
        db=db,
        query=query,
        limit=TOP_K,
        candidate_limit=HYBRID_CANDIDATE_LIMIT,
        rerank=True,
        reranker=reranker,
        document_ids=benchmark_document_ids,
    )
    latencies["hybrid_reranked"] = (perf_counter() - start) * 1000

    rankings = {
        "semantic": semantic_results,
        "bm25": bm25_results,
        "hybrid": hybrid_results,
        "hybrid_reranked": hybrid_reranked_results,
    }

    # Defensive verification: retrieval must never escape the benchmark scope.
    benchmark_document_id_set = set(benchmark_document_ids)

    for strategy, results in rankings.items():
        if any(
            result.document_id not in benchmark_document_id_set
            for result in results
        ):
            raise RetrievalEvaluationError(
                f"{strategy} returned a result outside the benchmark corpus."
            )

    return {
        "query_id": query_id,
        "query": query,
        "topic": query_item.get("topic"),
        "relevant_chunk_count": len(relevant_ids),
        "strategies": {
            strategy: {
                "retrieved_count": len(results),
                "retrieved_chunk_ids": [
                    str(result.chunk_id) for result in results
                ],
                "retrieved_document_ids": [
                    str(result.document_id) for result in results
                ],
                "latency_ms": round(latencies[strategy], 3),
                "metrics": calculate_metrics(
                    [result.chunk_id for result in results],
                    relevant_ids,
                ),
            }
            for strategy, results in rankings.items()
        },
    }


def calculate_aggregate_metrics(query_results: list[dict]) -> dict:
    """Calculate macro-averaged retrieval metrics and latency."""

    if not query_results:
        raise RetrievalEvaluationError(
            "Cannot aggregate an empty evaluation result."
        )

    aggregate = {}

    for strategy in STRATEGIES:
        aggregate[strategy] = {
            metric: mean(
                result["strategies"][strategy]["metrics"][metric]
                for result in query_results
            )
            for metric in METRIC_NAMES
        }

        aggregate[strategy]["mean_latency_ms"] = mean(
            result["strategies"][strategy]["latency_ms"]
            for result in query_results
        )

    return aggregate


def run_retrieval_evaluation() -> dict:
    """Run and save the multi-document retrieval evaluation."""

    dataset = load_dataset()

    document_ids = get_or_upload_benchmark_documents(
        dataset["corpus_files"]
    )

    print("\nLoading cross-encoder model...")

    reranker_load_started = perf_counter()

    try:
        reranker = get_reranker()
    except RerankingError as exc:
        raise RetrievalEvaluationError(
            "Unable to initialize the cross-encoder for evaluation."
        ) from exc

    reranker_load_latency_ms = (
        perf_counter() - reranker_load_started
    ) * 1000

    print(
        "Cross-encoder initialized in "
        f"{reranker_load_latency_ms:.2f} ms"
    )

    print("\n" + "=" * 70)
    print("KNOWLEDGE INTELLIGENCE PLATFORM - MULTI-DOCUMENT EVALUATION")
    print("=" * 70)

    print(f"\nDataset: {dataset['dataset_name']}")
    print(f"Dataset version: {dataset['version']}")
    print(f"Corpus documents: {len(document_ids)}")
    print(f"Evaluation queries: {len(dataset['queries'])}")
    print(f"Evaluation cutoff: {TOP_K}")

    print("\nRegistered benchmark documents:")

    for filename, document_id in document_ids.items():
        print(f"  {filename}: {document_id}")

    query_results = []

    with SessionLocal() as db:
        for index, query_item in enumerate(dataset["queries"], start=1):
            print(
                f"\nEvaluating query {index}/{len(dataset['queries'])}: "
                f"{query_item['id']}"
            )

            result = evaluate_query(
                db=db,
                query_item=query_item,
                document_ids=document_ids,
                reranker=reranker,
            )

            query_results.append(result)

            for strategy, strategy_result in result["strategies"].items():
                metrics = strategy_result["metrics"]

                print(
                    f"  {strategy:16} | "
                    f"P@5: {metrics['precision_at_5']:.3f} | "
                    f"R@5: {metrics['recall_at_5']:.3f} | "
                    f"MRR: {metrics['mrr']:.3f} | "
                    f"nDCG@5: {metrics['ndcg_at_5']:.3f} | "
                    f"Latency: {strategy_result['latency_ms']:.2f} ms"
                )

    report = {
        "dataset_name": dataset["dataset_name"],
        "dataset_version": dataset["version"],
        "corpus_documents": {
            filename: str(document_id)
            for filename, document_id in document_ids.items()
        },
        "corpus_document_count": len(document_ids),
        "evaluation_cutoff": TOP_K,
        "query_count": len(query_results),
        "reranker": {
            "model_name": settings.reranker_model_name,
            "model_load_latency_ms": round(
                reranker_load_latency_ms,
                3,
            ),
        },
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

    for strategy, strategy_metrics in report["aggregate_metrics"].items():
        print(f"\n{strategy.upper()}")

        for metric, value in strategy_metrics.items():
            print(f"  {metric}: {value:.4f}")

    print("\n" + "=" * 70)
    print("MULTI-DOCUMENT RETRIEVAL EVALUATION COMPLETED")
    print("=" * 70)
    print(f"Results saved to: {RESULTS_PATH}")
    print(f"Evaluated queries: {len(query_results)}")

    return report