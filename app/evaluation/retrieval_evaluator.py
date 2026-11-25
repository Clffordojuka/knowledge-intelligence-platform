
import hashlib
import json
from pathlib import Path
from random import Random
from statistics import mean, median
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

RESULTS_PATH = (
    EVALUATION_DIR / "retrieval_results_reliability_experiment.json"
)

TOP_K = 5
HYBRID_CANDIDATE_LIMIT = 20
WARMUP_RUNS = 1
MEASURED_TRIALS = 5
RANDOM_SEED = 42

# Fixed instruction for the controlled BGE query-instruction experiment.
# The same instruction is used for semantic and hybrid instructed retrieval.
QUERY_INSTRUCTION = (
    "Represent this sentence for searching relevant passages: "
)

STRATEGIES = (
    "semantic",
    "semantic_instructed",
    "bm25",
    "hybrid",
    "hybrid_instructed",
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
        if (
            Path(filename).name != filename
            or "\\" in filename
            or "/" in filename
        ):
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


def _build_retrieval_operations(
    db,
    query: str,
    benchmark_document_ids: list[UUID],
    reranker=None,
) -> dict:
    """Build the six retrieval operations for one query."""

    return {
        "semantic": lambda: search_similar_chunks(
            query=query,
            limit=TOP_K,
            document_ids=benchmark_document_ids,
        ),
        "semantic_instructed": lambda: search_similar_chunks(
            query=query,
            limit=TOP_K,
            document_ids=benchmark_document_ids,
            query_instruction=QUERY_INSTRUCTION,
        ),
        "bm25": lambda: search_bm25(
            db=db,
            query=query,
            limit=TOP_K,
            document_ids=benchmark_document_ids,
        ),
        "hybrid": lambda: search_hybrid_chunks(
            db=db,
            query=query,
            limit=TOP_K,
            candidate_limit=HYBRID_CANDIDATE_LIMIT,
            document_ids=benchmark_document_ids,
        ),
        "hybrid_instructed": lambda: search_hybrid_chunks(
            db=db,
            query=query,
            limit=TOP_K,
            candidate_limit=HYBRID_CANDIDATE_LIMIT,
            document_ids=benchmark_document_ids,
            query_instruction=QUERY_INSTRUCTION,
        ),
        "hybrid_reranked": lambda: search_hybrid_chunks(
            db=db,
            query=query,
            limit=TOP_K,
            candidate_limit=HYBRID_CANDIDATE_LIMIT,
            rerank=True,
            reranker=reranker,
            document_ids=benchmark_document_ids,
        ),
    }


def _summarize_latency(latencies: list[float]) -> dict[str, float | list[float]]:
    """Summarize measured wall-clock latency values in milliseconds."""

    if not latencies:
        raise RetrievalEvaluationError(
            "Cannot summarize an empty latency sample."
        )

    return {
        "mean_ms": round(mean(latencies), 3),
        "median_ms": round(median(latencies), 3),
        "min_ms": round(min(latencies), 3),
        "max_ms": round(max(latencies), 3),
        "trials_ms": [round(value, 3) for value in latencies],
    }


def evaluate_query(
    db,
    query_item: dict,
    document_ids: dict[str, UUID],
    reranker=None,
    *,
    warmup_runs: int = WARMUP_RUNS,
    measured_trials: int = MEASURED_TRIALS,
    rng: Random | None = None,
) -> dict:
    """Evaluate retrieval strategies using warm-ups and randomized trials."""

    if (
        isinstance(warmup_runs, bool)
        or not isinstance(warmup_runs, int)
        or warmup_runs < 0
    ):
        raise RetrievalEvaluationError(
            "warmup_runs must be a non-negative integer."
        )

    if (
        isinstance(measured_trials, bool)
        or not isinstance(measured_trials, int)
        or measured_trials < 1
    ):
        raise RetrievalEvaluationError(
            "measured_trials must be a positive integer."
        )

    query_id = query_item["id"]
    query = query_item["query"]

    relevant_ids = resolve_relevant_chunks(
        document_ids=document_ids,
        evidence_entries=query_item["relevant_evidence"],
    )

    # Every strategy is restricted to the same benchmark corpus.
    benchmark_document_ids = list(document_ids.values())
    benchmark_document_id_set = set(benchmark_document_ids)

    operations = _build_retrieval_operations(
        db=db,
        query=query,
        benchmark_document_ids=benchmark_document_ids,
        reranker=reranker,
    )

    randomizer = rng if rng is not None else Random(RANDOM_SEED)

    # Warm-up calls are deliberately excluded from metrics and latency.
    warmup_order = list(STRATEGIES)
    randomizer.shuffle(warmup_order)

    for _ in range(warmup_runs):
        for strategy in warmup_order:
            operations[strategy]()

    latency_samples = {strategy: [] for strategy in STRATEGIES}
    metric_samples = {strategy: [] for strategy in STRATEGIES}
    representative_rankings = {}
    trial_orders = []

    # Each trial executes all strategies once, in a newly shuffled order.
    for trial_number in range(measured_trials):
        trial_order = list(STRATEGIES)
        randomizer.shuffle(trial_order)
        trial_orders.append(trial_order)

        for strategy in trial_order:
            started_at = perf_counter()
            results = operations[strategy]()
            elapsed_ms = (perf_counter() - started_at) * 1000

            if any(
                result.document_id not in benchmark_document_id_set
                for result in results
            ):
                raise RetrievalEvaluationError(
                    f"{strategy} returned a result outside the benchmark "
                    "corpus."
                )

            retrieved_ids = [result.chunk_id for result in results]

            latency_samples[strategy].append(elapsed_ms)
            metric_samples[strategy].append(
                calculate_metrics(retrieved_ids, relevant_ids)
            )

            # Retain the first measured ranking as a concrete example.
            if trial_number == 0:
                representative_rankings[strategy] = results

    strategy_results = {}

    for strategy in STRATEGIES:
        results = representative_rankings[strategy]
        latency_summary = _summarize_latency(latency_samples[strategy])

        averaged_metrics = {
            metric: mean(
                sample[metric] for sample in metric_samples[strategy]
            )
            for metric in METRIC_NAMES
        }

        strategy_results[strategy] = {
            "retrieved_count": len(results),
            "retrieved_chunk_ids": [
                str(result.chunk_id) for result in results
            ],
            "retrieved_document_ids": [
                str(result.document_id) for result in results
            ],
            # Keep latency_ms as a compatibility field; it now represents
            # the median of the measured trials.
            "latency_ms": latency_summary["median_ms"],
            "latency": latency_summary,
            "metrics": {
                metric: round(value, 6)
                for metric, value in averaged_metrics.items()
            },
        }

    return {
        "query_id": query_id,
        "query": query,
        "topic": query_item.get("topic"),
        "relevant_chunk_count": len(relevant_ids),
        "warmup_runs_per_strategy": warmup_runs,
        "measured_trials_per_strategy": measured_trials,
        "trial_execution_orders": trial_orders,
        "strategies": strategy_results,
    }


def calculate_aggregate_metrics(query_results: list[dict]) -> dict:
    """Calculate macro-averaged retrieval metrics and latency statistics."""

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

        all_trial_latencies = [
            latency
            for result in query_results
            for latency in result["strategies"][strategy]["latency"]["trials_ms"]
        ]

        per_query_mean_latencies = [
            result["strategies"][strategy]["latency"]["mean_ms"]
            for result in query_results
        ]
        per_query_median_latencies = [
            result["strategies"][strategy]["latency"]["median_ms"]
            for result in query_results
        ]

        aggregate[strategy]["mean_latency_ms"] = mean(
            per_query_mean_latencies
        )
        aggregate[strategy]["median_latency_ms"] = median(
            per_query_median_latencies
        )
        aggregate[strategy]["overall_trial_mean_latency_ms"] = mean(
            all_trial_latencies
        )
        aggregate[strategy]["overall_trial_median_latency_ms"] = median(
            all_trial_latencies
        )

    return aggregate


def run_retrieval_evaluation() -> dict:
    """Run and save the repeated, randomized retrieval reliability experiment."""

    dataset = load_dataset()
    document_ids = get_or_upload_benchmark_documents(dataset["corpus_files"])

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

    print("\n" + "=" * 75)
    print("KNOWLEDGE INTELLIGENCE PLATFORM")
    print("RETRIEVAL EVALUATION RELIABILITY EXPERIMENT")
    print("=" * 75)

    print(f"\nDataset: {dataset['dataset_name']}")
    print(f"Dataset version: {dataset['version']}")
    print(f"Corpus documents: {len(document_ids)}")
    print(f"Evaluation queries: {len(dataset['queries'])}")
    print(f"Evaluation cutoff: {TOP_K}")
    print(f"Hybrid candidate limit: {HYBRID_CANDIDATE_LIMIT}")
    print(f"Warm-up runs per strategy/query: {WARMUP_RUNS}")
    print(f"Measured trials per strategy/query: {MEASURED_TRIALS}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Query instruction: {QUERY_INSTRUCTION!r}")
    print(f"Embedding model: {settings.embedding_model_name}")

    print("\nRegistered benchmark documents:")
    for filename, document_id in document_ids.items():
        print(f"  {filename}: {document_id}")

    query_results = []
    randomizer = Random(RANDOM_SEED)

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
                warmup_runs=WARMUP_RUNS,
                measured_trials=MEASURED_TRIALS,
                rng=randomizer,
            )
            query_results.append(result)

            for strategy, strategy_result in result["strategies"].items():
                metrics = strategy_result["metrics"]
                latency = strategy_result["latency"]

                print(
                    f"  {strategy:22} | "
                    f"P@5: {metrics['precision_at_5']:.3f} | "
                    f"R@5: {metrics['recall_at_5']:.3f} | "
                    f"MRR: {metrics['mrr']:.3f} | "
                    f"nDCG@5: {metrics['ndcg_at_5']:.3f} | "
                    f"Mean: {latency['mean_ms']:.2f} ms | "
                    f"Median: {latency['median_ms']:.2f} ms"
                )

    report = {
        "experiment": {
            "name": "Retrieval evaluation reliability",
            "query_instruction": QUERY_INSTRUCTION,
            "embedding_model": settings.embedding_model_name,
            "top_k": TOP_K,
            "hybrid_candidate_limit": HYBRID_CANDIDATE_LIMIT,
            "strategies": list(STRATEGIES),
            "warmup_runs_per_strategy": WARMUP_RUNS,
            "measured_trials_per_strategy": MEASURED_TRIALS,
            "random_seed": RANDOM_SEED,
            "randomized_order": True,
            "purpose": (
                "Measure retrieval metrics and latency over repeated trials "
                "after warm-up, with randomized strategy execution order."
            ),
            "latency_methodology": (
                "Warm-up calls are excluded. Each measured trial executes "
                "every strategy once in a seeded shuffled order. Per-query "
                "mean, median, minimum, maximum, and raw trial latencies are "
                "recorded. Aggregate metrics are macro-averaged across queries."
            ),
        },
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

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 75)
    print("AGGREGATE RETRIEVAL RESULTS")
    print("=" * 75)

    for strategy, strategy_metrics in report["aggregate_metrics"].items():
        print(f"\n{strategy.upper()}")
        for metric, value in strategy_metrics.items():
            print(f"  {metric}: {value:.4f}")

    print("\n" + "=" * 75)
    print("RETRIEVAL RELIABILITY EXPERIMENT COMPLETED")
    print("=" * 75)
    print(f"Results saved to: {RESULTS_PATH}")
    print(f"Evaluated queries: {len(query_results)}")
    print(
        "Measured retrieval calls: "
        f"{len(query_results) * len(STRATEGIES) * MEASURED_TRIALS}"
    )
    print(
        "Warm-up retrieval calls: "
        f"{len(query_results) * len(STRATEGIES) * WARMUP_RUNS}"
    )

    return report


if __name__ == "__main__":
    run_retrieval_evaluation()