
import asyncio
import hashlib
import json
from io import BytesIO
from random import Random
from time import perf_counter
from uuid import UUID

from fastapi import UploadFile
from sqlalchemy import func, select
from starlette.datastructures import Headers

from app.database.models import Chunk, Document
from app.database.session import SessionLocal
from app.evaluation import retrieval_evaluator as evaluator
from app.ingestion.chunking.chunker import ChunkingConfig
from app.ingestion.services.document_service import persist_document
from app.ingestion.services.processing_service import process_document
from app.ingestion.services.storage import store_upload
from app.retrieval.reranking import RerankingError, get_reranker
from app.vector_store.indexing import index_document

EXPERIMENT_NAME = "Section-aware chunking comparison"
EXPERIMENTAL_FILENAME_PREFIX = "section_aware__"

EXPERIMENT_RESULTS_PATH = (
    evaluator.EVALUATION_DIR / "retrieval_results_section_aware.json"
)


class SectionAwareExperimentError(Exception):
    """Raised when the section-aware experiment cannot be completed."""


def _normalize_newlines(text: str) -> str:
    """Normalize all supported newline sequences to LF."""

    return text.replace("\r\n", "\n").replace("\r", "\n")


def create_line_ending_variant(content: bytes) -> bytes:
    """
    Create different raw bytes while preserving the extracted text.

    The corpus documents contain multiple lines. We switch their line-ending
    convention rather than changing their actual textual content.
    """

    try:
        original_text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise SectionAwareExperimentError(
            "A benchmark corpus file is not valid UTF-8."
        ) from exc

    normalized_text = _normalize_newlines(original_text)

    if "\n" not in normalized_text:
        raise SectionAwareExperimentError(
            "Cannot create a line-ending variant for a single-line document."
        )

    # Use the opposite convention to ensure the resulting bytes differ.
    if b"\r\n" in content:
        variant_text = normalized_text
    else:
        variant_text = normalized_text.replace("\n", "\r\n")

    variant_content = variant_text.encode("utf-8")

    if hashlib.sha256(content).digest() == hashlib.sha256(
        variant_content
    ).digest():
        raise SectionAwareExperimentError(
            "The experimental copy has the same SHA-256 hash as the baseline."
        )

    if _normalize_newlines(
        variant_content.decode("utf-8-sig")
    ) != normalized_text:
        raise SectionAwareExperimentError(
            "The experimental copy does not preserve the original text."
        )

    return variant_content


def _load_corpus_content(filename: str) -> bytes:
    """Read one benchmark file without modifying the original corpus."""

    corpus_path = evaluator.CORPUS_DIR / filename

    try:
        return corpus_path.read_bytes()
    except OSError as exc:
        raise SectionAwareExperimentError(
            f"Unable to read benchmark document: {filename}"
        ) from exc


def _get_existing_experimental_document(
    filename: str,
    content_hash: str,
) -> UUID | None:
    """Find a previously ingested copy of this experimental document."""

    expected_filename = f"{EXPERIMENTAL_FILENAME_PREFIX}{filename}"

    with SessionLocal() as db:
        document = db.scalar(
            select(Document).where(Document.sha256 == content_hash)
        )

        if document is None:
            return None

        if document.filename != expected_filename:
            raise SectionAwareExperimentError(
                "An experimental document with the same content hash already "
                f"exists under an unexpected filename: {document.filename}"
            )

        if (
            document.status != "completed"
            or document.indexing_status != "completed"
        ):
            raise SectionAwareExperimentError(
                f"Experimental document {filename} exists but is not fully "
                "processed and indexed. Inspect its database record before "
                "retrying."
            )

        return document.id


def _ingest_experimental_document(
    filename: str,
    content: bytes,
) -> UUID:
    """Persist, section-chunk, and index one experimental document."""

    experimental_filename = f"{EXPERIMENTAL_FILENAME_PREFIX}{filename}"

    upload = UploadFile(
        filename=experimental_filename,
        file=BytesIO(content),
        headers=Headers({"content-type": "text/plain"}),
    )

    try:
        stored_file = asyncio.run(store_upload(upload))
    except Exception as exc:
        raise SectionAwareExperimentError(
            f"Unable to store experimental document {filename}."
        ) from exc

    with SessionLocal() as db:
        try:
            document = persist_document(
                db=db,
                original_filename=experimental_filename,
                stored_file=stored_file,
            )

            process_document(
                db=db,
                document_id=document.id,
                chunking_config=ChunkingConfig(
                    strategy="section_aware"
                ),
            )

            index_document(
                db=db,
                document_id=document.id,
            )

            db.refresh(document)

            if (
                document.status != "completed"
                or document.indexing_status != "completed"
            ):
                raise SectionAwareExperimentError(
                    f"Experimental document {filename} did not complete "
                    "processing and indexing."
                )

            return document.id

        except SectionAwareExperimentError:
            raise
        except Exception as exc:
            db.rollback()
            raise SectionAwareExperimentError(
                f"Failed to process or index experimental document {filename}. "
                "Inspect its database record and stored file before retrying."
            ) from exc


def get_or_create_section_aware_documents(
    corpus_files: list[str],
) -> dict[str, UUID]:
    """
    Create or reuse section-aware copies of the benchmark documents.

    The dictionary keys remain the original corpus filenames so the existing
    relevance-evidence definitions can be reused without modification.
    """

    document_ids: dict[str, UUID] = {}

    for filename in corpus_files:
        original_content = _load_corpus_content(filename)
        experimental_content = create_line_ending_variant(original_content)
        content_hash = hashlib.sha256(experimental_content).hexdigest()

        existing_id = _get_existing_experimental_document(
            filename=filename,
            content_hash=content_hash,
        )

        if existing_id is not None:
            document_ids[filename] = existing_id
            continue

        document_ids[filename] = _ingest_experimental_document(
            filename=filename,
            content=experimental_content,
        )

    if len(document_ids) != len(corpus_files):
        raise SectionAwareExperimentError(
            "Not all section-aware benchmark documents were registered."
        )

    return document_ids


def get_chunk_counts(
    document_ids: dict[str, UUID],
) -> dict[str, int]:
    """Return persisted chunk counts for each benchmark document."""

    counts: dict[str, int] = {}

    with SessionLocal() as db:
        for filename, document_id in document_ids.items():
            count = db.scalar(
                select(func.count(Chunk.id)).where(
                    Chunk.document_id == document_id
                )
            )
            counts[filename] = int(count or 0)

    return counts


def load_baseline_report(
    baseline_document_ids: dict[str, UUID],
) -> dict:
    """Load and verify the existing reliability report for comparison."""

    report_path = evaluator.RESULTS_PATH

    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SectionAwareExperimentError(
            "Unable to load the existing baseline reliability report."
        ) from exc

    if report.get("dataset_version") != "2.0":
        raise SectionAwareExperimentError(
            "The baseline report is not from retrieval dataset v2.0."
        )

    recorded_documents = report.get("corpus_documents", {})
    expected_documents = {
        filename: str(document_id)
        for filename, document_id in baseline_document_ids.items()
    }

    if recorded_documents != expected_documents:
        raise SectionAwareExperimentError(
            "The baseline report's document IDs do not match the currently "
            "registered benchmark documents. Re-run the baseline evaluation "
            "before comparing results."
        )

    if set(report.get("aggregate_metrics", {})) != set(evaluator.STRATEGIES):
        raise SectionAwareExperimentError(
            "The baseline report does not contain all expected retrieval "
            "strategies."
        )

    return report


def build_metric_comparison(
    baseline_aggregate: dict,
    experimental_aggregate: dict,
) -> dict:
    """Calculate section-aware minus baseline metric differences."""

    comparison = {}

    for strategy in evaluator.STRATEGIES:
        comparison[strategy] = {}

        for metric in evaluator.METRIC_NAMES:
            baseline_value = baseline_aggregate[strategy][metric]
            experimental_value = experimental_aggregate[strategy][metric]

            comparison[strategy][metric] = {
                "baseline": baseline_value,
                "section_aware": experimental_value,
                "delta": round(
                    experimental_value - baseline_value,
                    6,
                ),
            }

    return comparison


def run_section_aware_experiment() -> dict:
    """Run the section-aware evaluation and save a separate report."""

    dataset = evaluator.load_dataset()

    print("\nPreparing baseline benchmark documents...")
    baseline_document_ids = evaluator.get_or_upload_benchmark_documents(
        dataset["corpus_files"]
    )

    print("\nPreparing section-aware experimental documents...")
    experimental_document_ids = get_or_create_section_aware_documents(
        dataset["corpus_files"]
    )

    baseline_report = load_baseline_report(baseline_document_ids)

    baseline_chunk_counts = get_chunk_counts(baseline_document_ids)
    experimental_chunk_counts = get_chunk_counts(experimental_document_ids)

    print("\nChunk count comparison:")
    for filename in dataset["corpus_files"]:
        print(
            f"  {filename:38} "
            f"baseline={baseline_chunk_counts[filename]:3} "
            f"section-aware={experimental_chunk_counts[filename]:3}"
        )

    print("\nLoading cross-encoder model...")
    reranker_started_at = perf_counter()

    try:
        reranker = get_reranker()
    except RerankingError as exc:
        raise SectionAwareExperimentError(
            "Unable to initialize the cross-encoder for evaluation."
        ) from exc

    reranker_load_latency_ms = (
        perf_counter() - reranker_started_at
    ) * 1000

    print(
        "Cross-encoder initialized in "
        f"{reranker_load_latency_ms:.2f} ms"
    )

    print("\n" + "=" * 75)
    print("KNOWLEDGE INTELLIGENCE PLATFORM")
    print("SECTION-AWARE CHUNKING EXPERIMENT")
    print("=" * 75)

    print(f"\nDataset: {dataset['dataset_name']}")
    print(f"Dataset version: {dataset['version']}")
    print(f"Corpus documents: {len(experimental_document_ids)}")
    print(f"Evaluation queries: {len(dataset['queries'])}")
    print(f"Evaluation cutoff: {evaluator.TOP_K}")
    print(f"Warm-up runs: {evaluator.WARMUP_RUNS}")
    print(f"Measured trials: {evaluator.MEASURED_TRIALS}")

    query_results = []
    randomizer = Random(evaluator.RANDOM_SEED)

    with SessionLocal() as db:
        for index, query_item in enumerate(dataset["queries"], start=1):
            print(
                f"\nEvaluating query {index}/{len(dataset['queries'])}: "
                f"{query_item['id']}"
            )

            result = evaluator.evaluate_query(
                db=db,
                query_item=query_item,
                document_ids=experimental_document_ids,
                reranker=reranker,
                warmup_runs=evaluator.WARMUP_RUNS,
                measured_trials=evaluator.MEASURED_TRIALS,
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

    experimental_aggregate = evaluator.calculate_aggregate_metrics(
        query_results
    )

    comparison = build_metric_comparison(
        baseline_aggregate=baseline_report["aggregate_metrics"],
        experimental_aggregate=experimental_aggregate,
    )

    report = {
        "experiment": {
            "name": EXPERIMENT_NAME,
            "chunking_strategy": "section_aware",
            "baseline_chunking_strategy": "character",
            "embedding_model": evaluator.settings.embedding_model_name,
            "top_k": evaluator.TOP_K,
            "hybrid_candidate_limit": evaluator.HYBRID_CANDIDATE_LIMIT,
            "strategies": list(evaluator.STRATEGIES),
            "warmup_runs_per_strategy": evaluator.WARMUP_RUNS,
            "measured_trials_per_strategy": evaluator.MEASURED_TRIALS,
            "random_seed": evaluator.RANDOM_SEED,
            "purpose": (
                "Compare section-aware chunking against the existing "
                "character-based baseline using the same corpus text, "
                "queries, relevance evidence, retrieval strategies, "
                "and evaluation settings."
            ),
            "text_equivalence_method": (
                "Experimental copies use alternate newline conventions. "
                "Text is normalized to LF before equivalence validation; "
                "raw SHA-256 hashes must differ."
            ),
        },
        "dataset_name": dataset["dataset_name"],
        "dataset_version": dataset["version"],
        "baseline_report": str(evaluator.RESULTS_PATH),
        "corpus_documents": {
            filename: str(document_id)
            for filename, document_id in experimental_document_ids.items()
        },
        "baseline_corpus_documents": {
            filename: str(document_id)
            for filename, document_id in baseline_document_ids.items()
        },
        "corpus_document_count": len(experimental_document_ids),
        "baseline_chunk_counts": baseline_chunk_counts,
        "section_aware_chunk_counts": experimental_chunk_counts,
        "baseline_total_chunks": sum(baseline_chunk_counts.values()),
        "section_aware_total_chunks": sum(
            experimental_chunk_counts.values()
        ),
        "evaluation_cutoff": evaluator.TOP_K,
        "query_count": len(query_results),
        "reranker": {
            "model_name": evaluator.settings.reranker_model_name,
            "model_load_latency_ms": round(
                reranker_load_latency_ms,
                3,
            ),
        },
        "baseline_aggregate_metrics": baseline_report["aggregate_metrics"],
        "section_aware_aggregate_metrics": experimental_aggregate,
        "metric_comparison": comparison,
        "query_results": query_results,
    }

    EXPERIMENT_RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    EXPERIMENT_RESULTS_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 75)
    print("BASELINE VS SECTION-AWARE METRIC COMPARISON")
    print("=" * 75)

    for strategy, metrics in comparison.items():
        print(f"\n{strategy.upper()}")

        for metric, values in metrics.items():
            print(
                f"  {metric:16} "
                f"baseline={values['baseline']:.4f} "
                f"section-aware={values['section_aware']:.4f} "
                f"delta={values['delta']:+.4f}"
            )

    print("\n" + "=" * 75)
    print("SECTION-AWARE EXPERIMENT COMPLETED")
    print("=" * 75)
    print(f"Results saved to: {EXPERIMENT_RESULTS_PATH}")

    return report


if __name__ == "__main__":
    run_section_aware_experiment()