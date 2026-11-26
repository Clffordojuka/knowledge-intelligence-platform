
import json
from dataclasses import dataclass
from statistics import mean, median
from time import perf_counter
from uuid import UUID

from sqlalchemy import select

from app.core.config import settings
from app.database.models import Chunk, Document
from app.database.session import SessionLocal
from app.evaluation.retrieval_evaluator import (
    CORPUS_DIR,
    EVALUATION_DIR,
    QUERY_INSTRUCTION,
    STRATEGIES,
    load_dataset,
)
from app.retrieval.bm25_search import search_bm25
from app.retrieval.hybrid_search import search_hybrid_chunks
from app.retrieval.reranking import RerankingError, get_reranker
from app.retrieval.semantic_search import search_similar_chunks

SECTION_AWARE_REPORT_PATH = (
    EVALUATION_DIR / "retrieval_results_section_aware.json"
)
RESULTS_PATH = (
    EVALUATION_DIR / "retrieval_results_chunking_invariant.json"
)

CHUNK_CUTOFFS = (5, 10, 20)
CONTEXT_CHAR_BUDGETS = (1000, 2000, 4000)


class ChunkingInvariantEvaluationError(Exception):
    """Raised when chunking-invariant evaluation cannot be completed."""


@dataclass(frozen=True)
class SourceSpan:
    filename: str
    start: int
    end: int


@dataclass(frozen=True)
class IndexedChunk:
    chunk_id: UUID
    document_id: UUID
    filename: str
    chunk_index: int
    content: str
    normalized_content: str
    source_start: int
    source_end: int


@dataclass(frozen=True)
class EvidenceTarget:
    evidence_id: str
    filename: str
    phrase: str
    spans: tuple[SourceSpan, ...]


def normalize_text(text: str) -> str:
    """Normalize case and whitespace consistently across source and chunks."""

    return " ".join(text.casefold().split())


def _find_all_occurrences(text: str, phrase: str) -> list[tuple[int, int]]:
    """Return every occurrence of phrase in text as half-open intervals."""

    occurrences = []
    search_from = 0

    while True:
        start = text.find(phrase, search_from)

        if start < 0:
            break

        end = start + len(phrase)
        occurrences.append((start, end))
        search_from = start + 1

    return occurrences


def build_chunk_index(
    filename: str,
    source_text: str,
    chunks: list,
) -> dict[UUID, IndexedChunk]:
    """
    Map persisted chunks to positions in the normalized source document.

    Searching from the previous chunk's start plus one supports overlapping
    character chunks as well as non-overlapping section-aware chunks.
    """

    normalized_source = normalize_text(source_text)

    if not normalized_source:
        raise ChunkingInvariantEvaluationError(
            f"Source document is empty after normalization: {filename}"
        )

    ordered_chunks = sorted(chunks, key=lambda item: item.chunk_index)

    if not ordered_chunks:
        raise ChunkingInvariantEvaluationError(
            f"No persisted chunks found for {filename}."
        )

    indexed_chunks = {}
    search_from = 0

    for chunk in ordered_chunks:
        normalized_content = normalize_text(chunk.content)

        if not normalized_content:
            raise ChunkingInvariantEvaluationError(
                f"Chunk {chunk.id} has empty normalized content."
            )

        source_start = normalized_source.find(
            normalized_content,
            search_from,
        )

        if source_start < 0:
            raise ChunkingInvariantEvaluationError(
                "Could not map a persisted chunk back to its source text. "
                f"Document: {filename}; chunk index: {chunk.chunk_index}."
            )

        source_end = source_start + len(normalized_content)

        indexed_chunks[chunk.id] = IndexedChunk(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            filename=filename,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
            normalized_content=normalized_content,
            source_start=source_start,
            source_end=source_end,
        )

        # Advance by one rather than by the previous chunk's end because
        # character-based chunks can overlap.
        search_from = source_start + 1

    return indexed_chunks


def build_evidence_targets(
    query_item: dict,
    source_texts: dict[str, str],
) -> list[EvidenceTarget]:
    """Map dataset evidence phrases to source-document character spans."""

    targets = []

    for index, entry in enumerate(query_item["relevant_evidence"], start=1):
        filename = entry["filename"]
        phrase = entry["phrase"]

        if filename not in source_texts:
            raise ChunkingInvariantEvaluationError(
                f"Evidence references an unknown source document: {filename}"
            )

        normalized_phrase = normalize_text(phrase)
        normalized_source = normalize_text(source_texts[filename])

        if not normalized_phrase:
            raise ChunkingInvariantEvaluationError(
                f"Evidence phrase is empty for query {query_item['id']}."
            )

        occurrences = _find_all_occurrences(
            normalized_source,
            normalized_phrase,
        )

        if not occurrences:
            raise ChunkingInvariantEvaluationError(
                "Evidence phrase does not occur in its source document. "
                f"Query: {query_item['id']}; "
                f"document: {filename}; phrase: {phrase!r}."
            )

        targets.append(
            EvidenceTarget(
                evidence_id=f"e{index:03d}",
                filename=filename,
                phrase=phrase,
                spans=tuple(
                    SourceSpan(
                        filename=filename,
                        start=start,
                        end=end,
                    )
                    for start, end in occurrences
                ),
            )
        )

    if not targets:
        raise ChunkingInvariantEvaluationError(
            f"Query {query_item['id']} has no evidence targets."
        )

    return targets


def merge_source_spans(spans: list[SourceSpan]) -> list[SourceSpan]:
    """Merge overlapping or adjacent spans within the same source document."""

    grouped: dict[str, list[tuple[int, int]]] = {}

    for span in spans:
        if span.start < 0 or span.end <= span.start:
            raise ChunkingInvariantEvaluationError(
                "Invalid source span encountered."
            )

        grouped.setdefault(span.filename, []).append(
            (span.start, span.end)
        )

    merged = []

    for filename, intervals in grouped.items():
        intervals.sort()

        current_start, current_end = intervals[0]

        for start, end in intervals[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                merged.append(
                    SourceSpan(filename, current_start, current_end)
                )
                current_start, current_end = start, end

        merged.append(
            SourceSpan(filename, current_start, current_end)
        )

    return merged


def is_evidence_covered(
    target: EvidenceTarget,
    retrieved_spans: list[SourceSpan],
) -> bool:
    """Check whether any complete evidence occurrence has been retrieved."""

    merged_spans = merge_source_spans(retrieved_spans)

    for evidence_span in target.spans:
        for retrieved_span in merged_spans:
            if retrieved_span.filename != evidence_span.filename:
                continue

            if (
                retrieved_span.start <= evidence_span.start
                and retrieved_span.end >= evidence_span.end
            ):
                return True

    return False


def _context_spans(
    ranked_chunks: list[IndexedChunk],
    character_budget: int,
) -> tuple[list[SourceSpan], int]:
    """
    Simulate packing ranked chunk text into a normalized-character budget.

    Overlapping text is counted again because it consumes context space.
    Only the prefix that fits inside the budget contributes source coverage.
    """

    remaining = character_budget
    consumed = 0
    spans = []

    for chunk in ranked_chunks:
        if remaining <= 0:
            break

        chunk_length = len(chunk.normalized_content)
        included_length = min(chunk_length, remaining)

        if included_length <= 0:
            continue

        spans.append(
            SourceSpan(
                filename=chunk.filename,
                start=chunk.source_start,
                end=chunk.source_start + included_length,
            )
        )

        consumed += included_length
        remaining -= included_length

    return spans, consumed


def calculate_evidence_metrics(
    ranked_chunks: list[IndexedChunk],
    evidence_targets: list[EvidenceTarget],
) -> dict:
    """
    Calculate relevance metrics against source evidence, not chunk UUIDs.

    An evidence phrase can be covered by multiple retrieved chunks when
    its source span crosses a chunk boundary.
    """

    if not evidence_targets:
        raise ChunkingInvariantEvaluationError(
            "Evidence targets must not be empty."
        )

    retrieved_spans = []
    first_covering_rank: dict[str, int | None] = {
        target.evidence_id: None
        for target in evidence_targets
    }

    for rank, chunk in enumerate(ranked_chunks, start=1):
        retrieved_spans.append(
            SourceSpan(
                filename=chunk.filename,
                start=chunk.source_start,
                end=chunk.source_end,
            )
        )

        for target in evidence_targets:
            if first_covering_rank[target.evidence_id] is not None:
                continue

            if is_evidence_covered(target, retrieved_spans):
                first_covering_rank[target.evidence_id] = rank

    evidence_count = len(evidence_targets)

    metrics = {}

    for cutoff in CHUNK_CUTOFFS:
        covered_count = sum(
            rank is not None and rank <= cutoff
            for rank in first_covering_rank.values()
        )

        metrics[f"evidence_recall_at_{cutoff}"] = (
            covered_count / evidence_count
        )

    reciprocal_ranks = [
        1.0 / rank if rank is not None else 0.0
        for rank in first_covering_rank.values()
    ]

    metrics["evidence_mean_reciprocal_rank"] = mean(
        reciprocal_ranks
    )

    metrics["evidence_hit_rate_at_5"] = float(
        any(
            rank is not None and rank <= 5
            for rank in first_covering_rank.values()
        )
    )

    context_results = {}

    for budget in CONTEXT_CHAR_BUDGETS:
        budget_spans, consumed = _context_spans(
            ranked_chunks,
            budget,
        )

        covered_ids = [
            target.evidence_id
            for target in evidence_targets
            if is_evidence_covered(target, budget_spans)
        ]

        context_results[str(budget)] = {
            "character_budget": budget,
            "context_characters_used": consumed,
            "covered_evidence_count": len(covered_ids),
            "evidence_recall": len(covered_ids) / evidence_count,
            "covered_evidence_ids": covered_ids,
        }

    evidence_ranks = [
        {
            "evidence_id": target.evidence_id,
            "filename": target.filename,
            "phrase": target.phrase,
            "first_covering_rank": first_covering_rank[
                target.evidence_id
            ],
        }
        for target in evidence_targets
    ]

    return {
        "evidence_count": evidence_count,
        "metrics": metrics,
        "context_budgets": context_results,
        "evidence_ranks": evidence_ranks,
    }


def load_experiment_document_ids(
    corpus_files: list[str],
) -> dict[str, dict[str, UUID]]:
    """Load the existing baseline and section-aware document mappings."""

    try:
        report = json.loads(
            SECTION_AWARE_REPORT_PATH.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ChunkingInvariantEvaluationError(
            "Unable to load the existing section-aware experiment report."
        ) from exc

    experiment = report.get("experiment", {})

    if experiment.get("chunking_strategy") != "section_aware":
        raise ChunkingInvariantEvaluationError(
            "The experiment report does not describe section-aware chunking."
        )

    raw_mappings = {
        "character": report.get("baseline_corpus_documents"),
        "section_aware": report.get("corpus_documents"),
    }

    mappings = {}

    for variant, raw_mapping in raw_mappings.items():
        if not isinstance(raw_mapping, dict):
            raise ChunkingInvariantEvaluationError(
                f"Missing document mapping for {variant}."
            )

        if set(raw_mapping) != set(corpus_files):
            raise ChunkingInvariantEvaluationError(
                f"{variant} document mapping does not match the dataset corpus."
            )

        try:
            mappings[variant] = {
                filename: UUID(document_id)
                for filename, document_id in raw_mapping.items()
            }
        except (ValueError, TypeError) as exc:
            raise ChunkingInvariantEvaluationError(
                f"Invalid document UUID in {variant} mapping."
            ) from exc

    return mappings


def load_corpus_variant_indexes(
    db,
    corpus_files: list[str],
    document_ids: dict[str, UUID],
    variant: str,
) -> tuple[dict[UUID, IndexedChunk], dict[str, str], dict]:
    """Load and validate persisted documents and build source-span indexes."""

    if variant not in {"character", "section_aware"}:
        raise ChunkingInvariantEvaluationError(
            f"Unsupported corpus variant: {variant}"
        )

    all_document_ids = list(document_ids.values())

    documents = db.scalars(
        select(Document).where(Document.id.in_(all_document_ids))
    ).all()

    documents_by_id = {
        document.id: document
        for document in documents
    }

    missing_ids = set(all_document_ids) - set(documents_by_id)

    if missing_ids:
        raise ChunkingInvariantEvaluationError(
            f"{variant} corpus has missing database documents: "
            f"{sorted(str(value) for value in missing_ids)}"
        )

    expected_names = {}

    for filename, document_id in document_ids.items():
        document = documents_by_id[document_id]

        expected_filename = (
            f"section_aware__{filename}"
            if variant == "section_aware"
            else filename
        )

        if document.filename != expected_filename:
            raise ChunkingInvariantEvaluationError(
                f"Unexpected filename for {variant} document {document_id}: "
                f"expected {expected_filename!r}, got {document.filename!r}."
            )

        if document.status != "completed":
            raise ChunkingInvariantEvaluationError(
                f"{document.filename} is not fully processed."
            )

        if document.indexing_status != "completed":
            raise ChunkingInvariantEvaluationError(
                f"{document.filename} is not fully indexed."
            )

        expected_names[document_id] = filename

    chunks = db.scalars(
        select(Chunk)
        .where(Chunk.document_id.in_(all_document_ids))
        .order_by(Chunk.document_id, Chunk.chunk_index)
    ).all()

    chunks_by_document = {
        document_id: []
        for document_id in all_document_ids
    }

    for chunk in chunks:
        chunks_by_document[chunk.document_id].append(chunk)

    source_texts = {}
    chunk_index = {}
    chunk_counts = {}

    for filename in corpus_files:
        try:
            source_text = (CORPUS_DIR / filename).read_text(
                encoding="utf-8"
            )
        except OSError as exc:
            raise ChunkingInvariantEvaluationError(
                f"Unable to read source document: {filename}"
            ) from exc

        source_texts[filename] = source_text

        document_id = next(
            document_id
            for document_id, canonical_name in expected_names.items()
            if canonical_name == filename
        )

        document_chunks = chunks_by_document[document_id]

        indexed = build_chunk_index(
            filename=filename,
            source_text=source_text,
            chunks=document_chunks,
        )

        chunk_index.update(indexed)
        chunk_counts[filename] = len(indexed)

    return chunk_index, source_texts, {
        "document_count": len(corpus_files),
        "total_chunk_count": len(chunk_index),
        "chunk_counts_by_document": chunk_counts,
        "documents": {
            filename: str(document_id)
            for filename, document_id in document_ids.items()
        },
    }


def build_retrieval_operations(
    db,
    query: str,
    document_ids: list[UUID],
    retrieval_limit: int,
    reranker,
) -> dict:
    """Build the six existing retrieval strategies at the common limit."""

    return {
        "semantic": lambda: search_similar_chunks(
            query=query,
            limit=retrieval_limit,
            document_ids=document_ids,
        ),
        "semantic_instructed": lambda: search_similar_chunks(
            query=query,
            limit=retrieval_limit,
            document_ids=document_ids,
            query_instruction=QUERY_INSTRUCTION,
        ),
        "bm25": lambda: search_bm25(
            db=db,
            query=query,
            limit=retrieval_limit,
            document_ids=document_ids,
        ),
        "hybrid": lambda: search_hybrid_chunks(
            db=db,
            query=query,
            limit=retrieval_limit,
            candidate_limit=retrieval_limit,
            document_ids=document_ids,
        ),
        "hybrid_instructed": lambda: search_hybrid_chunks(
            db=db,
            query=query,
            limit=retrieval_limit,
            candidate_limit=retrieval_limit,
            document_ids=document_ids,
            query_instruction=QUERY_INSTRUCTION,
        ),
        "hybrid_reranked": lambda: search_hybrid_chunks(
            db=db,
            query=query,
            limit=retrieval_limit,
            candidate_limit=retrieval_limit,
            rerank=True,
            reranker=reranker,
            document_ids=document_ids,
        ),
    }


def evaluate_query(
    db,
    query_item: dict,
    document_ids: dict[str, UUID],
    chunk_index: dict[UUID, IndexedChunk],
    source_texts: dict[str, str],
    reranker,
    retrieval_limit: int,
) -> dict:
    """Evaluate all strategies using source evidence and two fair budgets."""

    evidence_targets = build_evidence_targets(
        query_item=query_item,
        source_texts=source_texts,
    )

    allowed_document_ids = set(document_ids.values())

    operations = build_retrieval_operations(
        db=db,
        query=query_item["query"],
        document_ids=list(document_ids.values()),
        retrieval_limit=retrieval_limit,
        reranker=reranker,
    )

    strategy_results = {}

    for strategy in STRATEGIES:
        started_at = perf_counter()
        results = operations[strategy]()
        latency_ms = (perf_counter() - started_at) * 1000

        ranked_chunks = []

        for result in results:
            if result.document_id not in allowed_document_ids:
                raise ChunkingInvariantEvaluationError(
                    f"{strategy} returned a result outside its corpus."
                )

            indexed_chunk = chunk_index.get(result.chunk_id)

            if indexed_chunk is None:
                raise ChunkingInvariantEvaluationError(
                    f"{strategy} returned an unknown chunk: "
                    f"{result.chunk_id}"
                )

            if indexed_chunk.document_id != result.document_id:
                raise ChunkingInvariantEvaluationError(
                    f"{strategy} returned inconsistent chunk metadata."
                )

            ranked_chunks.append(indexed_chunk)

        evaluation = calculate_evidence_metrics(
            ranked_chunks=ranked_chunks,
            evidence_targets=evidence_targets,
        )

        strategy_results[strategy] = {
            "retrieved_count": len(ranked_chunks),
            "latency_ms": round(latency_ms, 3),
            "retrieved_chunks": [
                {
                    "chunk_id": str(chunk.chunk_id),
                    "document_id": str(chunk.document_id),
                    "filename": chunk.filename,
                    "chunk_index": chunk.chunk_index,
                    "normalized_characters": len(
                        chunk.normalized_content
                    ),
                    "source_start": chunk.source_start,
                    "source_end": chunk.source_end,
                }
                for chunk in ranked_chunks
            ],
            "evidence_count": evaluation["evidence_count"],
            "metrics": evaluation["metrics"],
            "context_budgets": evaluation["context_budgets"],
            "evidence_ranks": evaluation["evidence_ranks"],
        }

    return {
        "query_id": query_item["id"],
        "query": query_item["query"],
        "topic": query_item.get("topic"),
        "evidence_count": len(evidence_targets),
        "strategies": strategy_results,
    }


def calculate_aggregate_metrics(query_results: list[dict]) -> dict:
    """Calculate macro-averaged evidence metrics and latency."""

    if not query_results:
        raise ChunkingInvariantEvaluationError(
            "Cannot aggregate an empty evaluation result."
        )

    aggregate = {}

    for strategy in STRATEGIES:
        metric_names = [
            f"evidence_recall_at_{cutoff}"
            for cutoff in CHUNK_CUTOFFS
        ]
        metric_names.extend(
            [
                "evidence_mean_reciprocal_rank",
                "evidence_hit_rate_at_5",
            ]
        )

        strategy_metrics = {
            metric: mean(
                result["strategies"][strategy]["metrics"][metric]
                for result in query_results
            )
            for metric in metric_names
        }

        for budget in CONTEXT_CHAR_BUDGETS:
            key = str(budget)

            strategy_metrics[
                f"evidence_recall_at_{budget}_characters"
            ] = mean(
                result["strategies"][strategy]["context_budgets"][key][
                    "evidence_recall"
                ]
                for result in query_results
            )

        latencies = [
            result["strategies"][strategy]["latency_ms"]
            for result in query_results
        ]

        strategy_metrics["mean_latency_ms"] = mean(latencies)
        strategy_metrics["median_latency_ms"] = median(latencies)

        aggregate[strategy] = {
            metric: round(value, 6)
            for metric, value in strategy_metrics.items()
        }

    return aggregate


def build_comparison(
    baseline_metrics: dict,
    section_aware_metrics: dict,
) -> dict:
    """Calculate section-aware minus character-based metric differences."""

    comparison = {}

    for strategy in STRATEGIES:
        comparison[strategy] = {}

        for metric, baseline_value in baseline_metrics[strategy].items():
            experimental_value = section_aware_metrics[strategy][metric]

            comparison[strategy][metric] = {
                "character": baseline_value,
                "section_aware": experimental_value,
                "delta": round(
                    experimental_value - baseline_value,
                    6,
                ),
            }

    return comparison


def run_chunking_invariant_evaluation() -> dict:
    """Run the new chunking-invariant comparison without re-ingestion."""

    dataset = load_dataset()

    document_mappings = load_experiment_document_ids(
        dataset["corpus_files"]
    )

    print("\nLoading cross-encoder model...")

    try:
        reranker = get_reranker()
    except RerankingError as exc:
        raise ChunkingInvariantEvaluationError(
            "Unable to initialize the cross-encoder."
        ) from exc

    corpus_variants = {}
    all_query_results = {}

    with SessionLocal() as db:
        for variant in ("character", "section_aware"):
            print(f"\nPreparing {variant} corpus indexes...")

            chunk_index, source_texts, corpus_metadata = (
                load_corpus_variant_indexes(
                    db=db,
                    corpus_files=dataset["corpus_files"],
                    document_ids=document_mappings[variant],
                    variant=variant,
                )
            )

            corpus_metadata["embedding_model"] = (
                settings.embedding_model_name
            )

            corpus_metadata["reranker_model"] = (
                settings.reranker_model_name
            )

            corpus_metadata["total_chunk_count"] = len(chunk_index)

            corpus_metadata["chunk_counts_by_document"] = {
                filename: sum(
                    1
                    for chunk in chunk_index.values()
                    if chunk.filename == filename
                )
                for filename in dataset["corpus_files"]
            }

            corpus_variants[variant] = corpus_metadata

            all_query_results[variant] = []

        retrieval_limit = max(
            metadata["total_chunk_count"]
            for metadata in corpus_variants.values()
        )

        print(f"\nCommon retrieval limit: {retrieval_limit}")
        print(f"Queries: {len(dataset['queries'])}")
        print(f"Strategies: {len(STRATEGIES)}")

        for variant in ("character", "section_aware"):
            print(f"\nEvaluating {variant} corpus...")

            chunk_index, source_texts, _ = load_corpus_variant_indexes(
                db=db,
                corpus_files=dataset["corpus_files"],
                document_ids=document_mappings[variant],
                variant=variant,
            )

            for index, query_item in enumerate(
                dataset["queries"],
                start=1,
            ):
                print(
                    f"  {variant}: query "
                    f"{index}/{len(dataset['queries'])} "
                    f"({query_item['id']})"
                )

                result = evaluate_query(
                    db=db,
                    query_item=query_item,
                    document_ids=document_mappings[variant],
                    chunk_index=chunk_index,
                    source_texts=source_texts,
                    reranker=reranker,
                    retrieval_limit=retrieval_limit,
                )

                all_query_results[variant].append(result)

    aggregate_metrics = {
        variant: calculate_aggregate_metrics(results)
        for variant, results in all_query_results.items()
    }

    report = {
        "experiment": {
            "name": "Chunking-invariant retrieval evaluation",
            "version": "1.0",
            "relevance_unit": "source_document_evidence_span",
            "chunk_identity_used_for_relevance": False,
            "source_normalization": (
                "Unicode casefolding and whitespace collapsing."
            ),
            "chunk_cutoffs": list(CHUNK_CUTOFFS),
            "context_character_budgets": list(
                CONTEXT_CHAR_BUDGETS
            ),
            "retrieval_limit": retrieval_limit,
            "strategies": list(STRATEGIES),
            "query_instruction": QUERY_INSTRUCTION,
            "latency_methodology": (
                "Single measured retrieval pass per query, corpus variant, "
                "and strategy. Model initialization is excluded. Latency "
                "is exploratory and is not a repeated-trial reliability "
                "measurement."
            ),
            "purpose": (
                "Compare character-based and section-aware chunking using "
                "source-level evidence spans, equal chunk cutoffs, and "
                "equal normalized-character context budgets."
            ),
        },
        "dataset_name": dataset["dataset_name"],
        "dataset_version": dataset["version"],
        "query_count": len(dataset["queries"]),
        "corpus_variants": corpus_variants,
        "aggregate_metrics": aggregate_metrics,
        "comparison": build_comparison(
            baseline_metrics=aggregate_metrics["character"],
            section_aware_metrics=aggregate_metrics["section_aware"],
        ),
        "query_results": all_query_results,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = RESULTS_PATH.with_suffix(".json.tmp")

    temporary_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    temporary_path.replace(RESULTS_PATH)

    print("\n" + "=" * 75)
    print("CHUNKING-INVARIANT RETRIEVAL COMPARISON")
    print("=" * 75)

    for strategy in STRATEGIES:
        print(f"\n{strategy.upper()}")

        for metric, values in report["comparison"][strategy].items():
            print(
                f"  {metric:42} "
                f"character={values['character']:.4f} "
                f"section-aware={values['section_aware']:.4f} "
                f"delta={values['delta']:+.4f}"
            )

    print("\n" + "=" * 75)
    print("CHUNKING-INVARIANT EVALUATION COMPLETED")
    print("=" * 75)
    print(f"Results saved to: {RESULTS_PATH}")

    return report


if __name__ == "__main__":
    run_chunking_invariant_evaluation()