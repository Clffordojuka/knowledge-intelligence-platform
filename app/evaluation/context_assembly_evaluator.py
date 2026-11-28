
import json
from statistics import mean, median
from time import perf_counter
from uuid import UUID

from app.core.config import settings
from app.database.session import SessionLocal
from app.evaluation import chunking_invariant_evaluator as base
from app.retrieval.context_assembly import (
    AssembledContext,
    assemble_expanded_contexts,
)
from app.retrieval.context_expansion import (
    ExpandedContext,
    expand_retrieved_context,
)
from app.retrieval.fusion import HybridSearchResult
from app.retrieval.reranking import RerankingError, get_reranker

RESULTS_PATH = (
    base.EVALUATION_DIR / "retrieval_results_context_assembly.json"
)

MAX_ASSEMBLED_CONTEXT_CHARS = 12_000
MAX_OVERLAP_CHARS = 1_000
EXPANSION_BEFORE = 1
EXPANSION_AFTER = 1


class ContextAssemblyEvaluationError(Exception):
    """Raised when context-assembly evaluation cannot be completed."""


def _to_hybrid_result(result) -> HybridSearchResult:
    """Adapt a retrieval result to the context-expansion interface."""

    if isinstance(result, HybridSearchResult):
        return result

    score = getattr(result, "score", 0.0)

    return HybridSearchResult(
        chunk_id=result.chunk_id,
        document_id=result.document_id,
        chunk_index=result.chunk_index,
        content=result.content,
        filename=result.filename,
        page_start=getattr(result, "page_start", None),
        page_end=getattr(result, "page_end", None),
        rrf_score=getattr(result, "rrf_score", 0.0),
        semantic_score=getattr(result, "semantic_score", score),
        bm25_score=getattr(result, "bm25_score", score),
        section_title=getattr(result, "section_title", None),
    )


def build_assembled_source_spans(
    assembled: AssembledContext,
    chunk_index: dict[UUID, base.IndexedChunk],
) -> list[base.SourceSpan]:
    """
    Map only included chunk text back to normalized source-document spans.

    Document labels and separators are deliberately excluded. The mapping
    accounts for the raw overlap removed from each chunk and for truncation.
    """

    spans = []

    for source in assembled.sources:
        if source.included_characters <= 0:
            continue

        indexed_chunk = chunk_index.get(source.chunk_id)

        if indexed_chunk is None:
            raise ContextAssemblyEvaluationError(
                f"Unknown assembled chunk: {source.chunk_id}"
            )

        if indexed_chunk.document_id != source.document_id:
            raise ContextAssemblyEvaluationError(
                "Assembled chunk document metadata is inconsistent."
            )

        raw_start = source.removed_overlap_characters
        raw_end = raw_start + source.included_characters

        if raw_start < 0 or raw_end > len(indexed_chunk.content):
            raise ContextAssemblyEvaluationError(
                f"Invalid included-text range for chunk {source.chunk_id}."
            )

        included_text = base.normalize_text(
            indexed_chunk.content[raw_start:raw_end]
        )

        if not included_text:
            continue

        normalized_prefix = base.normalize_text(
            indexed_chunk.content[:raw_start]
        )

        approximate_offset = len(normalized_prefix)

        normalized_chunk = indexed_chunk.normalized_content

        local_start = normalized_chunk.find(
            included_text,
            approximate_offset,
        )

        if local_start < 0:
            raise ContextAssemblyEvaluationError(
                "Could not map included assembled text to its normalized "
                f"chunk content: {source.chunk_id}."
            )

        local_end = local_start + len(included_text)

        if normalized_chunk[local_start:local_end] != included_text:
            raise ContextAssemblyEvaluationError(
                f"Normalized source mapping failed for {source.chunk_id}."
            )

        source_start = indexed_chunk.source_start + local_start
        source_end = indexed_chunk.source_start + local_end

        if source_end > indexed_chunk.source_end:
            raise ContextAssemblyEvaluationError(
                f"Mapped source range exceeds chunk boundaries: "
                f"{source.chunk_id}."
            )

        spans.append(
            base.SourceSpan(
                filename=indexed_chunk.filename,
                start=source_start,
                end=source_end,
            )
        )

    return base.merge_source_spans(spans)


def calculate_assembled_evidence_metrics(
    assembled: AssembledContext,
    chunk_index: dict[UUID, base.IndexedChunk],
    evidence_targets: list[base.EvidenceTarget],
) -> dict:
    """Measure evidence coverage in the actual assembled context."""

    if not evidence_targets:
        raise ContextAssemblyEvaluationError(
            "Evidence targets must not be empty."
        )

    source_spans = build_assembled_source_spans(
        assembled=assembled,
        chunk_index=chunk_index,
    )

    covered_ids = [
        target.evidence_id
        for target in evidence_targets
        if base.is_evidence_covered(target, source_spans)
    ]

    evidence_count = len(evidence_targets)

    evidence_ranks = [
        {
            "evidence_id": target.evidence_id,
            "filename": target.filename,
            "phrase": target.phrase,
            "covered": target.evidence_id in covered_ids,
        }
        for target in evidence_targets
    ]

    return {
        "evidence_count": evidence_count,
        "covered_evidence_count": len(covered_ids),
        "evidence_recall": len(covered_ids) / evidence_count,
        "covered_evidence_ids": covered_ids,
        "evidence_details": evidence_ranks,
        "source_span_count": len(source_spans),
    }


def _build_expanded_units(
    expanded_contexts: list[ExpandedContext],
    chunk_index: dict[UUID, base.IndexedChunk],
) -> list:
    """Convert expanded results into the existing context metric format."""

    from app.evaluation.context_expansion_evaluator import ContextUnit

    units = []

    for expanded in expanded_contexts:
        if not expanded.chunk_ids:
            raise ContextAssemblyEvaluationError(
                "Context expansion returned no chunk IDs."
            )

        indexed_chunks = []

        for chunk_id in expanded.chunk_ids:
            indexed_chunk = chunk_index.get(chunk_id)

            if indexed_chunk is None:
                raise ContextAssemblyEvaluationError(
                    f"Unknown expanded chunk: {chunk_id}"
                )

            indexed_chunks.append(indexed_chunk)

        source_spans = base.merge_source_spans(
            [
                base.SourceSpan(
                    filename=chunk.filename,
                    start=chunk.source_start,
                    end=chunk.source_end,
                )
                for chunk in indexed_chunks
            ]
        )

        if not source_spans:
            raise ContextAssemblyEvaluationError(
                "Expanded context could not be mapped to source text."
            )

        units.append(
            ContextUnit(
                chunk_id=expanded.retrieval_result.chunk_id,
                source_spans=tuple(source_spans),
                normalized_characters=len(
                    base.normalize_text(expanded.content)
                ),
                original_characters=len(
                    base.normalize_text(
                        expanded.retrieval_result.content
                    )
                ),
                included_chunk_count=len(expanded.chunk_ids),
            )
        )

    return units


def _evaluate_strategy(
    db,
    operation,
    strategy: str,
    allowed_document_ids: set[UUID],
    chunk_index: dict[UUID, base.IndexedChunk],
    evidence_targets: list[base.EvidenceTarget],
) -> dict:
    """Evaluate expansion and global context assembly for one strategy."""

    retrieval_started = perf_counter()
    results = operation()
    retrieval_latency_ms = (perf_counter() - retrieval_started) * 1000

    expanded_contexts = []
    expansion_started = perf_counter()

    for result in results:
        if result.document_id not in allowed_document_ids:
            raise ContextAssemblyEvaluationError(
                f"{strategy} returned a result outside its corpus."
            )

        indexed_chunk = chunk_index.get(result.chunk_id)

        if indexed_chunk is None:
            raise ContextAssemblyEvaluationError(
                f"{strategy} returned an unknown chunk: {result.chunk_id}"
            )

        if indexed_chunk.document_id != result.document_id:
            raise ContextAssemblyEvaluationError(
                "Retrieved chunk document metadata is inconsistent."
            )

        expanded = expand_retrieved_context(
            db=db,
            result=_to_hybrid_result(result),
            before=EXPANSION_BEFORE,
            after=EXPANSION_AFTER,
            max_context_chars=base.CONTEXT_CHAR_BUDGETS[-1],
        )

        expanded_contexts.append(expanded)

    expansion_latency_ms = (
        perf_counter() - expansion_started
    ) * 1000

    expanded_units = _build_expanded_units(
        expanded_contexts=expanded_contexts,
        chunk_index=chunk_index,
    )

    expanded_metrics = (
        __import__(
            "app.evaluation.context_expansion_evaluator",
            fromlist=["calculate_context_metrics"],
        ).calculate_context_metrics(
            units=expanded_units,
            evidence_targets=evidence_targets,
        )
    )

    assembly_results = {}
    assembly_latencies = []

    budgets = (
        *base.CONTEXT_CHAR_BUDGETS,
        MAX_ASSEMBLED_CONTEXT_CHARS,
    )

    for budget in budgets:
        assembly_started = perf_counter()

        assembled = assemble_expanded_contexts(
            db=db,
            expanded_contexts=expanded_contexts,
            max_context_chars=budget,
            max_overlap_chars=MAX_OVERLAP_CHARS,
        )

        assembly_latency_ms = (
            perf_counter() - assembly_started
        ) * 1000

        assembly_latencies.append(assembly_latency_ms)

        evidence_metrics = calculate_assembled_evidence_metrics(
            assembled=assembled,
            chunk_index=chunk_index,
            evidence_targets=evidence_targets,
        )

        content_utilization = (
            assembled.included_content_characters / budget
        )

        assembly_results[str(budget)] = {
            "character_budget": budget,
            "assembly_latency_ms": round(assembly_latency_ms, 3),
            "evidence": evidence_metrics,
            "context": {
                "actual_context_characters": len(assembled.content),
                "included_content_characters": (
                    assembled.included_content_characters
                ),
                "separator_and_label_characters": (
                    assembled.separator_and_label_characters
                ),
                "content_utilization": round(
                    content_utilization,
                    6,
                ),
                "total_source_characters": (
                    assembled.total_source_characters
                ),
                "duplicate_chunk_count": (
                    assembled.duplicate_chunk_count
                ),
                "removed_overlap_characters": (
                    assembled.removed_overlap_characters
                ),
                "unique_chunk_count": assembled.unique_chunk_count,
                "included_chunk_count": assembled.included_chunk_count,
                "truncated": assembled.truncated,
                "truncated_source_count": sum(
                    source.truncated
                    for source in assembled.sources
                ),
            },
        }

    return {
        "retrieved_count": len(results),
        "retrieval_latency_ms": round(retrieval_latency_ms, 3),
        "expansion_latency_ms": round(expansion_latency_ms, 3),
        "expanded_context": expanded_metrics,
        "assembly_mean_latency_ms": round(
            mean(assembly_latencies),
            3,
        ),
        "assembly_by_budget": assembly_results,
    }


def evaluate_query(
    db,
    query_item: dict,
    document_ids: dict[str, UUID],
    chunk_index: dict[UUID, base.IndexedChunk],
    source_texts: dict[str, str],
    reranker,
    retrieval_limit: int,
) -> dict:
    """Evaluate all strategies for one query."""

    evidence_targets = base.build_evidence_targets(
        query_item=query_item,
        source_texts=source_texts,
    )

    allowed_document_ids = set(document_ids.values())

    operations = base.build_retrieval_operations(
        db=db,
        query=query_item["query"],
        document_ids=list(document_ids.values()),
        retrieval_limit=retrieval_limit,
        reranker=reranker,
    )

    strategy_results = {}

    for strategy in base.STRATEGIES:
        print(f"    {strategy}")

        strategy_results[strategy] = _evaluate_strategy(
            db=db,
            operation=operations[strategy],
            strategy=strategy,
            allowed_document_ids=allowed_document_ids,
            chunk_index=chunk_index,
            evidence_targets=evidence_targets,
        )

    return {
        "query_id": query_item["id"],
        "query": query_item["query"],
        "topic": query_item.get("topic"),
        "evidence_count": len(evidence_targets),
        "strategies": strategy_results,
    }


def calculate_aggregate_metrics(query_results: list[dict]) -> dict:
    """Calculate macro-averaged evidence, assembly, and cost metrics."""

    if not query_results:
        raise ContextAssemblyEvaluationError(
            "Cannot aggregate an empty evaluation result."
        )

    aggregate = {}

    budgets = (
        *base.CONTEXT_CHAR_BUDGETS,
        MAX_ASSEMBLED_CONTEXT_CHARS,
    )

    for strategy in base.STRATEGIES:
        strategy_aggregate = {}

        expanded_metrics = [
            "evidence_recall_at_5",
            "evidence_recall_at_10",
            "evidence_recall_at_20",
            "evidence_mean_reciprocal_rank",
            "evidence_hit_rate_at_5",
        ]

        strategy_aggregate["expanded_context"] = {
            metric: round(
                mean(
                    result["strategies"][strategy][
                        "expanded_context"
                    ]["metrics"][metric]
                    for result in query_results
                ),
                6,
            )
            for metric in expanded_metrics
        }

        strategy_aggregate["assembly_by_budget"] = {}

        for budget in budgets:
            key = str(budget)

            def values(path):
                return [
                    result["strategies"][strategy][
                        "assembly_by_budget"
                    ][key][path]
                    for result in query_results
                ]

            recalls = [
                result["strategies"][strategy][
                    "assembly_by_budget"
                ][key]["evidence"]["evidence_recall"]
                for result in query_results
            ]

            context_metrics = {
                metric: mean(
                    result["strategies"][strategy][
                        "assembly_by_budget"
                    ][key]["context"][metric]
                    for result in query_results
                )
                for metric in (
                    "actual_context_characters",
                    "included_content_characters",
                    "separator_and_label_characters",
                    "content_utilization",
                    "total_source_characters",
                    "duplicate_chunk_count",
                    "removed_overlap_characters",
                    "unique_chunk_count",
                    "included_chunk_count",
                )
            }

            context_metrics["truncated_query_rate"] = mean(
                float(
                    result["strategies"][strategy][
                        "assembly_by_budget"
                    ][key]["context"]["truncated"]
                )
                for result in query_results
            )

            strategy_aggregate["assembly_by_budget"][key] = {
                "mean_evidence_recall": round(mean(recalls), 6),
                "mean_assembly_latency_ms": round(
                    mean(
                        result["strategies"][strategy][
                            "assembly_by_budget"
                        ][key]["assembly_latency_ms"]
                        for result in query_results
                    ),
                    3,
                ),
                "context": {
                    name: round(value, 3)
                    for name, value in context_metrics.items()
                },
            }

        retrieval_latencies = [
            result["strategies"][strategy]["retrieval_latency_ms"]
            for result in query_results
        ]

        expansion_latencies = [
            result["strategies"][strategy]["expansion_latency_ms"]
            for result in query_results
        ]

        strategy_aggregate["cost"] = {
            "mean_retrieval_latency_ms": round(
                mean(retrieval_latencies),
                3,
            ),
            "median_retrieval_latency_ms": round(
                median(retrieval_latencies),
                3,
            ),
            "mean_expansion_latency_ms": round(
                mean(expansion_latencies),
                3,
            ),
        }

        aggregate[strategy] = strategy_aggregate

    return aggregate


def run_context_assembly_evaluation() -> dict:
    """Run the isolated context-assembly experiment."""

    dataset = base.load_dataset()

    document_mappings = base.load_experiment_document_ids(
        dataset["corpus_files"]
    )

    print("\nLoading cross-encoder model...")

    try:
        reranker = get_reranker()
    except RerankingError as exc:
        raise ContextAssemblyEvaluationError(
            "Unable to initialize the cross-encoder."
        ) from exc

    loaded_variants = {}
    all_query_results = {}

    with SessionLocal() as db:
        for variant in ("character", "section_aware"):
            print(f"\nPreparing {variant} corpus...")

            chunk_index, source_texts, metadata = (
                base.load_corpus_variant_indexes(
                    db=db,
                    corpus_files=dataset["corpus_files"],
                    document_ids=document_mappings[variant],
                    variant=variant,
                )
            )

            metadata["embedding_model"] = settings.embedding_model_name
            metadata["reranker_model"] = settings.reranker_model_name

            loaded_variants[variant] = {
                "chunk_index": chunk_index,
                "source_texts": source_texts,
                "metadata": metadata,
            }

            all_query_results[variant] = []

        retrieval_limit = max(
            item["metadata"]["total_chunk_count"]
            for item in loaded_variants.values()
        )

        print(f"\nCommon retrieval limit: {retrieval_limit}")
        print(f"Queries: {len(dataset['queries'])}")
        print(f"Strategies: {len(base.STRATEGIES)}")

        for variant in ("character", "section_aware"):
            print(f"\nEvaluating {variant} corpus...")

            variant_data = loaded_variants[variant]

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
                    chunk_index=variant_data["chunk_index"],
                    source_texts=variant_data["source_texts"],
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
            "name": "Deduplicated context assembly evaluation",
            "version": "1.0",
            "relevance_unit": "source_document_evidence_span",
            "chunk_identity_used_for_relevance": False,
            "source_normalization": (
                "Unicode casefolding and whitespace collapsing."
            ),
            "conditions": {
                "expanded_context": (
                    "Retrieved chunks expanded with adjacent chunks "
                    "from the same document section."
                ),
                "assembled_context": (
                    "Expanded contexts deduplicated by chunk ID, exact "
                    "eligible overlap removed, and one global character "
                    "budget enforced."
                ),
            },
            "expansion_before": EXPANSION_BEFORE,
            "expansion_after": EXPANSION_AFTER,
            "max_expanded_context_characters": (
                base.CONTEXT_CHAR_BUDGETS[-1]
            ),
            "max_assembled_context_characters": (
                MAX_ASSEMBLED_CONTEXT_CHARS
            ),
            "max_overlap_characters": MAX_OVERLAP_CHARS,
            "context_character_budgets": [
                *base.CONTEXT_CHAR_BUDGETS,
                MAX_ASSEMBLED_CONTEXT_CHARS,
            ],
            "retrieval_limit": retrieval_limit,
            "strategies": list(base.STRATEGIES),
            "latency_methodology": (
                "One retrieval pass, one expansion pass, and one assembly "
                "pass per query, corpus variant, strategy, and budget. "
                "Model initialization is excluded. This is not a "
                "repeated-trial benchmark."
            ),
            "limitations": [
                "The benchmark labels relevant evidence phrases, not all "
                "irrelevant passages.",
                "Evidence recall does not measure answer correctness.",
                "Character budgets are not model-token budgets.",
                "Assembly is repeated independently at each budget so "
                "document labels and separators consume the actual budget.",
                "Latency is exploratory and is not a repeated-trial "
                "reliability measurement.",
            ],
        },
        "dataset_name": dataset["dataset_name"],
        "dataset_version": dataset["version"],
        "query_count": len(dataset["queries"]),
        "corpus_variants": {
            variant: data["metadata"]
            for variant, data in loaded_variants.items()
        },
        "aggregate_metrics": aggregate_metrics,
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
    print("CONTEXT ASSEMBLY EVALUATION")
    print("=" * 75)

    for variant in ("character", "section_aware"):
        print(f"\n{variant.upper()}")

        for strategy in base.STRATEGIES:
            values = aggregate_metrics[variant][strategy]

            print(f"\n  {strategy.upper()}")

            print(
                "    Expanded evidence MRR: "
                f"{values['expanded_context']['evidence_mean_reciprocal_rank']:.4f}"
            )

            for budget in base.CONTEXT_CHAR_BUDGETS:
                metrics = values["assembly_by_budget"][str(budget)]

                print(
                    f"    Assembled recall @{budget}: "
                    f"{metrics['mean_evidence_recall']:.4f}"
                )

                print(
                    f"      Duplicate chunks: "
                    f"{metrics['context']['duplicate_chunk_count']:.2f}"
                )

                print(
                    f"      Removed overlap chars: "
                    f"{metrics['context']['removed_overlap_characters']:.2f}"
                )

                print(
                    f"      Mean assembly latency: "
                    f"{metrics['mean_assembly_latency_ms']:.2f} ms"
                )

    print("\n" + "=" * 75)
    print("CONTEXT ASSEMBLY EVALUATION COMPLETED")
    print("=" * 75)
    print(f"Results saved to: {RESULTS_PATH}")

    return report


if __name__ == "__main__":
    run_context_assembly_evaluation()
