
import json
from dataclasses import dataclass
from statistics import mean, median
from time import perf_counter
from uuid import UUID

from app.core.config import settings
from app.database.session import SessionLocal
from app.evaluation import chunking_invariant_evaluator as base
from app.retrieval.context_expansion import (
    ExpandedContext,
    expand_retrieved_context,
)
from app.retrieval.fusion import HybridSearchResult
from app.retrieval.reranking import RerankingError, get_reranker

RESULTS_PATH = (
    base.EVALUATION_DIR / "retrieval_results_context_expansion.json"
)

EXPANSION_BEFORE = 1
EXPANSION_AFTER = 1
MAX_EXPANDED_CONTEXT_CHARS = 4000


class ContextExpansionEvaluationError(Exception):
    """Raised when context-expansion evaluation cannot be completed."""


@dataclass(frozen=True)
class ContextUnit:
    """One ranked context item and its source-document provenance."""

    chunk_id: UUID
    source_spans: tuple[base.SourceSpan, ...]
    normalized_characters: int
    original_characters: int
    included_chunk_count: int


def _to_hybrid_result(result) -> HybridSearchResult:
    """Adapt a retrieval result to the common context-expansion interface."""

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


def _build_baseline_unit(indexed_chunk) -> ContextUnit:
    """Represent one original retrieved chunk as a context item."""

    return ContextUnit(
        chunk_id=indexed_chunk.chunk_id,
        source_spans=(
            base.SourceSpan(
                filename=indexed_chunk.filename,
                start=indexed_chunk.source_start,
                end=indexed_chunk.source_end,
            ),
        ),
        normalized_characters=len(indexed_chunk.normalized_content),
        original_characters=len(indexed_chunk.normalized_content),
        included_chunk_count=1,
    )


def _build_expanded_unit(
    expanded: ExpandedContext,
    chunk_index: dict[UUID, base.IndexedChunk],
) -> ContextUnit:
    """Map expanded chunk IDs to source spans for evidence evaluation."""

    if not expanded.chunk_ids:
        raise ContextExpansionEvaluationError(
            "Context expansion returned no chunk IDs."
        )

    indexed_chunks = []

    for chunk_id in expanded.chunk_ids:
        indexed_chunk = chunk_index.get(chunk_id)

        if indexed_chunk is None:
            raise ContextExpansionEvaluationError(
                f"Expanded context contains an unknown chunk: {chunk_id}"
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
        raise ContextExpansionEvaluationError(
            "Expanded context could not be mapped to source text."
        )

    original_chunk = expanded.retrieval_result

    return ContextUnit(
        chunk_id=original_chunk.chunk_id,
        source_spans=tuple(source_spans),
        normalized_characters=len(base.normalize_text(expanded.content)),
        original_characters=len(
            base.normalize_text(original_chunk.content)
        ),
        included_chunk_count=len(expanded.chunk_ids),
    )


def _take_prefix_spans(
    spans: tuple[base.SourceSpan, ...],
    character_count: int,
) -> list[base.SourceSpan]:
    """Map a normalized context prefix to its source-document spans."""

    if character_count <= 0:
        return []

    remaining = character_count
    selected = []

    for index, span in enumerate(spans):
        if remaining <= 0:
            break

        # Disjoint source spans are separated by one normalized space
        # in the merged context representation.
        if index > 0:
            remaining -= 1

            if remaining <= 0:
                break

        span_length = span.end - span.start
        included_length = min(span_length, remaining)

        if included_length > 0:
            selected.append(
                base.SourceSpan(
                    filename=span.filename,
                    start=span.start,
                    end=span.start + included_length,
                )
            )
            remaining -= included_length

        if included_length < span_length:
            break

    return selected


def _pack_context_units(
    units: list[ContextUnit],
    character_budget: int,
) -> tuple[list[base.SourceSpan], int]:
    """Pack ranked context items into a normalized-character budget."""

    if character_budget <= 0:
        raise ValueError("Character budget must be positive.")

    remaining = character_budget
    consumed = 0
    spans = []

    for unit in units:
        if remaining <= 0:
            break

        included_characters = min(
            unit.normalized_characters,
            remaining,
        )

        if included_characters <= 0:
            continue

        spans.extend(
            _take_prefix_spans(
                unit.source_spans,
                included_characters,
            )
        )

        consumed += included_characters
        remaining -= included_characters

    return spans, consumed


def calculate_context_metrics(
    units: list[ContextUnit],
    evidence_targets: list[base.EvidenceTarget],
) -> dict:
    """Calculate evidence coverage for a ranked sequence of context items."""

    if not evidence_targets:
        raise ContextExpansionEvaluationError(
            "Evidence targets must not be empty."
        )

    first_covering_rank = {
        target.evidence_id: None
        for target in evidence_targets
    }

    accumulated_spans = []

    for rank, unit in enumerate(units, start=1):
        accumulated_spans.extend(unit.source_spans)

        for target in evidence_targets:
            if first_covering_rank[target.evidence_id] is not None:
                continue

            if base.is_evidence_covered(target, accumulated_spans):
                first_covering_rank[target.evidence_id] = rank

    evidence_count = len(evidence_targets)

    metrics = {}

    for cutoff in base.CHUNK_CUTOFFS:
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

    context_budgets = {}

    for budget in base.CONTEXT_CHAR_BUDGETS:
        budget_spans, consumed = _pack_context_units(
            units=units,
            character_budget=budget,
        )

        covered_ids = [
            target.evidence_id
            for target in evidence_targets
            if base.is_evidence_covered(target, budget_spans)
        ]

        context_budgets[str(budget)] = {
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
        "context_budgets": context_budgets,
        "evidence_ranks": evidence_ranks,
    }


def _evaluate_strategy(
    db,
    operation,
    strategy: str,
    allowed_document_ids: set[UUID],
    chunk_index: dict[UUID, base.IndexedChunk],
    evidence_targets: list[base.EvidenceTarget],
) -> dict:
    """Evaluate one retrieval ranking in baseline and expanded conditions."""

    retrieval_started = perf_counter()
    results = operation()
    retrieval_latency_ms = (
        perf_counter() - retrieval_started
    ) * 1000

    baseline_units = []
    expanded_units = []
    expansion_details = []

    expansion_started = perf_counter()

    for result in results:
        if result.document_id not in allowed_document_ids:
            raise ContextExpansionEvaluationError(
                f"{strategy} returned a result outside its corpus."
            )

        indexed_chunk = chunk_index.get(result.chunk_id)

        if indexed_chunk is None:
            raise ContextExpansionEvaluationError(
                f"{strategy} returned an unknown chunk: {result.chunk_id}"
            )

        if indexed_chunk.document_id != result.document_id:
            raise ContextExpansionEvaluationError(
                f"{strategy} returned inconsistent chunk metadata."
            )

        baseline_units.append(
            _build_baseline_unit(indexed_chunk)
        )

        expanded = expand_retrieved_context(
            db=db,
            result=_to_hybrid_result(result),
            before=EXPANSION_BEFORE,
            after=EXPANSION_AFTER,
            max_context_chars=MAX_EXPANDED_CONTEXT_CHARS,
        )

        expanded_unit = _build_expanded_unit(
            expanded=expanded,
            chunk_index=chunk_index,
        )

        expanded_units.append(expanded_unit)

        expansion_details.append(
            {
                "chunk_id": str(result.chunk_id),
                "original_characters": expanded_unit.original_characters,
                "expanded_characters": (
                    expanded_unit.normalized_characters
                ),
                "added_characters": max(
                    0,
                    expanded_unit.normalized_characters
                    - expanded_unit.original_characters,
                ),
                "included_chunk_count": (
                    expanded_unit.included_chunk_count
                ),
                "added_chunk_count": max(
                    0,
                    expanded_unit.included_chunk_count - 1,
                ),
                "expanded_chunk_ids": [
                    str(chunk_id)
                    for chunk_id in expanded.chunk_ids
                ],
            }
        )

    expansion_latency_ms = (
        perf_counter() - expansion_started
    ) * 1000

    baseline_evaluation = calculate_context_metrics(
        units=baseline_units,
        evidence_targets=evidence_targets,
    )

    expanded_evaluation = calculate_context_metrics(
        units=expanded_units,
        evidence_targets=evidence_targets,
    )

    total_added_characters = sum(
        item["added_characters"]
        for item in expansion_details
    )

    total_added_chunks = sum(
        item["added_chunk_count"]
        for item in expansion_details
    )

    return {
        "retrieved_count": len(results),
        "retrieval_latency_ms": round(retrieval_latency_ms, 3),
        "expansion_latency_ms": round(expansion_latency_ms, 3),
        "baseline": baseline_evaluation,
        "expanded": expanded_evaluation,
        "expansion_summary": {
            "expanded_result_count": sum(
                item["included_chunk_count"] > 1
                for item in expansion_details
            ),
            "total_added_chunks": total_added_chunks,
            "total_added_characters": total_added_characters,
            "mean_added_characters_per_result": (
                total_added_characters / len(results)
                if results else 0.0
            ),
            "mean_included_chunks_per_result": (
                mean(
                    item["included_chunk_count"]
                    for item in expansion_details
                )
                if expansion_details else 0.0
            ),
        },
        "expansion_details": expansion_details,
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
    """Evaluate all retrieval strategies with and without context expansion."""

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
    """Aggregate baseline, expanded, and expansion-cost metrics."""

    if not query_results:
        raise ContextExpansionEvaluationError(
            "Cannot aggregate an empty evaluation result."
        )

    aggregate = {}

    for strategy in base.STRATEGIES:
        strategy_aggregate = {}

        for condition in ("baseline", "expanded"):
            metric_names = [
                *[
                    f"evidence_recall_at_{cutoff}"
                    for cutoff in base.CHUNK_CUTOFFS
                ],
                "evidence_mean_reciprocal_rank",
                "evidence_hit_rate_at_5",
            ]

            metrics = {
                metric: mean(
                    result["strategies"][strategy][condition][
                        "metrics"
                    ][metric]
                    for result in query_results
                )
                for metric in metric_names
            }

            for budget in base.CONTEXT_CHAR_BUDGETS:
                key = str(budget)

                metrics[
                    f"evidence_recall_at_{budget}_characters"
                ] = mean(
                    result["strategies"][strategy][condition][
                        "context_budgets"
                    ][key]["evidence_recall"]
                    for result in query_results
                )

            strategy_aggregate[condition] = {
                metric: round(value, 6)
                for metric, value in metrics.items()
            }

        expansion_latencies = [
            result["strategies"][strategy]["expansion_latency_ms"]
            for result in query_results
        ]

        retrieval_latencies = [
            result["strategies"][strategy]["retrieval_latency_ms"]
            for result in query_results
        ]

        added_characters = [
            result["strategies"][strategy]["expansion_summary"][
                "total_added_characters"
            ]
            for result in query_results
        ]

        added_chunks = [
            result["strategies"][strategy]["expansion_summary"][
                "total_added_chunks"
            ]
            for result in query_results
        ]

        strategy_aggregate["cost"] = {
            "mean_retrieval_latency_ms": round(
                mean(retrieval_latencies), 3
            ),
            "median_retrieval_latency_ms": round(
                median(retrieval_latencies), 3
            ),
            "mean_expansion_latency_ms": round(
                mean(expansion_latencies), 3
            ),
            "median_expansion_latency_ms": round(
                median(expansion_latencies), 3
            ),
            "mean_added_characters_per_query": round(
                mean(added_characters), 3
            ),
            "mean_added_chunks_per_query": round(
                mean(added_chunks), 3
            ),
        }

        aggregate[strategy] = strategy_aggregate

    return aggregate


def build_comparison(aggregate_metrics: dict) -> dict:
    """Calculate expanded-minus-baseline metric differences."""

    comparison = {}

    for strategy in base.STRATEGIES:
        comparison[strategy] = {}

        baseline = aggregate_metrics[strategy]["baseline"]
        expanded = aggregate_metrics[strategy]["expanded"]

        for metric, baseline_value in baseline.items():
            expanded_value = expanded[metric]

            comparison[strategy][metric] = {
                "baseline": baseline_value,
                "expanded": expanded_value,
                "delta": round(
                    expanded_value - baseline_value,
                    6,
                ),
            }

    return comparison


def run_context_expansion_evaluation() -> dict:
    """Run the isolated context-expansion experiment."""

    dataset = base.load_dataset()

    document_mappings = base.load_experiment_document_ids(
        dataset["corpus_files"]
    )

    print("\nLoading cross-encoder model...")

    try:
        reranker = get_reranker()
    except RerankingError as exc:
        raise ContextExpansionEvaluationError(
            "Unable to initialize the cross-encoder."
        ) from exc

    loaded_variants = {}
    all_query_results = {}

    with SessionLocal() as db:
        for variant in ("character", "section_aware"):
            print(f"\nPreparing {variant} corpus...")

            chunk_index, source_texts, corpus_metadata = (
                base.load_corpus_variant_indexes(
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

            loaded_variants[variant] = {
                "chunk_index": chunk_index,
                "source_texts": source_texts,
                "metadata": corpus_metadata,
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
            "name": "Context expansion evaluation",
            "version": "1.0",
            "relevance_unit": "source_document_evidence_span",
            "chunk_identity_used_for_relevance": False,
            "baseline_condition": (
                "Original ranked retrieval chunks without expansion."
            ),
            "expanded_condition": (
                "The same ranked chunks expanded with adjacent chunks "
                "from the same document section."
            ),
            "expansion_before": EXPANSION_BEFORE,
            "expansion_after": EXPANSION_AFTER,
            "max_expanded_context_characters": (
                MAX_EXPANDED_CONTEXT_CHARS
            ),
            "context_character_budgets": list(
                base.CONTEXT_CHAR_BUDGETS
            ),
            "chunk_cutoffs": list(base.CHUNK_CUTOFFS),
            "retrieval_limit": retrieval_limit,
            "strategies": list(base.STRATEGIES),
            "latency_methodology": (
                "One retrieval pass and one expansion pass per query, "
                "corpus variant, and strategy. Retrieval and expansion "
                "latencies are reported separately. Model initialization "
                "is excluded. This is not a repeated-trial benchmark."
            ),
            "limitations": [
                "The dataset labels relevant evidence phrases, not all "
                "irrelevant passages.",
                "Added context size is measured, but irrelevant-content "
                "rate cannot be established without additional judgments.",
                "Context budgets use normalized characters, not model tokens.",
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
        "comparison": {
            variant: build_comparison(metrics)
            for variant, metrics in aggregate_metrics.items()
        },
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
    print("CONTEXT EXPANSION EVALUATION")
    print("=" * 75)

    for variant in ("character", "section_aware"):
        print(f"\n{variant.upper()}")

        for strategy in base.STRATEGIES:
            values = aggregate_metrics[variant][strategy]

            baseline = values["baseline"]
            expanded = values["expanded"]
            cost = values["cost"]

            print(f"\n  {strategy.upper()}")
            print(
                "    Evidence recall @ 5: "
                f"{baseline['evidence_recall_at_5']:.4f} -> "
                f"{expanded['evidence_recall_at_5']:.4f}"
            )
            print(
                "    Evidence MRR:        "
                f"{baseline['evidence_mean_reciprocal_rank']:.4f} -> "
                f"{expanded['evidence_mean_reciprocal_rank']:.4f}"
            )
            print(
                "    Recall @ 1000 chars: "
                f"{baseline['evidence_recall_at_1000_characters']:.4f} -> "
                f"{expanded['evidence_recall_at_1000_characters']:.4f}"
            )
            print(
                "    Added chars/query:   "
                f"{cost['mean_added_characters_per_query']:.1f}"
            )
            print(
                "    Expansion latency:   "
                f"{cost['mean_expansion_latency_ms']:.2f} ms"
            )

    print("\n" + "=" * 75)
    print("CONTEXT EXPANSION EVALUATION COMPLETED")
    print("=" * 75)
    print(f"Results saved to: {RESULTS_PATH}")

    return report


if __name__ == "__main__":
    run_context_expansion_evaluation()
