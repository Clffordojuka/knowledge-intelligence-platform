from dataclasses import dataclass
from math import isfinite
from uuid import UUID

from app.retrieval.bm25_search import BM25SearchResult
from app.retrieval.semantic_search import SearchResult


class FusionError(Exception):
    """Raised when retrieval rankings cannot be fused."""


@dataclass(frozen=True)
class HybridSearchResult:
    """A fused retrieval result with its source ranking scores."""

    chunk_id: UUID
    document_id: UUID
    chunk_index: int
    content: str
    filename: str | None
    page_start: int | None
    page_end: int | None
    rrf_score: float
    semantic_score: float | None
    bm25_score: float | None


def reciprocal_rank_fusion(
    semantic_results: list[SearchResult],
    bm25_results: list[BM25SearchResult],
    k: int = 60,
    limit: int = 10,
) -> list[HybridSearchResult]:
    """Combine semantic and BM25 rankings using RRF."""

    if not isinstance(semantic_results, list):
        raise ValueError("Semantic results must be a list.")

    if not isinstance(bm25_results, list):
        raise ValueError("BM25 results must be a list.")

    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("RRF k must be a positive integer.")

    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("Result limit must be a positive integer.")

    fused = {}

    rankings = [
        ("semantic", semantic_results),
        ("bm25", bm25_results),
    ]

    for retrieval_type, results in rankings:
        seen_chunk_ids = set()

        for rank, result in enumerate(results, start=1):
            if not isinstance(
                result,
                (SearchResult, BM25SearchResult),
            ):
                raise FusionError("Unsupported retrieval result type.")

            if result.chunk_id in seen_chunk_ids:
                raise FusionError("A retrieval ranking contains duplicate chunk IDs.")

            seen_chunk_ids.add(result.chunk_id)

            if not isfinite(float(result.score)):
                raise FusionError("Retrieval scores must be finite.")

            entry = fused.setdefault(
                result.chunk_id,
                {
                    "result": result,
                    "rrf_score": 0.0,
                    "semantic_score": None,
                    "bm25_score": None,
                },
            )

            existing = entry["result"]

            if (
                existing.document_id != result.document_id
                or existing.chunk_index != result.chunk_index
                or existing.content != result.content
            ):
                raise FusionError("Conflicting metadata for the same chunk ID.")

            entry["rrf_score"] += 1.0 / (k + rank)

            entry[f"{retrieval_type}_score"] = float(result.score)

    ranked_entries = sorted(
        fused.values(),
        key=lambda entry: (
            -entry["rrf_score"],
            str(entry["result"].chunk_id),
        ),
    )

    hybrid_results = []

    for entry in ranked_entries[:limit]:
        result = entry["result"]

        hybrid_results.append(
            HybridSearchResult(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                chunk_index=result.chunk_index,
                content=result.content,
                filename=result.filename,
                page_start=result.page_start,
                page_end=result.page_end,
                rrf_score=entry["rrf_score"],
                semantic_score=entry["semantic_score"],
                bm25_score=entry["bm25_score"],
            )
        )

    return hybrid_results
