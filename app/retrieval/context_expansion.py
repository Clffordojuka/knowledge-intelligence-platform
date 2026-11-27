
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select

from app.database.models import Chunk, Document
from app.retrieval.fusion import HybridSearchResult


class ContextExpansionError(Exception):
    """Raised when retrieved context cannot be expanded safely."""


@dataclass(frozen=True)
class ExpandedContext:
    """Expanded text and provenance for one retrieved result."""

    retrieval_result: HybridSearchResult
    content: str
    chunk_ids: tuple[UUID, ...]
    section_title: str | None

    @property
    def was_expanded(self) -> bool:
        return len(self.chunk_ids) > 1


def _validate_expansion_parameters(
    before: int,
    after: int,
    max_context_chars: int,
) -> None:
    """Validate expansion limits."""

    for name, value in (("before", before), ("after", after)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer.")

    if (
        isinstance(max_context_chars, bool)
        or not isinstance(max_context_chars, int)
        or max_context_chars <= 0
    ):
        raise ValueError("Maximum context characters must be a positive integer.")


def _merge_chunk_content(chunks: list[Chunk]) -> str:
    """Merge ordered chunks while removing overlap using character offsets."""

    if not chunks:
        return ""

    merged = chunks[0].content
    previous_end = chunks[0].char_end

    for chunk in chunks[1:]:
        content = chunk.content
        current_start = chunk.char_start

        if previous_end is None or current_start is None:
            merged = f"{merged}\n\n{content}"
            previous_end = chunk.char_end
            continue

        overlap = previous_end - current_start

        if overlap > 0:
            if overlap >= len(content):
                if chunk.char_end is not None:
                    previous_end = max(previous_end, chunk.char_end)
                continue

            content = content[overlap:]

        elif overlap < 0:
            merged = f"{merged}\n\n{content}"
            previous_end = chunk.char_end
            continue

        merged += content

        if chunk.char_end is not None:
            previous_end = chunk.char_end

    return merged


def _select_neighbor_chunks(
    chunks_by_index: dict[int, Chunk],
    original_index: int,
    before: int,
    after: int,
) -> set[int]:
    """Select only consecutive neighbors around the retrieved chunk."""

    selected_indexes = {original_index}

    for direction, distance in ((-1, before), (1, after)):
        for step in range(1, distance + 1):
            candidate_index = original_index + direction * step

            if candidate_index not in chunks_by_index:
                break

            selected_indexes.add(candidate_index)

    return selected_indexes


def _apply_context_budget(
    chunks_by_index: dict[int, Chunk],
    selected_indexes: set[int],
    original_index: int,
    max_context_chars: int,
) -> set[int]:
    """Retain the original chunk and add closest neighbors within budget."""

    original_chunk = chunks_by_index[original_index]
    selected = {original_index}

    remaining_chars = max_context_chars - len(original_chunk.content)

    if remaining_chars <= 0:
        return selected

    neighbor_indexes = sorted(
        selected_indexes - {original_index},
        key=lambda index: (
            abs(index - original_index),
            0 if index < original_index else 1,
        ),
    )

    for index in neighbor_indexes:
        candidate = chunks_by_index[index]

        # Reserve space for a paragraph separator. This is conservative
        # when character offsets later allow overlap removal.
        estimated_cost = len(candidate.content) + 2

        if estimated_cost <= remaining_chars:
            selected.add(index)
            remaining_chars -= estimated_cost

    return selected


def expand_retrieved_context(
    db,
    result: HybridSearchResult,
    *,
    before: int = 1,
    after: int = 1,
    max_context_chars: int = 4000,
) -> ExpandedContext:
    """Expand a retrieved chunk with adjacent chunks from the same section.

    The original retrieval result and its ranking scores are never modified.

    Expansion is restricted to:
    - The same document.
    - The same section title.
    - Consecutive chunk indexes.
    - Completed documents.

    The original chunk is always retained, even when it exceeds the
    configured context-character budget.
    """

    if not isinstance(result, HybridSearchResult):
        raise ValueError("Result must be a HybridSearchResult.")

    _validate_expansion_parameters(
        before=before,
        after=after,
        max_context_chars=max_context_chars,
    )

    lower_index = max(0, result.chunk_index - before)
    upper_index = result.chunk_index + after

    statement = (
        select(Chunk)
        .join(Document, Chunk.document_id == Document.id)
        .where(
            Chunk.document_id == result.document_id,
            Chunk.chunk_index >= lower_index,
            Chunk.chunk_index <= upper_index,
            Document.status == "completed",
        )
    )

    if result.section_title is None:
        statement = statement.where(Chunk.section_title.is_(None))
    else:
        statement = statement.where(
            Chunk.section_title == result.section_title
        )

    statement = statement.order_by(Chunk.chunk_index)

    candidate_chunks = list(db.scalars(statement).all())

    chunks_by_index = {
        chunk.chunk_index: chunk for chunk in candidate_chunks
    }

    original_chunk = chunks_by_index.get(result.chunk_index)

    if original_chunk is None or original_chunk.id != result.chunk_id:
        raise ContextExpansionError(
            "The retrieved chunk is no longer available in its document section."
        )

    selected_indexes = _select_neighbor_chunks(
        chunks_by_index=chunks_by_index,
        original_index=result.chunk_index,
        before=before,
        after=after,
    )

    selected_indexes = _apply_context_budget(
        chunks_by_index=chunks_by_index,
        selected_indexes=selected_indexes,
        original_index=result.chunk_index,
        max_context_chars=max_context_chars,
    )

    selected_chunks = [
        chunks_by_index[index]
        for index in sorted(selected_indexes)
    ]

    expanded_content = _merge_chunk_content(selected_chunks)

    return ExpandedContext(
        retrieval_result=result,
        content=expanded_content,
        chunk_ids=tuple(chunk.id for chunk in selected_chunks),
        section_title=result.section_title,
    )
