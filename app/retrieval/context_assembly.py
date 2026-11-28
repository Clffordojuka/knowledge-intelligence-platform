
from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.models import Chunk, Document
from app.retrieval.context_expansion import ExpandedContext


class ContextAssemblyError(Exception):
    """Raised when retrieved context cannot be assembled safely."""


@dataclass(frozen=True)
class ContextSource:
    """Provenance for a chunk included in the assembled context."""

    chunk_id: UUID
    document_id: UUID
    filename: str
    section_title: str | None
    chunk_index: int
    page_start: int | None
    page_end: int | None
    retrieval_ranks: tuple[int, ...]
    primary_retrieval_ranks: tuple[int, ...]
    included_characters: int
    removed_overlap_characters: int
    truncated: bool


@dataclass(frozen=True)
class AssembledContext:
    """Generation-ready context and its source provenance."""

    content: str
    sources: tuple[ContextSource, ...]
    max_context_chars: int
    total_source_characters: int
    included_content_characters: int
    separator_and_label_characters: int
    duplicate_chunk_count: int
    removed_overlap_characters: int
    unique_chunk_count: int
    included_chunk_count: int
    truncated: bool


@dataclass(frozen=True)
class _ChunkRecord:
    """Minimal persisted chunk data required for assembly."""

    chunk_id: UUID
    document_id: UUID
    filename: str
    section_title: str | None
    chunk_index: int
    content: str
    page_start: int | None
    page_end: int | None


def _validate_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _longest_suffix_prefix_overlap(
    previous_text: str,
    current_text: str,
    max_overlap_chars: int,
) -> int:
    """Return the longest exact suffix/prefix overlap between two chunks."""

    maximum = min(
        len(previous_text),
        len(current_text),
        max_overlap_chars,
    )

    for overlap_size in range(maximum, 0, -1):
        if previous_text[-overlap_size:] == current_text[:overlap_size]:
            return overlap_size

    return 0


def _build_context_header(chunk: _ChunkRecord) -> str:
    """Build a compact document and section label."""

    parts = [f"Document: {chunk.filename}"]

    if chunk.section_title and chunk.section_title.strip():
        parts.append(f"Section: {chunk.section_title.strip()}")

    return "\n".join(parts)


def _load_chunks(
    db: Session,
    chunk_ids: Sequence[UUID],
) -> dict[UUID, _ChunkRecord]:
    """Load all requested chunks and their document filenames in one query."""

    if not chunk_ids:
        return {}

    statement = (
        select(Chunk, Document.filename)
        .join(Document, Chunk.document_id == Document.id)
        .where(Chunk.id.in_(chunk_ids))
    )

    rows = db.execute(statement).all()

    records: dict[UUID, _ChunkRecord] = {}

    for chunk, filename in rows:
        records[chunk.id] = _ChunkRecord(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            filename=filename,
            section_title=chunk.section_title,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
        )

    missing_ids = set(chunk_ids) - set(records)

    if missing_ids:
        missing = ", ".join(str(chunk_id) for chunk_id in sorted(missing_ids))
        raise ContextAssemblyError(
            f"Some expanded chunks no longer exist in the database: {missing}"
        )

    return records


def assemble_expanded_contexts(
    db: Session,
    expanded_contexts: Sequence[ExpandedContext],
    *,
    max_context_chars: int = 12_000,
    max_overlap_chars: int = 1_000,
) -> AssembledContext:
    """
    Assemble expanded retrieval results into one deduplicated context.

    Contexts are processed in retrieval-result order. Within each expanded
    result, chunks retain the order supplied by the context-expansion service.

    Duplicate chunk IDs are included only once. Exact textual overlap is
    removed only between consecutive chunks belonging to the same document
    and section.

    The global character budget includes document/section labels and
    separators. A final chunk may be truncated to fit the remaining budget.
    """

    _validate_positive_integer(max_context_chars, "max_context_chars")
    _validate_positive_integer(max_overlap_chars, "max_overlap_chars")

    if not expanded_contexts:
        return AssembledContext(
            content="",
            sources=(),
            max_context_chars=max_context_chars,
            total_source_characters=0,
            included_content_characters=0,
            separator_and_label_characters=0,
            duplicate_chunk_count=0,
            removed_overlap_characters=0,
            unique_chunk_count=0,
            included_chunk_count=0,
            truncated=False,
        )

    ordered_chunk_ids: list[UUID] = []
    seen_chunk_ids: set[UUID] = set()
    retrieval_ranks: dict[UUID, list[int]] = {}
    primary_ranks: dict[UUID, list[int]] = {}
    total_chunk_references = 0

    for rank, expanded in enumerate(expanded_contexts, start=1):
        primary_chunk_id = expanded.retrieval_result.chunk_id

        for chunk_id in expanded.chunk_ids:
            total_chunk_references += 1

            if chunk_id not in retrieval_ranks:
                retrieval_ranks[chunk_id] = []
            retrieval_ranks[chunk_id].append(rank)

            if chunk_id == primary_chunk_id:
                if chunk_id not in primary_ranks:
                    primary_ranks[chunk_id] = []
                primary_ranks[chunk_id].append(rank)

            if chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                ordered_chunk_ids.append(chunk_id)

    chunk_records = _load_chunks(db, ordered_chunk_ids)

    total_source_characters = sum(
        len(chunk_records[chunk_id].content)
        for chunk_id in ordered_chunk_ids
    )

    parts: list[str] = []
    sources: list[ContextSource] = []
    used_characters = 0
    included_content_characters = 0
    removed_overlap_characters = 0
    previous_chunk: _ChunkRecord | None = None
    previous_group: tuple[UUID, str | None] | None = None
    truncated = False

    for position, chunk_id in enumerate(ordered_chunk_ids):
        chunk = chunk_records[chunk_id]
        current_group = (chunk.document_id, chunk.section_title)

        if current_group != previous_group:
            header = _build_context_header(chunk)
            separator = "\n\n" if parts else ""
            required = len(separator) + len(header)
            remaining = max_context_chars - used_characters

            if required > remaining:
                truncated = True
                break

            parts.append(f"{separator}{header}")
            used_characters += required
            previous_group = current_group

        overlap = 0

        if (
            previous_chunk is not None
            and previous_chunk.document_id == chunk.document_id
            and previous_chunk.section_title == chunk.section_title
            and previous_chunk.chunk_index + 1 == chunk.chunk_index
        ):
            overlap = _longest_suffix_prefix_overlap(
                previous_chunk.content,
                chunk.content,
                max_overlap_chars,
            )

        new_content = chunk.content[overlap:]
        separator = "\n\n" if parts else ""
        remaining = max_context_chars - used_characters
        available_content = max(0, remaining - len(separator))

        included_text = new_content[:available_content]
        chunk_truncated = len(included_text) < len(new_content)

        if included_text:
            parts.append(f"{separator}{included_text}")
            used_characters += len(separator) + len(included_text)
            included_content_characters += len(included_text)

        removed_overlap_characters += overlap

        sources.append(
            ContextSource(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                filename=chunk.filename,
                section_title=chunk.section_title,
                chunk_index=chunk.chunk_index,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                retrieval_ranks=tuple(retrieval_ranks[chunk_id]),
                primary_retrieval_ranks=tuple(
                    primary_ranks.get(chunk_id, [])
                ),
                included_characters=len(included_text),
                removed_overlap_characters=overlap,
                truncated=chunk_truncated,
            )
        )

        if chunk_truncated:
            truncated = True
            break

        previous_chunk = chunk

        if position < len(ordered_chunk_ids) - 1:
            next_chunk = chunk_records[ordered_chunk_ids[position + 1]]

            next_group = (next_chunk.document_id, next_chunk.section_title)

            next_header_length = (
                len(_build_context_header(next_chunk))
                if next_group != current_group
                else 0
            )

            next_required = (
                2 + next_header_length + 2
                if next_group != current_group
                else 2
            )

            if max_context_chars - used_characters < next_required:
                truncated = True
                break

    content = "".join(parts)

    separator_and_label_characters = (
        len(content) - included_content_characters
    )

    duplicate_chunk_count = (
        total_chunk_references - len(ordered_chunk_ids)
    )

    return AssembledContext(
        content=content,
        sources=tuple(sources),
        max_context_chars=max_context_chars,
        total_source_characters=total_source_characters,
        included_content_characters=included_content_characters,
        separator_and_label_characters=separator_and_label_characters,
        duplicate_chunk_count=duplicate_chunk_count,
        removed_overlap_characters=removed_overlap_characters,
        unique_chunk_count=len(ordered_chunk_ids),
        included_chunk_count=sum(
            source.included_characters > 0 for source in sources
        ),
        truncated=truncated,
    )