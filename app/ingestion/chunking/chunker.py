import re
from dataclasses import dataclass

from app.ingestion.extractors.base import ExtractionResult
from app.ingestion.services.text_cleaner import clean_text


@dataclass(frozen=True)
class ChunkingConfig:
    """Configuration for document text chunking."""

    chunk_size: int = 800
    chunk_overlap: int = 120

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")

        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")


@dataclass(frozen=True)
class TextChunk:
    """A text chunk with its source location and character offsets."""

    chunk_index: int
    content: str
    page_start: int
    page_end: int
    char_start: int
    char_end: int


@dataclass(frozen=True)
class _PageSpan:
    """Character range occupied by a page in normalized document text."""

    page_number: int
    start: int
    end: int


PARAGRAPH_BOUNDARY = re.compile(r"\n{2,}")
SENTENCE_BOUNDARY = re.compile(r'(?<=[.!?])["\')\]]*\s+')
WORD_BOUNDARY = re.compile(r"\s+")


def _build_document_text(
    extraction: ExtractionResult,
) -> tuple[str, list[_PageSpan]]:
    """Combine cleaned pages and retain their positions in the text."""

    text_parts: list[str] = []
    page_spans: list[_PageSpan] = []
    current_position = 0

    for page in extraction.pages:
        cleaned_text = clean_text(page.text)

        if not cleaned_text:
            continue

        if text_parts:
            separator = "\n\n"
            text_parts.append(separator)
            current_position += len(separator)

        page_start = current_position
        text_parts.append(cleaned_text)
        current_position += len(cleaned_text)

        page_spans.append(
            _PageSpan(
                page_number=page.page_number,
                start=page_start,
                end=current_position,
            )
        )

    return "".join(text_parts), page_spans


def _find_chunk_end(
    text: str,
    start: int,
    limit: int,
) -> int:
    """Find a natural boundary without exceeding the maximum size."""

    minimum_end = start + max(1, (limit - start) // 2)
    search_text = text[start:limit]

    # Prefer paragraph boundaries.
    paragraph_positions = [
        start + match.start()
        for match in PARAGRAPH_BOUNDARY.finditer(search_text)
        if start + match.start() >= minimum_end
    ]

    if paragraph_positions:
        return paragraph_positions[-1]

    # Otherwise, prefer sentence boundaries.
    sentence_positions = [
        start + match.end()
        for match in SENTENCE_BOUNDARY.finditer(search_text)
        if start + match.end() >= minimum_end
    ]

    if sentence_positions:
        return sentence_positions[-1]

    # Otherwise, split at a word boundary.
    word_positions = [
        start + match.end()
        for match in WORD_BOUNDARY.finditer(search_text)
        if start + match.end() >= minimum_end
    ]

    if word_positions:
        return word_positions[-1]

    # No natural boundary: split at the maximum allowed length.
    return limit


def _find_next_start(
    text: str,
    end: int,
    overlap: int,
) -> int:
    """Calculate the next chunk start while preserving useful overlap."""

    target = max(0, end - overlap)

    if target == 0:
        return target

    # Prefer starting at the next word boundary.
    match = WORD_BOUNDARY.search(text, target, end)

    if match is not None:
        return match.end()

    return target


def _get_page_range(
    start: int,
    end: int,
    page_spans: list[_PageSpan],
) -> tuple[int, int]:
    """Determine which source pages overlap a chunk."""

    overlapping_pages = [
        span.page_number for span in page_spans if span.start < end and span.end > start
    ]

    if not overlapping_pages:
        raise ValueError("Could not determine source pages for the generated chunk.")

    return min(overlapping_pages), max(overlapping_pages)


def chunk_extraction_result(
    extraction: ExtractionResult,
    config: ChunkingConfig | None = None,
) -> list[TextChunk]:
    """Convert extracted pages into overlapping, traceable text chunks."""

    config = config or ChunkingConfig()

    text, page_spans = _build_document_text(extraction)

    if not text.strip():
        return []

    chunks: list[TextChunk] = []
    start = 0

    while start < len(text):
        limit = min(start + config.chunk_size, len(text))

        if limit == len(text):
            end = limit
        else:
            end = _find_chunk_end(text, start, limit)

        # Remove whitespace from chunk edges while retaining
        # offsets that refer to the normalized document text.
        actual_start = start
        actual_end = end

        while actual_start < actual_end and text[actual_start].isspace():
            actual_start += 1

        while actual_end > actual_start and text[actual_end - 1].isspace():
            actual_end -= 1

        if actual_start < actual_end:
            page_start, page_end = _get_page_range(
                actual_start,
                actual_end,
                page_spans,
            )

            chunks.append(
                TextChunk(
                    chunk_index=len(chunks),
                    content=text[actual_start:actual_end],
                    page_start=page_start,
                    page_end=page_end,
                    char_start=actual_start,
                    char_end=actual_end,
                )
            )

        if end >= len(text):
            break

        next_start = _find_next_start(
            text,
            end,
            config.chunk_overlap,
        )

        # Defensive safeguard against an infinite loop.
        if next_start <= start:
            next_start = end

        start = next_start

    return chunks
