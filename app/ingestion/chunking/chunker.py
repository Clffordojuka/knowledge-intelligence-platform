
import re
from dataclasses import dataclass
from typing import Literal

from app.ingestion.extractors.base import ExtractionResult
from app.ingestion.services.text_cleaner import clean_text

ChunkingStrategy = Literal["character", "section_aware"]


@dataclass(frozen=True)
class ChunkingConfig:
    """Configuration for document text chunking."""

    chunk_size: int = 800
    chunk_overlap: int = 120
    strategy: ChunkingStrategy = "character"

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")

        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )

        if self.strategy not in {"character", "section_aware"}:
            raise ValueError(
                "strategy must be 'character' or 'section_aware'."
            )


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


@dataclass(frozen=True)
class _SectionSpan:
    """Character range occupied by a numbered document section."""

    start: int
    end: int


PARAGRAPH_BOUNDARY = re.compile(r"\n{2,}")
SENTENCE_BOUNDARY = re.compile(r"""(?<=[.!?])["')\]]*\s+""")
WORD_BOUNDARY = re.compile(r"\s+")

SECTION_HEADING = re.compile(
    r"^\s*\d+(?:\.\d+)*[.)]?\s+[A-Z][^\n]{1,119}\s*$"
)


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

    paragraph_positions = [
        start + match.start()
        for match in PARAGRAPH_BOUNDARY.finditer(search_text)
        if start + match.start() >= minimum_end
    ]

    if paragraph_positions:
        return paragraph_positions[-1]

    sentence_positions = [
        start + match.end()
        for match in SENTENCE_BOUNDARY.finditer(search_text)
        if start + match.end() >= minimum_end
    ]

    if sentence_positions:
        return sentence_positions[-1]

    word_positions = [
        start + match.end()
        for match in WORD_BOUNDARY.finditer(search_text)
        if start + match.end() >= minimum_end
    ]

    if word_positions:
        return word_positions[-1]

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
        span.page_number
        for span in page_spans
        if span.start < end and span.end > start
    ]

    if not overlapping_pages:
        raise ValueError(
            "Could not determine source pages for the generated chunk."
        )

    return min(overlapping_pages), max(overlapping_pages)


def _find_section_spans(text: str) -> list[_SectionSpan]:
    """
    Identify numbered sections.

    The document title and preamble remain attached to the first
    numbered section. Only headings after the first establish
    new section boundaries.
    """

    heading_positions: list[int] = []
    current_position = 0
    first_heading_found = False

    for line in text.splitlines(keepends=True):
        if SECTION_HEADING.match(line.rstrip("\r\n")):
            if first_heading_found:
                heading_positions.append(current_position)
            else:
                first_heading_found = True

        current_position += len(line)

    if not first_heading_found:
        return []

    boundaries = [0, *heading_positions, len(text)]

    return [
        _SectionSpan(start=start, end=end)
        for start, end in zip(boundaries, boundaries[1:])
        if start < end
    ]


def _append_chunks_for_range(
    text: str,
    section_start: int,
    section_end: int,
    page_spans: list[_PageSpan],
    config: ChunkingConfig,
    chunks: list[TextChunk],
) -> None:
    """
    Chunk a text range independently.

    Overlap is restricted to the current range, preventing chunks
    from inheriting content from an unrelated section.
    """

    start = section_start

    while start < section_end:
        limit = min(start + config.chunk_size, section_end)

        if limit == section_end:
            end = limit
        else:
            end = _find_chunk_end(text, start, limit)

        actual_start = start
        actual_end = end

        while (
            actual_start < actual_end
            and text[actual_start].isspace()
        ):
            actual_start += 1

        while (
            actual_end > actual_start
            and text[actual_end - 1].isspace()
        ):
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

        if end >= section_end:
            break

        next_start = _find_next_start(
            text,
            end,
            config.chunk_overlap,
        )

        # Keep overlap inside the current section.
        next_start = max(next_start, section_start)

        if next_start <= start:
            next_start = end

        start = next_start


def chunk_extraction_result(
    extraction: ExtractionResult,
    config: ChunkingConfig | None = None,
) -> list[TextChunk]:
    """Convert extracted pages into traceable text chunks."""

    config = config or ChunkingConfig()

    text, page_spans = _build_document_text(extraction)

    if not text.strip():
        return []

    chunks: list[TextChunk] = []

    if config.strategy == "section_aware":
        sections = _find_section_spans(text)

        if sections:
            for section in sections:
                _append_chunks_for_range(
                    text=text,
                    section_start=section.start,
                    section_end=section.end,
                    page_spans=page_spans,
                    config=config,
                    chunks=chunks,
                )

            return chunks

    # Default behavior and fallback when no numbered headings exist.
    _append_chunks_for_range(
        text=text,
        section_start=0,
        section_end=len(text),
        page_spans=page_spans,
        config=config,
        chunks=chunks,
    )

    return chunks