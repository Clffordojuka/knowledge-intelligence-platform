import pytest

from app.ingestion.chunking.chunker import (
    ChunkingConfig,
    chunk_extraction_result,
)
from app.ingestion.extractors.base import (
    ExtractedPage,
    ExtractionResult,
)


def make_extraction(*pages: tuple[int, str]) -> ExtractionResult:
    return ExtractionResult(
        pages=tuple(
            ExtractedPage(page_number=number, text=text) for number, text in pages
        ),
        source_format="txt",
    )


def test_empty_extraction_returns_no_chunks():
    extraction = make_extraction()

    assert chunk_extraction_result(extraction) == []


def test_whitespace_only_pages_return_no_chunks():
    extraction = make_extraction(
        (1, "   "),
        (2, "\n\n"),
    )

    assert chunk_extraction_result(extraction) == []


def test_short_document_produces_one_chunk():
    extraction = make_extraction(
        (1, "Credit policy introduction."),
    )

    chunks = chunk_extraction_result(extraction)

    assert len(chunks) == 1
    assert chunks[0].content == "Credit policy introduction."
    assert chunks[0].chunk_index == 0
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 1
    assert chunks[0].char_start == 0
    assert chunks[0].char_end == len(chunks[0].content)


def test_long_document_respects_maximum_chunk_size():
    text = "This is a sentence containing useful policy information. " * 30
    extraction = make_extraction((1, text))

    chunks = chunk_extraction_result(
        extraction,
        ChunkingConfig(chunk_size=100, chunk_overlap=20),
    )

    assert len(chunks) > 1
    assert all(len(chunk.content) <= 100 for chunk in chunks)


def test_chunk_indices_are_sequential():
    text = "A sentence with policy information. " * 30
    extraction = make_extraction((1, text))

    chunks = chunk_extraction_result(
        extraction,
        ChunkingConfig(chunk_size=100, chunk_overlap=20),
    )

    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


def test_chunks_preserve_source_page_provenance():
    extraction = make_extraction(
        (1, "First page contains policy introduction."),
        (2, "Second page contains eligibility requirements."),
    )

    chunks = chunk_extraction_result(
        extraction,
        ChunkingConfig(chunk_size=200, chunk_overlap=20),
    )

    assert len(chunks) == 1
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 2


def test_large_document_chunks_retain_page_ranges():
    extraction = make_extraction(
        (1, "First page. " * 20),
        (2, "Second page. " * 20),
        (3, "Third page. " * 20),
    )

    chunks = chunk_extraction_result(
        extraction,
        ChunkingConfig(chunk_size=100, chunk_overlap=20),
    )

    assert len(chunks) > 1
    assert all(1 <= chunk.page_start <= chunk.page_end <= 3 for chunk in chunks)

    assert any(chunk.page_start == 1 for chunk in chunks)
    assert any(chunk.page_end == 3 for chunk in chunks)


def test_chunk_offsets_match_normalized_document_text():
    extraction = make_extraction(
        (1, "First paragraph.\n\nSecond paragraph."),
    )

    chunks = chunk_extraction_result(extraction)

    normalized_text = "First paragraph.\n\nSecond paragraph."

    for chunk in chunks:
        assert normalized_text[chunk.char_start : chunk.char_end] == chunk.content


def test_overlapping_chunks_retain_shared_context():
    text = "This sentence contains important policy information. " * 20
    extraction = make_extraction((1, text))

    chunks = chunk_extraction_result(
        extraction,
        ChunkingConfig(chunk_size=120, chunk_overlap=30),
    )

    assert len(chunks) > 1

    for previous, current in zip(chunks, chunks[1:]):
        assert current.char_start < previous.char_end


def test_chunking_handles_a_long_unbroken_word():
    text = "A" * 250
    extraction = make_extraction((1, text))

    chunks = chunk_extraction_result(
        extraction,
        ChunkingConfig(chunk_size=100, chunk_overlap=20),
    )

    assert len(chunks) > 1
    assert all(len(chunk.content) <= 100 for chunk in chunks)


def test_invalid_chunk_size_is_rejected():
    with pytest.raises(ValueError, match="chunk_size"):
        ChunkingConfig(chunk_size=0)


def test_negative_overlap_is_rejected():
    with pytest.raises(ValueError, match="chunk_overlap"):
        ChunkingConfig(chunk_overlap=-1)


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError, match="smaller"):
        ChunkingConfig(chunk_size=100, chunk_overlap=100)


def test_zero_overlap_is_supported():
    extraction = make_extraction(
        (1, "Policy information. " * 20),
    )

    chunks = chunk_extraction_result(
        extraction,
        ChunkingConfig(chunk_size=100, chunk_overlap=0),
    )

    assert len(chunks) > 1
    assert all(
        current.char_start >= previous.char_end
        for previous, current in zip(chunks, chunks[1:])
    )
