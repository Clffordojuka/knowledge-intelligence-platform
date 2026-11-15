from pathlib import Path

from app.ingestion.extractors.base import (
    DocumentExtractor,
    ExtractedPage,
    ExtractionResult,
)


def test_extraction_result_preserves_page_provenance():
    result = ExtractionResult(
        pages=(
            ExtractedPage(
                page_number=1,
                text="Introduction to credit policy.",
            ),
            ExtractedPage(
                page_number=2,
                text="Eligibility requirements.",
            ),
        ),
        source_format="pdf",
    )

    assert result.page_count == 2
    assert result.pages[0].page_number == 1
    assert result.pages[1].page_number == 2
    assert result.source_format == "pdf"


def test_full_text_combines_pages_in_order():
    result = ExtractionResult(
        pages=(
            ExtractedPage(page_number=1, text="First page."),
            ExtractedPage(page_number=2, text="Second page."),
        ),
        source_format="pdf",
    )

    assert result.full_text == "First page.\nSecond page."


def test_extractor_protocol_accepts_compatible_implementation():
    class FakeExtractor:
        def extract(self, file_path: Path) -> ExtractionResult:
            return ExtractionResult(
                pages=(
                    ExtractedPage(
                        page_number=1,
                        text="Sample document.",
                    ),
                ),
                source_format="txt",
            )

    extractor: DocumentExtractor = FakeExtractor()

    result = extractor.extract(Path("sample.txt"))

    assert result.page_count == 1
    assert result.full_text == "Sample document."
