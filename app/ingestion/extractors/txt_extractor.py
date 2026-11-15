from pathlib import Path

from app.ingestion.extractors.base import (
    DocumentExtractionError,
    ExtractedPage,
    ExtractionResult,
)


class TXTExtractor:
    """Extract UTF-8 text from TXT documents."""

    def extract(self, file_path: Path) -> ExtractionResult:
        """Read a TXT document and return its extracted text."""

        try:
            # utf-8-sig also handles an optional UTF-8 byte-order mark.
            text = file_path.read_text(encoding="utf-8-sig")

        except UnicodeDecodeError as exc:
            raise DocumentExtractionError(
                "The TXT document is not valid UTF-8."
            ) from exc

        except OSError as exc:
            raise DocumentExtractionError(
                "The TXT document could not be read."
            ) from exc

        if not text.strip():
            raise DocumentExtractionError(
                "The TXT document contains no extractable text."
            )

        return ExtractionResult(
            pages=(
                ExtractedPage(
                    page_number=1,
                    text=text,
                ),
            ),
            source_format="txt",
        )
