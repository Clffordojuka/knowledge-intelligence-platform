from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.ingestion.extractors.base import (
    DocumentExtractionError,
    ExtractedPage,
    ExtractionResult,
)


class PDFExtractor:
    """Extract text from PDF documents while preserving page numbers."""

    def extract(self, file_path: Path) -> ExtractionResult:
        """Extract text from each page of a PDF document."""

        try:
            reader = PdfReader(str(file_path), strict=True)

            if reader.is_encrypted:
                raise DocumentExtractionError(
                    "Encrypted PDF documents are not supported."
                )

            pages = []

            for page_number, page in enumerate(reader.pages, start=1):
                try:
                    text = page.extract_text() or ""
                except Exception as exc:
                    raise DocumentExtractionError(
                        f"Failed to extract text from PDF page {page_number}."
                    ) from exc

                pages.append(
                    ExtractedPage(
                        page_number=page_number,
                        text=text,
                    )
                )

            if not pages:
                raise DocumentExtractionError("The PDF document contains no pages.")

            if not any(page.text.strip() for page in pages):
                raise DocumentExtractionError(
                    "The PDF contains no extractable text. "
                    "It may be a scanned document requiring OCR."
                )

            return ExtractionResult(
                pages=tuple(pages),
                source_format="pdf",
            )

        except DocumentExtractionError:
            raise

        except (PdfReadError, OSError, ValueError) as exc:
            raise DocumentExtractionError(
                "The PDF document could not be read."
            ) from exc
