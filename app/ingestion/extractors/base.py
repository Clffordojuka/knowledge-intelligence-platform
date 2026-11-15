from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class ExtractedPage:
    """Text extracted from one logical document page."""

    page_number: int
    text: str


@dataclass(frozen=True)
class ExtractionResult:
    """Standardized result returned by a document extractor."""

    pages: tuple[ExtractedPage, ...]
    source_format: str

    @property
    def page_count(self) -> int:
        """Return the number of extracted pages."""

        return len(self.pages)

    @property
    def full_text(self) -> str:
        """Combine page text while preserving page order."""

        return "\n".join(page.text for page in self.pages)


class DocumentExtractionError(Exception):
    """Raised when a document cannot be extracted successfully."""


class DocumentExtractor(Protocol):
    """Contract that all document extractors must implement."""

    def extract(self, file_path: Path) -> ExtractionResult:
        """Extract text and provenance from a document."""
        ...
