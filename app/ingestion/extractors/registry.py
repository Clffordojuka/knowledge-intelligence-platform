from app.ingestion.extractors.base import (
    DocumentExtractionError,
    DocumentExtractor,
)
from app.ingestion.extractors.pdf_extractor import PDFExtractor
from app.ingestion.extractors.txt_extractor import TXTExtractor

EXTRACTORS: dict[str, DocumentExtractor] = {
    "application/pdf": PDFExtractor(),
    "text/plain": TXTExtractor(),
}


def get_extractor(content_type: str) -> DocumentExtractor:
    """Return the extractor registered for a document content type."""

    extractor = EXTRACTORS.get(content_type)

    if extractor is None:
        raise DocumentExtractionError(
            f"Unsupported document content type: {content_type}"
        )

    return extractor
