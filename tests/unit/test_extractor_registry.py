import pytest

from app.ingestion.extractors.base import DocumentExtractionError
from app.ingestion.extractors.pdf_extractor import PDFExtractor
from app.ingestion.extractors.registry import get_extractor
from app.ingestion.extractors.txt_extractor import TXTExtractor


def test_registry_returns_pdf_extractor():
    extractor = get_extractor("application/pdf")

    assert isinstance(extractor, PDFExtractor)


def test_registry_returns_txt_extractor():
    extractor = get_extractor("text/plain")

    assert isinstance(extractor, TXTExtractor)


def test_registry_rejects_unsupported_content_type():
    with pytest.raises(DocumentExtractionError, match="Unsupported"):
        get_extractor("application/msword")
