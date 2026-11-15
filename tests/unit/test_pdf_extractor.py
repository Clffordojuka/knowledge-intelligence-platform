import pytest

from app.ingestion.extractors import pdf_extractor
from app.ingestion.extractors.base import DocumentExtractionError
from app.ingestion.extractors.pdf_extractor import PDFExtractor


class FakePage:
    def __init__(self, text):
        self.text = text

    def extract_text(self):
        return self.text


class FakePdfReader:
    def __init__(self, file_path, strict=True):
        self.is_encrypted = False
        self.pages = [
            FakePage("Credit policy introduction."),
            FakePage("Eligibility requirements."),
        ]


def test_extract_pdf_preserves_page_numbers(monkeypatch, tmp_path):
    monkeypatch.setattr(
        pdf_extractor,
        "PdfReader",
        FakePdfReader,
    )

    result = PDFExtractor().extract(tmp_path / "policy.pdf")

    assert result.source_format == "pdf"
    assert result.page_count == 2
    assert result.pages[0].page_number == 1
    assert result.pages[1].page_number == 2
    assert result.pages[0].text == "Credit policy introduction."
    assert result.pages[1].text == "Eligibility requirements."


def test_reject_encrypted_pdf(monkeypatch, tmp_path):
    class EncryptedPdfReader:
        def __init__(self, file_path, strict=True):
            self.is_encrypted = True

    monkeypatch.setattr(
        pdf_extractor,
        "PdfReader",
        EncryptedPdfReader,
    )

    with pytest.raises(DocumentExtractionError, match="Encrypted"):
        PDFExtractor().extract(tmp_path / "encrypted.pdf")


def test_reject_pdf_without_pages(monkeypatch, tmp_path):
    class EmptyPdfReader:
        def __init__(self, file_path, strict=True):
            self.is_encrypted = False
            self.pages = []

    monkeypatch.setattr(
        pdf_extractor,
        "PdfReader",
        EmptyPdfReader,
    )

    with pytest.raises(DocumentExtractionError, match="no pages"):
        PDFExtractor().extract(tmp_path / "empty.pdf")


def test_reject_pdf_without_extractable_text(monkeypatch, tmp_path):
    class ScannedPdfReader:
        def __init__(self, file_path, strict=True):
            self.is_encrypted = False
            self.pages = [FakePage(""), FakePage("   ")]

    monkeypatch.setattr(
        pdf_extractor,
        "PdfReader",
        ScannedPdfReader,
    )

    with pytest.raises(DocumentExtractionError, match="no extractable text"):
        PDFExtractor().extract(tmp_path / "scanned.pdf")


def test_reject_pdf_when_page_extraction_fails(monkeypatch, tmp_path):
    class BrokenPage:
        def extract_text(self):
            raise RuntimeError("PDF page extraction failed")

    class BrokenPdfReader:
        def __init__(self, file_path, strict=True):
            self.is_encrypted = False
            self.pages = [BrokenPage()]

    monkeypatch.setattr(
        pdf_extractor,
        "PdfReader",
        BrokenPdfReader,
    )

    with pytest.raises(
        DocumentExtractionError,
        match="page 1",
    ):
        PDFExtractor().extract(tmp_path / "broken.pdf")
