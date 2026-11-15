import pytest

from app.ingestion.extractors.base import DocumentExtractionError
from app.ingestion.extractors.txt_extractor import TXTExtractor


def test_extract_valid_txt_file(tmp_path):
    content = "Internal credit policy.\nEligibility requirements."

    file_path = tmp_path / "policy.txt"
    file_path.write_text(content, encoding="utf-8")

    result = TXTExtractor().extract(file_path)

    assert result.source_format == "txt"
    assert result.page_count == 1
    assert result.pages[0].page_number == 1
    assert result.pages[0].text == content
    assert result.full_text == content


def test_extract_txt_with_utf8_bom(tmp_path):
    content = "Credit policy document."

    file_path = tmp_path / "policy.txt"
    file_path.write_text(
        "\ufeff" + content,
        encoding="utf-8",
    )

    result = TXTExtractor().extract(file_path)

    assert result.pages[0].text == content


def test_reject_empty_txt_file(tmp_path):
    file_path = tmp_path / "empty.txt"
    file_path.write_text("", encoding="utf-8")

    with pytest.raises(DocumentExtractionError, match="no extractable text"):
        TXTExtractor().extract(file_path)


def test_reject_invalid_utf8_file(tmp_path):
    file_path = tmp_path / "invalid.txt"
    file_path.write_bytes(b"\xff\xfe\xfa")

    with pytest.raises(DocumentExtractionError, match="valid UTF-8"):
        TXTExtractor().extract(file_path)


def test_reject_missing_txt_file(tmp_path):
    file_path = tmp_path / "missing.txt"

    with pytest.raises(DocumentExtractionError, match="could not be read"):
        TXTExtractor().extract(file_path)
