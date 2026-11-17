import pytest

from app.ingestion.services.text_cleaner import clean_text


@pytest.mark.parametrize(
    ("input_text", "expected"),
    [
        ("  Hello world  ", "Hello world"),
        ("Hello\r\nWorld", "Hello\nWorld"),
        ("Hello\rWorld", "Hello\nWorld"),
        ("Hello     world", "Hello world"),
        ("Hello\n\n\n\nWorld", "Hello\n\nWorld"),
        ("Hello\x00World", "HelloWorld"),
        ("Hello \n World", "Hello\nWorld"),
        ("", ""),
        ("   ", ""),
    ],
)
def test_clean_text_normalizes_extracted_text(input_text, expected):
    assert clean_text(input_text) == expected


def test_clean_text_preserves_paragraphs():
    text = "First paragraph.\n\nSecond paragraph."

    assert clean_text(text) == text


def test_clean_text_preserves_punctuation_and_numbering():
    text = "1. Eligibility requirements: applicants must be 18+."

    assert clean_text(text) == text
