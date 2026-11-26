
from app.retrieval.bm25_search import build_bm25_search_text
from app.vector_store.indexing import build_embedding_text


def test_embedding_text_includes_document_section_and_original_content():
    result = build_embedding_text(
        filename="credit_policy.pdf",
        section_title="2. Repayment Capacity",
        content="Assess the borrower's ability to repay.",
    )

    assert result == (
        "Document: credit_policy.pdf\n\n"
        "Section: 2. Repayment Capacity\n\n"
        "Assess the borrower's ability to repay."
    )


def test_embedding_text_without_section_title_keeps_document_context():
    result = build_embedding_text(
        filename="credit_policy.pdf",
        section_title=None,
        content="Assess the borrower's ability to repay.",
    )

    assert result == (
        "Document: credit_policy.pdf\n\n"
        "Assess the borrower's ability to repay."
    )


def test_bm25_search_text_includes_section_title():
    result = build_bm25_search_text(
        section_title="2. Repayment Capacity",
        content="Assess the borrower's ability to repay.",
    )

    assert result == (
        "2. Repayment Capacity\n\n"
        "Assess the borrower's ability to repay."
    )


def test_bm25_search_text_without_section_title_returns_content():
    content = "Assess the borrower's ability to repay."

    result = build_bm25_search_text(
        section_title=None,
        content=content,
    )

    assert result == content


def test_bm25_tokenizer_can_match_terms_from_section_title():
    result = build_bm25_search_text(
        section_title="2. Repayment Capacity",
        content="Assess the borrower's ability to repay.",
    )

    tokens = result.lower().split()

    assert "repayment" in tokens
    assert "capacity" in tokens