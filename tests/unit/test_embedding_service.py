from unittest.mock import MagicMock

import pytest

from app.embeddings.service import (
    EmbeddingService,
    EmbeddingServiceError,
)


@pytest.fixture
def mock_model(monkeypatch):
    """Provide a mock SentenceTransformer model."""

    model = MagicMock()
    model.get_embedding_dimension.return_value = 384

    model.encode.return_value.tolist.return_value = [
        [0.1] * 384,
        [0.2] * 384,
    ]

    monkeypatch.setattr(
        "app.embeddings.service.SentenceTransformer",
        lambda model_name: model,
    )

    return model


def test_embedding_service_initializes_model(mock_model):
    service = EmbeddingService()

    assert service.model_name == "BAAI/bge-small-en-v1.5"
    assert service.vector_dimension == 384
    assert service.batch_size == 32


def test_embedding_service_rejects_invalid_batch_size():
    with pytest.raises(ValueError, match="must be positive"):
        EmbeddingService(batch_size=0)


def test_embed_texts_generates_embeddings(mock_model):
    service = EmbeddingService()

    texts = ["Credit approval policy", "Loan eligibility requirements"]

    embeddings = service.embed_texts(texts)

    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384

    mock_model.encode.assert_called_once_with(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def test_embed_texts_returns_empty_list_for_empty_input(mock_model):
    service = EmbeddingService()

    assert service.embed_texts([]) == []

    mock_model.encode.assert_not_called()


@pytest.mark.parametrize(
    "texts",
    [
        [""],
        ["   "],
        ["Valid text", ""],
    ],
)
def test_embed_texts_rejects_empty_text(texts, mock_model):
    service = EmbeddingService()

    with pytest.raises(ValueError, match="non-empty strings"):
        service.embed_texts(texts)


def test_embed_query_generates_single_embedding(mock_model):
    service = EmbeddingService()

    embedding = service.embed_query("How are loans approved?")

    assert len(embedding) == 384

    mock_model.encode.assert_called_once_with(
        ["How are loans approved?"],
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def test_embed_query_supports_optional_instruction(mock_model):
    service = EmbeddingService()

    instruction = "Represent this sentence for searching relevant passages: "
    query = "How are loans approved?"

    embedding = service.embed_query(
        query,
        instruction=instruction,
    )

    assert len(embedding) == 384

    mock_model.encode.assert_called_once_with(
        [instruction + query],
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def test_embed_query_without_instruction_preserves_original_text(mock_model):
    service = EmbeddingService()

    service.embed_query(
        "How are loans approved?",
        instruction=None,
    )

    mock_model.encode.assert_called_once_with(
        ["How are loans approved?"],
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def test_embed_query_rejects_empty_query(mock_model):
    service = EmbeddingService()

    with pytest.raises(ValueError, match="non-empty string"):
        service.embed_query("   ")


def test_embed_query_rejects_invalid_instruction(mock_model):
    service = EmbeddingService()

    with pytest.raises(ValueError, match="instruction must be a string"):
        service.embed_query(
            "How are loans approved?",
            instruction=123,
        )


def test_model_loading_failure_is_wrapped(monkeypatch):
    def raise_error(model_name):
        raise RuntimeError("Model loading failed")

    monkeypatch.setattr(
        "app.embeddings.service.SentenceTransformer",
        raise_error,
    )

    with pytest.raises(EmbeddingServiceError, match="Failed to load"):
        EmbeddingService()
