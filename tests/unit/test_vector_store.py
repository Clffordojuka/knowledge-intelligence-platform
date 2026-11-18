from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from qdrant_client.http.models import Distance, VectorParams

from app.core.config import settings
from app.vector_store.service import (
    VectorStoreError,
    check_qdrant_connection,
    ensure_collection,
)


def test_qdrant_connection_success():
    client = MagicMock()

    assert check_qdrant_connection(client) is True

    client.get_collections.assert_called_once()


def test_qdrant_connection_failure():
    client = MagicMock()
    client.get_collections.side_effect = RuntimeError("Connection failed")

    with pytest.raises(VectorStoreError, match="Unable to connect"):
        check_qdrant_connection(client)


def test_ensure_collection_creates_collection():
    client = MagicMock()
    client.collection_exists.return_value = False

    client.get_collection.return_value = SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(
                vectors=VectorParams(
                    size=settings.embedding_dimension,
                    distance=Distance.COSINE,
                )
            )
        )
    )

    ensure_collection(client)

    client.create_collection.assert_called_once()


def test_ensure_collection_does_not_recreate_existing_collection():
    client = MagicMock()
    client.collection_exists.return_value = True

    client.get_collection.return_value = SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(
                vectors=VectorParams(
                    size=settings.embedding_dimension,
                    distance=Distance.COSINE,
                )
            )
        )
    )

    ensure_collection(client)

    client.create_collection.assert_not_called()


def test_ensure_collection_rejects_wrong_vector_dimension():
    client = MagicMock()
    client.collection_exists.return_value = True

    client.get_collection.return_value = SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(
                vectors=VectorParams(
                    size=768,
                    distance=Distance.COSINE,
                )
            )
        )
    )

    with pytest.raises(VectorStoreError, match="dimension mismatch"):
        ensure_collection(client)


def test_ensure_collection_rejects_wrong_distance_metric():
    client = MagicMock()
    client.collection_exists.return_value = True

    client.get_collection.return_value = SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(
                vectors=VectorParams(
                    size=settings.embedding_dimension,
                    distance=Distance.DOT,
                )
            )
        )
    )

    with pytest.raises(VectorStoreError, match="cosine"):
        ensure_collection(client)
