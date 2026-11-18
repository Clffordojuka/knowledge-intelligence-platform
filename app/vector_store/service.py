from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from app.core.config import settings
from app.vector_store.client import get_qdrant_client


class VectorStoreError(Exception):
    """Raised when a vector store operation fails."""


def check_qdrant_connection(
    client: QdrantClient | None = None,
) -> bool:
    """Verify that Qdrant is reachable."""

    client = client or get_qdrant_client()

    try:
        client.get_collections()
        return True

    except Exception as exc:
        raise VectorStoreError("Unable to connect to Qdrant.") from exc


def ensure_collection(
    client: QdrantClient | None = None,
) -> None:
    """Create the document chunk collection if necessary.

    Validate the existing collection's vector configuration.
    """

    client = client or get_qdrant_client()

    collection_name = settings.qdrant_collection_name
    expected_dimension = settings.embedding_dimension

    try:
        if not client.collection_exists(collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=expected_dimension,
                    distance=Distance.COSINE,
                ),
            )

        collection = client.get_collection(collection_name)

        vectors = collection.config.params.vectors

        if not isinstance(vectors, VectorParams):
            raise VectorStoreError(
                "The collection must use a single unnamed dense vector."
            )

        if vectors.size != expected_dimension:
            raise VectorStoreError(
                f"Vector dimension mismatch. Expected "
                f"{expected_dimension}, found {vectors.size}."
            )

        if vectors.distance != Distance.COSINE:
            raise VectorStoreError("Collection must use cosine similarity.")

    except VectorStoreError:
        raise

    except Exception as exc:
        raise VectorStoreError(
            f"Failed to initialize Qdrant collection: {collection_name}"
        ) from exc
