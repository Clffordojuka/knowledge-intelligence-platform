from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.core.config import settings


class EmbeddingServiceError(Exception):
    """Raised when embedding generation fails."""


class EmbeddingService:
    """Generate dense vector embeddings using Sentence Transformers."""

    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        self.model_name = (
            settings.embedding_model_name if model_name is None else model_name
        )
        self.batch_size = (
            settings.embedding_batch_size if batch_size is None else batch_size
        )

        if self.batch_size <= 0:
            raise ValueError("Embedding batch size must be positive.")

        try:
            self.model = SentenceTransformer(self.model_name)
            self.vector_dimension = self.model.get_embedding_dimension()

            if self.vector_dimension is None:
                raise EmbeddingServiceError(
                    "Could not determine the embedding vector dimension."
                )

        except Exception as exc:
            raise EmbeddingServiceError(
                f"Failed to load embedding model: {self.model_name}"
            ) from exc

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Convert a batch of texts into normalized embedding vectors."""

        if not texts:
            return []

        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("All input texts must be non-empty strings.")

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            return embeddings.tolist()

        except Exception as exc:
            raise EmbeddingServiceError("Failed to generate text embeddings.") from exc

    def embed_query(
        self,
        text: str,
        instruction: str | None = None,
    ) -> list[float]:
        """Generate an embedding for a query with an optional instruction."""

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Query text must be a non-empty string.")

        if instruction is not None and not isinstance(instruction, str):
            raise ValueError("Query instruction must be a string or None.")

        query_text = text

        if instruction:
            query_text = f"{instruction}{text}"

        return self.embed_texts([query_text])[0]


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    """Return a shared embedding service instance."""

    return EmbeddingService()
