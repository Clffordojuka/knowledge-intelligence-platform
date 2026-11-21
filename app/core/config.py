from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    app_name: str = "Knowledge Intelligence Platform"
    environment: str = "development"
    debug: bool = False

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "knowledge_db"
    postgres_user: str = "kip_user"
    postgres_password: str = "local_dev_password"

    # Embedding configuration
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    embedding_batch_size: int = 32
    embedding_dimension: int = 384

    # Reranking configuration
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_batch_size: int = 16
    reranker_max_length: int = 512

    # Vector database configuration
    qdrant_collection_name: str = "kip_document_chunks"

    qdrant_url: str = "http://localhost:6333"

    # Document upload configuration
    max_upload_size_bytes: int = 10 * 1024 * 1024
    upload_storage_dir: Path = Path("data/raw")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="KIP_",
        extra="ignore",
    )


settings = Settings()
