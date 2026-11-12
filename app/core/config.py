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

    qdrant_url: str = "http://localhost:6333"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="KIP_",
        extra="ignore",
    )


settings = Settings()
