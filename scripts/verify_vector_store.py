from app.core.config import settings
from app.vector_store.service import (
    check_qdrant_connection,
    ensure_collection,
)


def main() -> None:
    print("Checking Qdrant connection...")

    check_qdrant_connection()

    print("Qdrant connection successful.")

    print("\nInitializing vector collection...")

    ensure_collection()

    print("Collection initialized successfully.")
    print(f"Collection: {settings.qdrant_collection_name}")
    print(f"Vector dimension: {settings.embedding_dimension}")
    print("Distance metric: Cosine")


if __name__ == "__main__":
    main()
