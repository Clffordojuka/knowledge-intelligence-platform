from app.embeddings.service import get_embedding_service


def main() -> None:
    service = get_embedding_service()

    texts = [
        "How does the bank assess loan applications?",
        "What criteria are used to approve credit?",
        "The weather is sunny today.",
    ]

    embeddings = service.embed_texts(texts)

    print(f"Model: {service.model_name}")
    print(f"Vector dimension: {service.vector_dimension}")
    print(f"Number of embeddings: {len(embeddings)}")

    for index, embedding in enumerate(embeddings):
        print(
            f"Embedding {index + 1}: "
            f"dimensions={len(embedding)}, "
            f"norm={sum(value**2 for value in embedding) ** 0.5:.4f}"
        )

    query_embedding = service.embed_query("What determines whether a loan is approved?")

    print(f"Query vector dimensions: {len(query_embedding)}")


if __name__ == "__main__":
    main()
