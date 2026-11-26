
import json
from pathlib import Path

EXPORT_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "evaluation"
    / "section_aware_corpus_chunks.json"
)


def main() -> None:
    data = json.loads(EXPORT_PATH.read_text(encoding="utf-8"))

    print(f"Documents: {data['document_count']}")
    print(f"Total chunks: {data['total_chunk_count']}")

    assert data["document_count"] == 5
    assert data["total_chunk_count"] == 24

    for document in data["documents"]:
        print(
            f"\n{document['filename']} "
            f"- {document['chunk_count']} chunks"
        )

        for chunk in document["chunks"]:
            preview = chunk["content"][:180].replace("\n", " ")

            print(
                f"  Chunk {chunk['chunk_index']} | "
                f"chars {chunk['char_start']}:{chunk['char_end']} | "
                f"{len(chunk['content'])} characters"
            )
            print(f"    {preview}")

    print("\nExport validation passed.")


if __name__ == "__main__":
    main()