
import re
from dataclasses import dataclass
from math import isfinite
from uuid import UUID

from rank_bm25 import BM25Okapi
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.models import Chunk, Document


class BM25RetrievalError(Exception):
    """Raised when keyword retrieval fails."""


@dataclass(frozen=True)
class BM25SearchResult:
    """A keyword-retrieved chunk with its provenance and score."""

    chunk_id: UUID
    document_id: UUID
    chunk_index: int
    content: str
    score: float
    filename: str
    page_start: int | None
    page_end: int | None


def tokenize(text: str) -> list[str]:
    """Normalize text into searchable lowercase tokens."""

    return re.findall(
        r"[a-z0-9]+(?:[-_][a-z0-9]+)*",
        text.lower(),
    )


def build_bm25_search_text(
    section_title: str | None,
    content: str,
) -> str:
    """Combine section metadata and chunk content for lexical matching."""

    if section_title and section_title.strip():
        return f"{section_title.strip()}\n\n{content}"

    return content


def search_bm25(
    db: Session,
    query: str,
    limit: int = 5,
    document_id: UUID | None = None,
    document_ids: list[UUID] | None = None,
) -> list[BM25SearchResult]:
    """Retrieve document chunks using BM25 keyword ranking."""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Search query must be a non-empty string.")

    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("Search limit must be a positive integer.")

    if document_id is not None and not isinstance(document_id, UUID):
        raise ValueError("Document ID must be a UUID.")

    if document_id is not None and document_ids is not None:
        raise ValueError(
            "Specify either document_id or document_ids, not both."
        )

    if document_ids is not None:
        if not isinstance(document_ids, list) or not document_ids:
            raise ValueError("Document IDs must be a non-empty list.")

        if any(not isinstance(item, UUID) for item in document_ids):
            raise ValueError("Every document ID must be a UUID.")

        if len(document_ids) != len(set(document_ids)):
            raise ValueError("Document IDs must not contain duplicates.")

    query_tokens = tokenize(query)

    if not query_tokens:
        return []

    try:
        statement = (
            select(Chunk, Document.filename)
            .join(Document, Chunk.document_id == Document.id)
            .where(Document.status == "completed")
        )

        if document_id is not None:
            statement = statement.where(
                Chunk.document_id == document_id
            )
        elif document_ids is not None:
            statement = statement.where(
                Chunk.document_id.in_(document_ids)
            )

        rows = db.execute(statement).all()

        if not rows:
            return []

        chunks = [row[0] for row in rows]
        filenames = [row[1] for row in rows]

        tokenized_corpus = [
            tokenize(
                build_bm25_search_text(
                    section_title=chunk.section_title,
                    content=chunk.content,
                )
            )
            for chunk in chunks
        ]

        valid_indices = [
            index
            for index, tokens in enumerate(tokenized_corpus)
            if tokens
        ]

        if not valid_indices:
            return []

        valid_chunks = [
            chunks[index]
            for index in valid_indices
        ]

        valid_filenames = [
            filenames[index]
            for index in valid_indices
        ]

        valid_corpus = [
            tokenized_corpus[index]
            for index in valid_indices
        ]

        bm25 = BM25Okapi(valid_corpus)
        scores = bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: (-scores[index], index),
        )

        results = []

        for index in ranked_indices:
            score = float(scores[index])

            # Do not return chunks that contain none of the query terms.
            if not isfinite(score) or score <= 0:
                continue

            chunk = valid_chunks[index]

            results.append(
                BM25SearchResult(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    score=score,
                    filename=valid_filenames[index],
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                )
            )

            if len(results) >= limit:
                break

        return results

    except BM25RetrievalError:
        raise

    except Exception as exc:
        raise BM25RetrievalError(
            "BM25 keyword retrieval failed."
        ) from exc