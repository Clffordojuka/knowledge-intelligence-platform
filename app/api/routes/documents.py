from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.api.schemas.documents import DocumentResponse
from app.database.dependencies import get_db
from app.database.models import Document
from app.ingestion.extractors.base import DocumentExtractionError
from app.ingestion.services.document_service import (
    DuplicateDocumentError,
    persist_document,
)
from app.ingestion.services.processing_service import (
    DocumentProcessingError,
    process_document,
)
from app.ingestion.services.storage import (
    InvalidUploadError,
    UploadTooLargeError,
    store_upload,
)
from app.vector_store.indexing import IndexingError, index_document

router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


@router.post(
    "",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> DocumentResponse:
    """Upload, process, and index a PDF or TXT document."""

    try:
        stored_file = await store_upload(file)

    except UploadTooLargeError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=str(exc),
        ) from exc

    except InvalidUploadError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    original_filename = Path(file.filename or "").name

    try:
        document = persist_document(
            db=db,
            original_filename=original_filename,
            stored_file=stored_file,
        )

    except DuplicateDocumentError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    try:
        chunks = process_document(
            db=db,
            document_id=document.id,
        )

    except (DocumentExtractionError, DocumentProcessingError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": "Document was uploaded but processing failed.",
                "document_id": str(document.id),
                "error": str(exc),
            },
        ) from exc

    try:
        index_document(
            db=db,
            document_id=document.id,
        )

    except IndexingError:
        # Processing succeeded, but vector indexing failed.
        # index_document() records indexing_status="failed".
        pass

    response = DocumentResponse.model_validate(document)

    return response.model_copy(update={"chunk_count": len(chunks)})


@router.post(
    "/{document_id}/index",
    response_model=DocumentResponse,
    status_code=status.HTTP_200_OK,
)
def retry_document_indexing(
    document_id: UUID,
    db: Session = Depends(get_db),
) -> DocumentResponse:
    """Retry vector indexing for an already processed document."""

    document = db.get(Document, document_id)

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    try:
        result = index_document(
            db=db,
            document_id=document_id,
        )

    except IndexingError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    response = DocumentResponse.model_validate(document)

    return response.model_copy(update={"chunk_count": result.indexed_chunks})
