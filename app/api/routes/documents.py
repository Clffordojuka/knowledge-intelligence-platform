from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.api.schemas.documents import DocumentResponse
from app.database.dependencies import get_db
from app.ingestion.services.document_service import (
    DuplicateDocumentError,
    persist_document,
)
from app.ingestion.services.storage import (
    InvalidUploadError,
    UploadTooLargeError,
    store_upload,
)

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
    """Upload a PDF or TXT document and persist its metadata."""

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

    return document
