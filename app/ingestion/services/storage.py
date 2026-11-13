import codecs
import hashlib
import os
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

from fastapi import UploadFile

from app.core.config import settings

CHUNK_SIZE = 64 * 1024
PDF_HEADER = b"%PDF-"
PDF_HEADER_SEARCH_LIMIT = 1024


class InvalidUploadError(Exception):
    """Raised when an uploaded file fails validation."""


class UploadTooLargeError(Exception):
    """Raised when an uploaded file exceeds the configured size limit."""


@dataclass(frozen=True)
class StoredFile:
    """Information about a successfully stored file."""

    path: Path
    sha256: str
    size_bytes: int
    content_type: str


async def store_upload(
    upload: UploadFile,
    storage_dir: Path | None = None,
    max_size_bytes: int | None = None,
) -> StoredFile:
    """Validate, hash, and safely store an uploaded PDF or UTF-8 TXT file."""

    destination_dir = storage_dir or settings.upload_storage_dir
    size_limit = (
        max_size_bytes if max_size_bytes is not None else settings.max_upload_size_bytes
    )

    if not upload.filename:
        raise InvalidUploadError("A filename is required.")

    extension = Path(upload.filename).suffix.lower()

    if extension not in {".pdf", ".txt"}:
        raise InvalidUploadError("Only PDF and TXT files are supported.")

    if size_limit <= 0:
        raise ValueError("The maximum upload size must be positive.")

    destination_dir.mkdir(parents=True, exist_ok=True)

    temp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".kip-upload-",
            suffix=".tmp",
            dir=destination_dir,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)

            hasher = hashlib.sha256()
            total_size = 0
            header = bytearray()

            decoder = (
                codecs.getincrementaldecoder("utf-8")() if extension == ".txt" else None
            )

            while chunk := await upload.read(CHUNK_SIZE):
                total_size += len(chunk)

                if total_size > size_limit:
                    raise UploadTooLargeError(
                        f"File exceeds the {size_limit}-byte limit."
                    )

                if len(header) < PDF_HEADER_SEARCH_LIMIT:
                    remaining = PDF_HEADER_SEARCH_LIMIT - len(header)
                    header.extend(chunk[:remaining])

                if decoder is not None:
                    if b"\x00" in chunk:
                        raise InvalidUploadError(
                            "TXT files must contain valid UTF-8 text."
                        )

                    try:
                        decoder.decode(chunk, final=False)
                    except UnicodeDecodeError as exc:
                        raise InvalidUploadError(
                            "TXT files must contain valid UTF-8 text."
                        ) from exc

                hasher.update(chunk)
                temp_file.write(chunk)

            if total_size == 0:
                raise InvalidUploadError("Empty files are not allowed.")

            if extension == ".pdf":
                if PDF_HEADER not in header:
                    raise InvalidUploadError(
                        "The uploaded file does not have a valid PDF header."
                    )

                content_type = "application/pdf"

            else:
                try:
                    decoder.decode(b"", final=True)
                except UnicodeDecodeError as exc:
                    raise InvalidUploadError(
                        "TXT files must contain valid UTF-8 text."
                    ) from exc

                content_type = "text/plain"

            temp_file.flush()
            os.fsync(temp_file.fileno())

        final_filename = f"{uuid.uuid4()}{extension}"
        final_path = destination_dir / final_filename

        os.replace(temp_path, final_path)
        temp_path = None

        return StoredFile(
            path=final_path.resolve(),
            sha256=hasher.hexdigest(),
            size_bytes=total_size,
            content_type=content_type,
        )

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
