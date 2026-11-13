from io import BytesIO

import pytest
from starlette.datastructures import UploadFile

from app.ingestion.services.storage import (
    InvalidUploadError,
    UploadTooLargeError,
    store_upload,
)


@pytest.mark.anyio
async def test_store_valid_txt_file(tmp_path):
    content = b"Internal credit policy document."

    upload = UploadFile(
        filename="credit_policy.txt",
        file=BytesIO(content),
    )

    result = await store_upload(upload, storage_dir=tmp_path)

    assert result.path.exists()
    assert result.path.read_bytes() == content
    assert result.size_bytes == len(content)
    assert result.content_type == "text/plain"
    assert len(result.sha256) == 64
    assert result.path.suffix == ".txt"
    assert result.path.name != "credit_policy.txt"


@pytest.mark.anyio
async def test_reject_unsupported_extension(tmp_path):
    upload = UploadFile(
        filename="malicious.exe",
        file=BytesIO(b"some content"),
    )

    with pytest.raises(InvalidUploadError):
        await store_upload(upload, storage_dir=tmp_path)

    assert list(tmp_path.iterdir()) == []


@pytest.mark.anyio
async def test_reject_empty_file(tmp_path):
    upload = UploadFile(
        filename="empty.txt",
        file=BytesIO(b""),
    )

    with pytest.raises(InvalidUploadError):
        await store_upload(upload, storage_dir=tmp_path)

    assert list(tmp_path.iterdir()) == []


@pytest.mark.anyio
async def test_reject_invalid_pdf_signature(tmp_path):
    upload = UploadFile(
        filename="fake.pdf",
        file=BytesIO(b"This is not a PDF file."),
    )

    with pytest.raises(InvalidUploadError):
        await store_upload(upload, storage_dir=tmp_path)

    assert list(tmp_path.iterdir()) == []


@pytest.mark.anyio
async def test_reject_oversized_file(tmp_path):
    upload = UploadFile(
        filename="large.txt",
        file=BytesIO(b"1234567890"),
    )

    with pytest.raises(UploadTooLargeError):
        await store_upload(
            upload,
            storage_dir=tmp_path,
            max_size_bytes=5,
        )

    assert list(tmp_path.iterdir()) == []


@pytest.mark.anyio
async def test_identical_files_produce_identical_hashes(tmp_path):
    content = b"Same document content."

    first_upload = UploadFile(
        filename="first.txt",
        file=BytesIO(content),
    )

    second_upload = UploadFile(
        filename="second.txt",
        file=BytesIO(content),
    )

    first_result = await store_upload(
        first_upload,
        storage_dir=tmp_path,
    )

    second_result = await store_upload(
        second_upload,
        storage_dir=tmp_path,
    )

    assert first_result.sha256 == second_result.sha256
    assert first_result.path != second_result.path


@pytest.mark.anyio
async def test_store_valid_pdf_file(tmp_path):
    content = b"%PDF-1.7\nExample PDF content"

    upload = UploadFile(
        filename="policy.pdf",
        file=BytesIO(content),
    )

    result = await store_upload(upload, storage_dir=tmp_path)

    assert result.path.exists()
    assert result.path.read_bytes() == content
    assert result.content_type == "application/pdf"
    assert result.path.suffix == ".pdf"


@pytest.mark.anyio
async def test_reject_invalid_utf8_txt_file(tmp_path):
    upload = UploadFile(
        filename="invalid.txt",
        file=BytesIO(b"\xff\xfe\xfa"),
    )

    with pytest.raises(InvalidUploadError):
        await store_upload(upload, storage_dir=tmp_path)

    assert list(tmp_path.iterdir()) == []
