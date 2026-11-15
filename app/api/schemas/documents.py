from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class DocumentResponse(BaseModel):
    """Public response representation of an uploaded document."""

    id: UUID
    filename: str
    content_type: str
    file_size_bytes: int
    status: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
