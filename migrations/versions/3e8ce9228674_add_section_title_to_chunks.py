
"""add section title to chunks

Revision ID: 3e8ce9228674
Revises: fd843bed43fe
Create Date: 2026-11-26 16:04:33.280516
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "3e8ce9228674"
down_revision: Union[str, Sequence[str], None] = "fd843bed43fe"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add nullable section metadata to document chunks."""
    op.add_column(
        "chunks",
        sa.Column("section_title", sa.String(length=255), nullable=True),
    )


def downgrade() -> None:
    """Remove section metadata from document chunks."""
    op.drop_column("chunks", "section_title")
