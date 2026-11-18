"""add indexing status check constraint

Revision ID: fd843bed43fe
Revises: 4b8f2c1d6a90
"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "fd843bed43fe"
down_revision: Union[str, Sequence[str], None] = "4b8f2c1d6a90"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add allowed indexing status values."""

    op.create_check_constraint(
        "ck_documents_indexing_status",
        "documents",
        "indexing_status IN ('pending', 'processing', 'completed', 'failed')",
    )


def downgrade() -> None:
    """Remove indexing status check constraint."""

    op.drop_constraint(
        "ck_documents_indexing_status",
        "documents",
        type_="check",
    )
