"""add document indexing status

Revision ID: 4b8f2c1d6a90
Revises: e84100357196
Create Date: 2026-11-18 17:51:25.933000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "4b8f2c1d6a90"
down_revision: Union[str, Sequence[str], None] = "e84100357196"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column(
            "indexing_status",
            sa.String(length=20),
            server_default="pending",
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_column("documents", "indexing_status")
