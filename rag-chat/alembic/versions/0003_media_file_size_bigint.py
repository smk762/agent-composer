"""Widen media_assets.file_size_bytes to BIGINT (a 32-bit INTEGER overflows at ~2.1 GB).

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-17
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # batch_alter_table: an in-place ALTER on Postgres, a table copy on SQLite (no ALTER TYPE).
    with op.batch_alter_table("media_assets") as batch:
        batch.alter_column(
            "file_size_bytes",
            existing_type=sa.Integer(),
            type_=sa.BigInteger(),
            existing_nullable=True,
        )


def downgrade() -> None:
    with op.batch_alter_table("media_assets") as batch:
        batch.alter_column(
            "file_size_bytes",
            existing_type=sa.BigInteger(),
            type_=sa.Integer(),
            existing_nullable=True,
        )
