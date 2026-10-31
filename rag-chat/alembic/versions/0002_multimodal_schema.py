"""Add provider_configs, generations, and media_assets tables for multimodal support.

Revision ID: 0002
Revises: 0001
Create Date: 2026-02-25
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect as sa_inspect

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(table_name: str) -> bool:
    bind = op.get_bind()
    inspector = sa_inspect(bind)
    return table_name in inspector.get_table_names()


def upgrade() -> None:
    if not _table_exists("provider_configs"):
        op.create_table(
            "provider_configs",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("user_id", sa.String(), nullable=False, index=True),
            sa.Column("provider_name", sa.String(), nullable=False, index=True),
            sa.Column("encrypted_api_key", sa.Text(), nullable=True),
            sa.Column("settings_json", sa.Text(), nullable=True),
            sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        )

    if not _table_exists("generations"):
        op.create_table(
            "generations",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("user_id", sa.String(), nullable=False, index=True),
            sa.Column("provider_name", sa.String(), nullable=False),
            sa.Column("capability", sa.String(), nullable=False),
            sa.Column("status", sa.String(), nullable=False, server_default="pending"),
            sa.Column("request_params_json", sa.Text(), nullable=True),
            sa.Column("result_json", sa.Text(), nullable=True),
            sa.Column("error_message", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        )

    if not _table_exists("media_assets"):
        op.create_table(
            "media_assets",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("generation_id", sa.String(), nullable=True, index=True),
            sa.Column("user_id", sa.String(), nullable=False, index=True),
            sa.Column("media_type", sa.String(), nullable=False),
            sa.Column("mime_type", sa.String(), nullable=False),
            sa.Column("filename", sa.String(), nullable=False),
            sa.Column("file_path", sa.String(), nullable=False),
            sa.Column("file_size_bytes", sa.Integer(), nullable=True),
            sa.Column("width", sa.Integer(), nullable=True),
            sa.Column("height", sa.Integer(), nullable=True),
            sa.Column("duration_seconds", sa.Float(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        )


def downgrade() -> None:
    op.drop_table("media_assets")
    op.drop_table("generations")
    op.drop_table("provider_configs")
