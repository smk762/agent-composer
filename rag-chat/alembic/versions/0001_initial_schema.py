"""Initial schema: conversations, messages, api_keys

Revision ID: 0001
Revises:
Create Date: 2026-02-13

Handles three scenarios:
  1. Fresh database  -> creates all tables from scratch.
  2. Existing DB without tool columns -> adds tool_calls / tool_call_id.
  3. Existing DB already fully migrated -> no-ops gracefully.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect as sa_inspect


revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(table_name: str) -> bool:
    """Check whether a table already exists in the database."""
    bind = op.get_bind()
    inspector = sa_inspect(bind)
    return table_name in inspector.get_table_names()


def _column_exists(table_name: str, column_name: str) -> bool:
    """Check whether a column already exists on a table."""
    bind = op.get_bind()
    inspector = sa_inspect(bind)
    columns = [c["name"] for c in inspector.get_columns(table_name)]
    return column_name in columns


def upgrade() -> None:
    # ---- conversations ----
    if not _table_exists("conversations"):
        op.create_table(
            "conversations",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("user_id", sa.String(), nullable=False, index=True),
            sa.Column("title", sa.String(), nullable=True),
            sa.Column("summary", sa.Text(), nullable=True),
            sa.Column("model", sa.String(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        )

    # ---- messages ----
    if not _table_exists("messages"):
        op.create_table(
            "messages",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("conversation_id", sa.String(), nullable=False, index=True),
            sa.Column("role", sa.String(), nullable=False),
            sa.Column("content", sa.Text(), nullable=True),
            sa.Column("tool_calls", sa.Text(), nullable=True),
            sa.Column("tool_call_id", sa.String(), nullable=True),
            sa.Column("seq", sa.Integer(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        )
    else:
        # Existing table -- add columns introduced in the tool-calling update.
        with op.batch_alter_table("messages") as batch_op:
            if not _column_exists("messages", "tool_calls"):
                batch_op.add_column(sa.Column("tool_calls", sa.Text(), nullable=True))
            if not _column_exists("messages", "tool_call_id"):
                batch_op.add_column(sa.Column("tool_call_id", sa.String(), nullable=True))

    # ---- api_keys ----
    if not _table_exists("api_keys"):
        op.create_table(
            "api_keys",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("user_id", sa.String(), nullable=False, index=True),
            sa.Column("name", sa.String(), nullable=True),
            sa.Column("prefix", sa.String(), nullable=False, index=True),
            sa.Column("hashed_key", sa.String(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        )


def downgrade() -> None:
    op.drop_table("api_keys")
    op.drop_table("messages")
    op.drop_table("conversations")
