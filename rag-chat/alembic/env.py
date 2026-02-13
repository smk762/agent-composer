"""Alembic environment configuration (async-aware for aiosqlite)."""

import asyncio
import os
import sys
from logging.config import fileConfig

# Ensure the project root (/app) is on sys.path so that `app.main` is
# importable regardless of how alembic is invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alembic import context  # noqa: E402
from sqlalchemy import pool  # noqa: E402
from sqlalchemy.ext.asyncio import create_async_engine  # noqa: E402

# Import Base so autogenerate can introspect the models.
from app.main import Base  # noqa: E402


# ---- Alembic Config ----
config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _get_url() -> str:
    """Resolve the database URL the same way the application does."""
    raw = os.getenv("CHAT_DB_URL", config.get_main_option("sqlalchemy.url", ""))
    # Ensure sqlite uses the aiosqlite driver (mirrors app.main._normalize_db_url).
    if raw.startswith("sqlite:") and "+aiosqlite" not in raw:
        raw = raw.replace("sqlite:", "sqlite+aiosqlite:", 1)
    return raw


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emit SQL to stdout)."""
    url = _get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,  # Required for SQLite ALTER TABLE support
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):
    """Sync helper executed inside the async engine connection."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        render_as_batch=True,  # Required for SQLite ALTER TABLE support
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations using an async engine."""
    connectable = create_async_engine(_get_url(), poolclass=pool.NullPool)
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode (connect to the database)."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
