import asyncio
from sqlalchemy import text
from sqlalchemy.engine.url import make_url
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import CHAT_DB_URL, log


def _is_postgres_url(db_url: str) -> bool:
    driver = (make_url(db_url).drivername or "").lower()
    return driver.startswith("postgresql")


def _quote_ident(ident: str) -> str:
    return '"' + ident.replace('"', '""') + '"'


async def ensure_database_exists(db_url: str) -> None:
    url = make_url(db_url)
    db_name = url.database
    if not db_name:
        raise RuntimeError("CHAT_DB_URL/DATABASE_URL must include a database name")

    # Connect to maintenance DB, then create target DB if missing.
    admin_url = url.set(database="postgres")
    engine = create_async_engine(admin_url.render_as_string(hide_password=False), isolation_level="AUTOCOMMIT")
    try:
        async with engine.connect() as conn:
            exists = await conn.scalar(
                text("SELECT 1 FROM pg_database WHERE datname = :name"),
                {"name": db_name},
            )
            if exists:
                log.info("db bootstrap: database %r already exists", db_name)
                return
            await conn.execute(text(f"CREATE DATABASE {_quote_ident(db_name)}"))
            log.info("db bootstrap: created database %r", db_name)
    finally:
        await engine.dispose()


async def _main() -> None:
    if not _is_postgres_url(CHAT_DB_URL):
        log.info("db bootstrap: non-postgres URL detected, skipping create-database step")
        return
    await ensure_database_exists(CHAT_DB_URL)


if __name__ == "__main__":
    asyncio.run(_main())
