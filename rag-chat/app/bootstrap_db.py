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


async def _can_connect(db_url: str) -> bool:
    engine = create_async_engine(db_url, isolation_level="AUTOCOMMIT")
    try:
        async with engine.connect() as conn:
            await conn.scalar(text("SELECT 1"))
        return True
    except Exception:
        return False
    finally:
        await engine.dispose()


async def ensure_database_exists(db_url: str) -> None:
    url = make_url(db_url)
    db_name = url.database
    if not db_name:
        raise RuntimeError("CHAT_DB_URL/DATABASE_URL must include a database name")

    # Fast path: if the target DB is already reachable it exists, so there is
    # nothing to do. This is the normal case on a managed/remote Postgres (the
    # stateful node), where the app user may have no rights on the maintenance
    # DB and isn't allowed to CREATE DATABASE.
    if await _can_connect(db_url):
        log.info("db bootstrap: database %r reachable, skipping create", db_name)
        return

    # Target not reachable yet: try to create it via the maintenance DB. If we
    # lack privileges or can't reach the admin DB, don't crash — log and let the
    # subsequent `alembic upgrade head` surface the real connection error.
    admin_url = url.set(database="postgres")
    try:
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
    except Exception as e:
        log.warning(
            "db bootstrap: could not auto-create database %r (%s); assuming it is "
            "provisioned on the stateful node and continuing",
            db_name,
            e,
        )


async def _main() -> None:
    if not _is_postgres_url(CHAT_DB_URL):
        log.info("db bootstrap: non-postgres URL detected, skipping create-database step")
        return
    await ensure_database_exists(CHAT_DB_URL)


if __name__ == "__main__":
    asyncio.run(_main())
