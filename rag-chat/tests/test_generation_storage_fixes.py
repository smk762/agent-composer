"""Generation/provider/storage hardening: provider errors don't reach the client, bad stored
provider config is a diagnostic result, local media I/O runs off the event loop, and
media_assets.file_size_bytes is a BIGINT. In-memory SQLite and fakes; no network."""

import asyncio
import importlib.util
import io
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.models.orm import Generation, MediaAsset, ProviderConfig
from app.models.schemas import AuthContext, ImageGenerateRequest
from app.routers import generation, providers
from app.storage import LocalMediaStore

AUTH = AuthContext(user_id="u1", via="dev")
SECRET = "http://10.0.0.5:8188/internal/api?token=abc"


class _DB(unittest.IsolatedAsyncioTestCase):
    tables = (Generation, MediaAsset, ProviderConfig)

    async def asyncSetUp(self) -> None:
        self.engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with self.engine.begin() as conn:
            for t in self.tables:
                await conn.run_sync(t.__table__.create)
        self.Session = sessionmaker(self.engine, class_=AsyncSession, expire_on_commit=False)

    async def asyncTearDown(self) -> None:
        await self.engine.dispose()


class _LeakyProvider:
    async def generate_image(self, prompt, **kw):
        raise RuntimeError(f"upstream {SECRET} returned 500")

    async def generate_video(self, prompt, **kw):
        raise RuntimeError(f"upstream {SECRET} returned 500")


class _KeyErrorProvider:
    async def generate_image(self, prompt, **kw):
        raise KeyError(f"'output' missing in response from {SECRET}")


class ProviderErrorLeakTests(_DB):
    async def _patch(self, name, prov):
        async def get(db, user_id, provider_name):
            return prov
        p = patch.object(generation.ProviderRegistry, name, get)
        p.start()
        self.addCleanup(p.stop)

    async def _gen_rows(self):
        async with self.Session() as db:
            return (await db.execute(select(Generation))).scalars().all()

    async def test_image_failure_hides_exception_text(self) -> None:
        await self._patch("get_image_provider", _LeakyProvider())
        async with self.Session() as db:
            with self.assertRaises(HTTPException) as cm:
                await generation.generate_image(ImageGenerateRequest(prompt="p"), auth=AUTH, db=db)
        self.assertEqual(cm.exception.status_code, 502)
        self.assertNotIn("10.0.0.5", cm.exception.detail)
        gens = await self._gen_rows()
        self.assertEqual(gens[0].status, "failed")
        self.assertIn(gens[0].id, cm.exception.detail)
        self.assertNotIn("10.0.0.5", gens[0].error_message)

    async def test_keyerror_during_inference_is_not_a_400_with_its_text(self) -> None:
        await self._patch("get_image_provider", _KeyErrorProvider())
        async with self.Session() as db:
            with self.assertRaises(HTTPException) as cm:
                await generation.generate_image(ImageGenerateRequest(prompt="p"), auth=AUTH, db=db)
        self.assertEqual(cm.exception.status_code, 502)
        self.assertNotIn("10.0.0.5", cm.exception.detail)

    async def test_unconfigured_provider_is_still_a_400(self) -> None:
        async def get(db, user_id, provider_name):
            raise LookupError("No configured provider for image_generate")
        p = patch.object(generation.ProviderRegistry, "get_image_provider", get)
        p.start()
        self.addCleanup(p.stop)
        async with self.Session() as db:
            with self.assertRaises(HTTPException) as cm:
                await generation.generate_image(ImageGenerateRequest(prompt="p"), auth=AUTH, db=db)
        self.assertEqual(cm.exception.status_code, 400)
        self.assertIn("No configured provider", cm.exception.detail)

    async def test_video_failure_hides_exception_text(self) -> None:
        await self._patch("get_video_provider", _LeakyProvider())
        async with self.Session() as db:
            with self.assertRaises(HTTPException) as cm:
                await generation.generate_video(prompt="p", image=None, provider=None, model=None,
                                                duration=None, aspect_ratio=None, auth=AUTH, db=db)
        self.assertEqual(cm.exception.status_code, 502)
        self.assertNotIn("10.0.0.5", cm.exception.detail)
        gens = await self._gen_rows()
        self.assertNotIn("10.0.0.5", gens[0].error_message)


class ProviderTestEndpointTests(_DB):
    async def _store(self, *, key="plain:c2stdGVzdA==", settings_json=None):
        async with self.Session() as db:
            db.add(ProviderConfig(user_id="u1", provider_name="openai",
                                  encrypted_api_key=key, settings_json=settings_json))
            await db.commit()

    async def _test(self):
        async with self.Session() as db:
            return await providers.test_provider("openai", auth=AUTH, db=db)

    async def test_api_key_in_settings_is_not_a_500(self) -> None:
        await self._store(settings_json='{"api_key": "other"}')
        with patch.object(providers._PROVIDER_CLASSES["openai"], "validate_config",
                          return_value=True):
            out = await self._test()
        self.assertTrue(out["ok"], out)

    async def test_corrupt_settings_json_is_a_result(self) -> None:
        await self._store(settings_json="{not json")
        out = await self._test()
        self.assertFalse(out["ok"])
        self.assertIn("settings", out["error"])

    async def test_non_object_settings_is_a_result(self) -> None:
        await self._store(settings_json="[1, 2]")
        out = await self._test()
        self.assertFalse(out["ok"])

    async def test_undecryptable_key_is_a_result(self) -> None:
        await self._store(key="fernet:garbage")
        with patch("app.providers._crypto.decrypt_value", return_value=None):
            out = await self._test()
        self.assertFalse(out["ok"])
        self.assertIn("decrypt", out["error"])

    async def test_decrypt_raising_is_a_result(self) -> None:
        await self._store()
        with patch("app.providers._crypto.decrypt_value", side_effect=ValueError("bad padding")):
            out = await self._test()
        self.assertFalse(out["ok"])


class LocalStoreOffLoopTests(unittest.TestCase):
    def test_file_io_runs_off_the_event_loop_thread(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            store = LocalMediaStore(d)
            loop_thread = threading.get_ident()
            threads = []
            real_write = Path.write_bytes
            real_read = Path.read_bytes

            def write(self, data):
                threads.append(threading.get_ident())
                return real_write(self, data)

            def read(self):
                threads.append(threading.get_ident())
                return real_read(self)

            async def go():
                rel = await store.save_bytes(b"x" * 10, user_id="u", generation_id="g", filename="a.png")
                data = await store.read_bytes(rel, user_id="u")
                await store.delete(rel)
                gone = await store.read_bytes(rel)
                return data, gone

            with patch.object(Path, "write_bytes", write), patch.object(Path, "read_bytes", read):
                data, gone = asyncio.run(go())
            self.assertEqual(data, b"x" * 10)
            self.assertIsNone(gone)
            self.assertEqual(len(threads), 2)
            self.assertNotIn(loop_thread, threads)


def _load_migration():
    path = Path(__file__).resolve().parents[1] / "alembic" / "versions" / "0003_media_file_size_bigint.py"
    spec = importlib.util.spec_from_file_location("mig_0003", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FileSizeBigIntTests(unittest.TestCase):
    def test_orm_column_is_bigint(self) -> None:
        self.assertIsInstance(MediaAsset.__table__.c.file_size_bytes.type, sa.BigInteger)

    def test_migration_follows_0002(self) -> None:
        mig = _load_migration()
        self.assertEqual((mig.revision, mig.down_revision), ("0003", "0002"))

    def test_migration_emits_postgres_alter_type(self) -> None:
        mig = _load_migration()
        buf = io.StringIO()
        ctx = MigrationContext.configure(dialect=postgresql.dialect(),
                                         opts={"as_sql": True, "output_buffer": buf})
        with Operations.context(ctx):
            mig.upgrade()
        sql = buf.getvalue()
        self.assertIn("ALTER TABLE media_assets ALTER COLUMN file_size_bytes TYPE BIGINT", sql)

    def test_migration_runs_on_sqlite_and_keeps_rows(self) -> None:
        mig = _load_migration()
        engine = sa.create_engine("sqlite://")
        meta = sa.MetaData()
        sa.Table("media_assets", meta,
                 sa.Column("id", sa.String(), primary_key=True),
                 sa.Column("file_size_bytes", sa.Integer(), nullable=True))
        with engine.begin() as conn:
            meta.create_all(conn)
            conn.execute(sa.text("INSERT INTO media_assets VALUES ('a', 5)"))
            with Operations.context(MigrationContext.configure(conn)):
                mig.upgrade()
            cols = {c["name"]: c["type"] for c in sa.inspect(conn).get_columns("media_assets")}
            self.assertIsInstance(cols["file_size_bytes"], sa.BigInteger)
            conn.execute(sa.text("UPDATE media_assets SET file_size_bytes = :n"), {"n": 5 * 2**30})
            self.assertEqual(conn.execute(sa.text("SELECT file_size_bytes FROM media_assets")).scalar(),
                             5 * 2**30)


if __name__ == "__main__":
    unittest.main()
