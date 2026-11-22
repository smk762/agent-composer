"""Generation routes: uploads are size-capped, and a failure while storing results marks the
generation failed and removes its files. In-memory SQLite, fake provider and media store."""

import asyncio
import io
import unittest
from unittest.mock import patch

from fastapi import HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.models.orm import Generation, MediaAsset
from app.models.schemas import AuthContext, ImageGenerateRequest
from app.providers.base import GeneratedMedia, GenerationResult
from app.routers import generation

AUTH = AuthContext(user_id="u1", via="dev")


def _upload(size: int) -> UploadFile:
    return UploadFile(file=io.BytesIO(b"x" * size), filename="in.png")


class UploadCapTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        p = patch.object(generation, "GENERATION_UPLOAD_MAX_BYTES", 1024, create=True)
        p.start()
        self.addCleanup(p.stop)

    async def test_oversized_edit_image_is_a_413(self) -> None:
        with self.assertRaises(HTTPException) as cm:
            await generation.edit_image(prompt="p", image=_upload(4096), mask=None, provider=None,
                                        model=None, size=None, auth=AUTH, db=None)
        self.assertEqual(cm.exception.status_code, 413)

    async def test_oversized_mask_is_a_413(self) -> None:
        with self.assertRaises(HTTPException) as cm:
            await generation.edit_image(prompt="p", image=_upload(10), mask=_upload(4096), provider=None,
                                        model=None, size=None, auth=AUTH, db=None)
        self.assertEqual(cm.exception.status_code, 413)

    async def test_oversized_video_first_frame_is_a_413(self) -> None:
        with self.assertRaises(HTTPException) as cm:
            await generation.generate_video(prompt="p", image=_upload(4096), provider=None, model=None,
                                            duration=None, aspect_ratio=None, auth=AUTH, db=None)
        self.assertEqual(cm.exception.status_code, 413)


class _FlakyStore:
    """Saves the first item, then fails like a full disk."""

    def __init__(self) -> None:
        self.saved: list[str] = []
        self.deleted: list[str] = []

    def ensure_dirs(self) -> None:
        pass

    async def save_bytes(self, data, *, user_id, generation_id, filename):
        if self.saved:
            raise OSError(28, "No space left on device")
        path = f"{user_id}/{generation_id}/{filename}"
        self.saved.append(path)
        return path

    async def delete(self, file_path):
        self.deleted.append(file_path)


class _Provider:
    async def generate_image(self, prompt, **kw):
        media = [GeneratedMedia(data=b"png", mime_type="image/png", filename=f"{i}.png") for i in range(2)]
        return GenerationResult(media=media, model="m", provider="fake")


class _GenerationDB(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with self.engine.begin() as conn:
            await conn.run_sync(Generation.__table__.create)
            await conn.run_sync(MediaAsset.__table__.create)
        self.Session = sessionmaker(self.engine, class_=AsyncSession, expire_on_commit=False)
        self.store = _FlakyStore()

        async def get_provider(db, user_id, name):
            return _Provider()

        for p in (
            patch.object(generation, "media_store", self.store),
            patch.object(generation.ProviderRegistry, "get_image_provider", get_provider),
        ):
            p.start()
            self.addCleanup(p.stop)

    async def asyncTearDown(self) -> None:
        await self.engine.dispose()


class StoreFailureTests(_GenerationDB):
    async def test_storage_failure_marks_the_generation_failed_and_cleans_up(self) -> None:
        async with self.Session() as db:
            with self.assertRaises(HTTPException) as cm:
                await generation.generate_image(ImageGenerateRequest(prompt="p", n=2), auth=AUTH, db=db)
        self.assertEqual(cm.exception.status_code, 500)
        async with self.Session() as db:
            gens = (await db.execute(select(Generation))).scalars().all()
            assets = (await db.execute(select(MediaAsset))).scalars().all()
        self.assertEqual([g.status for g in gens], ["failed"])
        self.assertIsNotNone(gens[0].completed_at)
        self.assertIn("No space left", gens[0].error_message)
        self.assertEqual(assets, [])
        self.assertEqual(self.store.deleted, self.store.saved)

    async def test_failed_failure_commit_still_raises_500_and_marks_failed(self) -> None:
        async with self.Session() as db:
            gen = Generation(user_id="u1", provider_name="fake", capability="image", status="running")
            db.add(gen)
            await db.commit()
            gen_id = gen.id

            async def broken_commit():
                raise ConnectionError("connection dropped")

            db.commit = broken_commit  # every commit on this session fails, like a dead connection
            result = await _Provider().generate_image("p")
            with self.assertRaises(HTTPException) as cm:
                await generation._store_results(db, gen, result, user_id="u1", media_type="image")
        self.assertEqual(cm.exception.status_code, 500)
        async with self.Session() as db:
            row = (await db.execute(select(Generation).where(Generation.id == gen_id))).scalar_one()
        self.assertEqual(row.status, "failed")
        self.assertIn("No space left", row.error_message)

    async def _running_gen(self, db) -> Generation:
        gen = Generation(user_id="u1", provider_name="fake", capability="image", status="running")
        db.add(gen)
        await db.commit()
        return gen

    async def _row(self, gen_id):
        async with self.Session() as db:
            row = (await db.execute(select(Generation).where(Generation.id == gen_id))).scalar_one()
            assets = (await db.execute(select(MediaAsset))).scalars().all()
        return row, assets

    async def test_a_failure_after_the_commit_keeps_the_stored_media(self) -> None:
        async def save(data, *, user_id, generation_id, filename):
            self.store.saved.append(f"{user_id}/{generation_id}/{filename}")
            return self.store.saved[-1]

        self.store.save_bytes = save
        async with self.Session() as db:
            gen = await self._running_gen(db)
            gen_id = gen.id

            async def broken_refresh(obj, *a, **kw):
                raise ConnectionError("pool timeout")

            db.refresh = broken_refresh
            result = await _Provider().generate_image("p")
            with self.assertRaises(Exception):
                await generation._store_results(db, gen, result, user_id="u1", media_type="image")
        row, assets = await self._row(gen_id)
        self.assertEqual((row.status, len(assets), self.store.deleted), ("completed", 2, []))

    async def test_cancellation_while_storing_cleans_up_and_marks_failed(self) -> None:
        async def cancelled_save(data, *, user_id, generation_id, filename):
            if self.store.saved:
                raise asyncio.CancelledError()
            path = f"{user_id}/{generation_id}/{filename}"
            self.store.saved.append(path)
            return path

        self.store.save_bytes = cancelled_save
        async with self.Session() as db:
            gen = await self._running_gen(db)
            gen_id = gen.id
            result = await _Provider().generate_image("p")
            with self.assertRaises(asyncio.CancelledError):
                await generation._store_results(db, gen, result, user_id="u1", media_type="image")
        row, assets = await self._row(gen_id)
        self.assertEqual((row.status, assets, self.store.deleted), ("failed", [], self.store.saved))


class LookupProviderTests(_GenerationDB):
    """A KeyError while resolving the provider is a provider error (502), not "not configured"."""

    async def test_keyerror_in_provider_lookup_is_a_502_not_a_400(self) -> None:
        async def get_provider(db, user_id, name):
            return {}["choices"]

        with patch.object(generation.ProviderRegistry, "get_image_provider", get_provider):
            async with self.Session() as db:
                with self.assertRaises(HTTPException) as cm:
                    await generation.generate_image(ImageGenerateRequest(prompt="p"), auth=AUTH, db=db)
        self.assertEqual(cm.exception.status_code, 502)
        self.assertNotIn("choices", cm.exception.detail)


if __name__ == "__main__":
    unittest.main()
