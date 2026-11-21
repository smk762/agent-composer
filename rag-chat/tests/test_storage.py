"""Media storage: one path/key confinement for save and read on both backends; boto3 off the event loop.

No network: the MinIO store gets a fake S3 client.
"""

import asyncio
import tempfile
import threading
import unittest
from pathlib import Path

from app.storage import LocalMediaStore, MinioMediaStore, _safe_filename


class SafeFilenameTests(unittest.TestCase):
    def _assert_fallback(self, raw: str) -> None:
        self.assertEqual(_safe_filename(raw), "asset.bin", raw)

    def test_dotdot_falls_back(self) -> None:
        self._assert_fallback("..")
        self._assert_fallback("a/..")

    def test_dot_and_root_fall_back(self) -> None:
        self._assert_fallback(".")
        self._assert_fallback("/")

    def test_backslash_and_nul_fall_back(self) -> None:
        self._assert_fallback("..\\..\\etc\\passwd")
        self._assert_fallback("a\x00.png")

    def test_plain_names_are_kept(self) -> None:
        self.assertEqual(_safe_filename("cat.png"), "cat.png")
        self.assertEqual(_safe_filename("dir/cat.png"), "cat.png")
        self.assertEqual(_safe_filename(""), "asset.bin")


class LocalStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.media = self.root / "media"
        self.store = LocalMediaStore(str(self.media))

    def _save(self, **kw) -> str:
        args = {"user_id": "u@example.com", "generation_id": "g1", "filename": "x.png"} | kw
        return asyncio.run(self.store.save_bytes(b"data", **args))

    def test_save_refuses_dotdot_generation_id(self) -> None:
        with self.assertRaises(ValueError):
            self._save(generation_id="../../outside")
        self.assertFalse((self.root / "outside").exists())

    def test_save_refuses_absolute_or_slashed_user_id(self) -> None:
        for uid in (str(self.root / "abs"), "a/b", "..", ""):
            with self.assertRaises(ValueError, msg=uid):
                self._save(user_id=uid)
        self.assertFalse((self.root / "abs").exists())

    def test_save_ext_cannot_carry_path_parts(self) -> None:
        rel = self._save(filename="evil.p\\..\\..")
        self.assertEqual(rel.count("/"), 2)
        self.assertTrue((self.media / rel).is_file())

    def test_save_then_read_roundtrip_and_owner_check(self) -> None:
        rel = self._save()
        self.assertEqual(asyncio.run(self.store.read_bytes(rel)), b"data")
        self.assertEqual(asyncio.run(self.store.read_bytes(rel, user_id="u@example.com")), b"data")
        self.assertIsNone(asyncio.run(self.store.read_bytes(rel, user_id="other@example.com")))

    def test_read_refuses_traversal(self) -> None:
        (self.root / "secret.txt").write_bytes(b"s")
        self.media.mkdir()
        self.assertIsNone(asyncio.run(self.store.read_bytes("../secret.txt")))
        self.assertIsNone(self.store.resolve_path("u/../../secret.txt"))


class _FakeS3:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.threads: list[threading.Thread] = []
        self.head_calls = 0

    def head_bucket(self, **kw) -> None:
        self.threads.append(threading.current_thread())
        self.head_calls += 1

    def put_object(self, *, Bucket, Key, Body) -> None:
        self.threads.append(threading.current_thread())
        self.objects[Key] = Body

    def get_object(self, *, Bucket, Key):
        self.threads.append(threading.current_thread())

        class _Body:
            def read(_self):
                return self.objects[Key]

        return {"Body": _Body()}

    def delete_object(self, *, Bucket, Key) -> None:
        self.threads.append(threading.current_thread())
        self.objects.pop(Key, None)


def _minio(prefix: str = "media") -> tuple[MinioMediaStore, _FakeS3]:
    store = MinioMediaStore(
        endpoint_url="http://minio.invalid:9000", region="us-east-1", bucket="b",
        access_key_id="k", secret_access_key="s", use_ssl=False, key_prefix=prefix,
    )
    fake = _FakeS3()
    store._client = fake
    return store, fake


class MinioStoreTests(unittest.TestCase):
    def test_read_refuses_keys_outside_the_prefix(self) -> None:
        store, fake = _minio()
        fake.objects["other-service/u/g/f.png"] = b"x"
        fake.objects["media/../other-service/u/g/f.png"] = b"x"
        self.assertIsNone(asyncio.run(store.read_bytes("other-service/u/g/f.png")))
        self.assertIsNone(asyncio.run(store.read_bytes("media/../other-service/u/g/f.png")))
        self.assertEqual(fake.threads, [], "a refused key must not reach S3")

    def test_read_checks_the_owner_segment(self) -> None:
        store, _ = _minio()
        key = asyncio.run(store.save_bytes(b"x", user_id="alice", generation_id="g", filename="a.png"))
        self.assertTrue(key.startswith("media/alice/g/"))
        self.assertEqual(asyncio.run(store.read_bytes(key, user_id="alice")), b"x")
        self.assertIsNone(asyncio.run(store.read_bytes(key, user_id="bob")))

    def test_save_refuses_traversal_in_ids(self) -> None:
        store, fake = _minio()
        with self.assertRaises(ValueError):
            asyncio.run(store.save_bytes(b"x", user_id="..", generation_id="g", filename="a.png"))
        self.assertEqual(fake.objects, {})

    def test_boto3_calls_run_off_the_event_loop_thread(self) -> None:
        store, fake = _minio()

        async def go() -> threading.Thread:
            loop_thread = threading.current_thread()
            key = await store.save_bytes(b"x", user_id="u", generation_id="g", filename="a.png")
            await store.read_bytes(key)
            await store.delete(key)
            return loop_thread

        loop_thread = asyncio.run(go())
        self.assertTrue(fake.threads)
        self.assertNotIn(loop_thread, fake.threads)

    def test_bucket_is_checked_once(self) -> None:
        store, fake = _minio()
        for _ in range(3):
            asyncio.run(store.save_bytes(b"x", user_id="u", generation_id="g", filename="a.png"))
        self.assertEqual(fake.head_calls, 1)

    def test_losing_a_bucket_creation_race_is_not_a_failure(self) -> None:
        from botocore.exceptions import ClientError

        store, fake = _minio()

        def missing(**kw):
            raise ClientError({"Error": {"Code": "404"}}, "HeadBucket")

        def already_created(**kw):
            raise ClientError({"Error": {"Code": "BucketAlreadyOwnedByYou"}}, "CreateBucket")

        fake.head_bucket = missing
        fake.create_bucket = already_created
        rel = asyncio.run(store.save_bytes(b"x", user_id="u", generation_id="g", filename="a.png"))
        self.assertTrue(rel)
        self.assertEqual(len(fake.objects), 1)

    def test_other_bucket_creation_errors_still_raise(self) -> None:
        from botocore.exceptions import ClientError

        store, fake = _minio()

        def missing(**kw):
            raise ClientError({"Error": {"Code": "404"}}, "HeadBucket")

        def denied(**kw):
            raise ClientError({"Error": {"Code": "AccessDenied"}}, "CreateBucket")

        fake.head_bucket = missing
        fake.create_bucket = denied
        with self.assertRaises(ClientError):
            store.ensure_dirs()


if __name__ == "__main__":
    unittest.main()
