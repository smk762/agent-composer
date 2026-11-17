"""Media asset storage backends (local filesystem or MinIO/S3, chosen by MEDIA_BACKEND) and the module-level media_store instance.

Every stored path/key is ``{user_id}/{generation_id}/{name}``. ``_confine`` is the one check both
backends run on that relative path, when saving and when reading or deleting.
"""

import asyncio
import re
import uuid
from pathlib import Path
from typing import Optional

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from app.config import (
    MEDIA_BACKEND,
    MEDIA_DIR,
    MEDIA_S3_ACCESS_KEY_ID,
    MEDIA_S3_BUCKET,
    MEDIA_S3_ENDPOINT_URL,
    MEDIA_S3_PREFIX,
    MEDIA_S3_REGION,
    MEDIA_S3_SECRET_ACCESS_KEY,
    MEDIA_S3_USE_SSL,
)


def _safe_filename(filename: str) -> str:
    """A usable leaf filename: never empty, ``.``, ``..``, or carrying a separator or NUL."""
    name = Path((filename or "").strip()).name
    if name in ("", ".", "..") or "\\" in name or "\x00" in name:
        return "asset.bin"
    return name


_SAFE_EXT = re.compile(r"^\.[A-Za-z0-9]{1,16}$")


def _safe_ext(filename: str) -> str:
    ext = Path(_safe_filename(filename)).suffix
    return ext if _SAFE_EXT.match(ext) else ""


def _confine(rel_path: str, *, user_id: Optional[str] = None) -> Optional[str]:
    """Return ``rel_path`` if it is ``user/generation/name`` made of plain path components, else None.

    Rejects absolute paths, ``.``/``..``/empty segments, backslashes and NUL, so a stored path can
    never leave the store's root (or key prefix). With ``user_id`` set, the first segment must match it.
    """
    if not isinstance(rel_path, str):
        return None
    parts = rel_path.split("/")
    if len(parts) != 3:
        return None
    for part in parts:
        if part in ("", ".", "..") or "\\" in part or "\x00" in part:
            return None
    if user_id is not None and parts[0] != user_id:
        return None
    return rel_path


def _rel_path(*, user_id: str, generation_id: str, name: str) -> str:
    rel = _confine(f"{user_id}/{generation_id}/{name}")
    if rel is None:
        raise ValueError("user_id and generation_id must each be a single safe path component")
    return rel


class LocalMediaStore:
    def __init__(self, media_dir: str):
        self._media_dir = Path(media_dir)

    def ensure_dirs(self) -> None:
        self._media_dir.mkdir(parents=True, exist_ok=True)

    def _full_path(self, rel_path: str, *, user_id: Optional[str] = None) -> Optional[Path]:
        """``rel_path`` under the media dir, or None if it is malformed or escapes it (symlinks resolved)."""
        if _confine(rel_path, user_id=user_id) is None:
            return None
        root = self._media_dir.resolve()
        p = (root / rel_path).resolve()
        try:
            p.relative_to(root)
        except ValueError:
            return None
        return p

    async def save_bytes(self, data: bytes, *, user_id: str, generation_id: str, filename: str) -> str:
        self.ensure_dirs()
        rel_path = _rel_path(user_id=user_id, generation_id=generation_id,
                             name=f"{uuid.uuid4().hex}{_safe_ext(filename)}")
        full_path = self._full_path(rel_path)
        if full_path is None:
            raise ValueError("media path escapes the media directory")
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_bytes(data)
        return rel_path

    def resolve_path(self, file_path: str, *, user_id: Optional[str] = None) -> Optional[str]:
        p = self._full_path(file_path, user_id=user_id)
        if p is None or not p.exists() or not p.is_file():
            return None
        return str(p)

    async def read_bytes(self, file_path: str, *, user_id: Optional[str] = None) -> Optional[bytes]:
        p = self.resolve_path(file_path, user_id=user_id)
        if p is None:
            return None
        return Path(p).read_bytes()

    async def delete(self, file_path: str) -> None:
        p = self.resolve_path(file_path)
        if p is not None:
            Path(p).unlink(missing_ok=True)


class MinioMediaStore:
    def __init__(
        self,
        *,
        endpoint_url: str,
        region: str,
        bucket: str,
        access_key_id: str,
        secret_access_key: str,
        use_ssl: bool,
        key_prefix: str,
    ):
        if not endpoint_url or not bucket or not access_key_id or not secret_access_key:
            raise RuntimeError("MinIO media backend requires endpoint, bucket, access key, and secret key")
        self._bucket = bucket
        self._key_prefix = key_prefix.strip("/")
        self._bucket_ready = False
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=region,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            use_ssl=use_ssl,
            config=Config(signature_version="s3v4"),
        )

    def ensure_dirs(self) -> None:
        """Check (or create) the bucket once; later calls make no request. Blocking: call via a thread."""
        if self._bucket_ready:
            return
        try:
            self._client.head_bucket(Bucket=self._bucket)
        except ClientError as e:
            code = str((e.response or {}).get("Error", {}).get("Code", ""))
            if code not in {"404", "NoSuchBucket"}:
                raise
            self._client.create_bucket(Bucket=self._bucket)
        self._bucket_ready = True

    def _object_key(self, *, user_id: str, generation_id: str, filename: str) -> str:
        rel = _rel_path(user_id=user_id, generation_id=generation_id,
                        name=f"{uuid.uuid4().hex}-{_safe_filename(filename)}")
        if self._key_prefix:
            return f"{self._key_prefix}/{rel}"
        return rel

    def _confined_key(self, key: str, *, user_id: Optional[str] = None) -> Optional[str]:
        """``key`` if it lies under this store's prefix and passes ``_confine``, else None."""
        if not isinstance(key, str):
            return None
        rel = key
        if self._key_prefix:
            head = f"{self._key_prefix}/"
            if not key.startswith(head):
                return None
            rel = key[len(head):]
        return key if _confine(rel, user_id=user_id) is not None else None

    def _put(self, key: str, data: bytes) -> None:
        self.ensure_dirs()
        self._client.put_object(Bucket=self._bucket, Key=key, Body=data)

    def _get(self, key: str) -> Optional[bytes]:
        try:
            out = self._client.get_object(Bucket=self._bucket, Key=key)
        except ClientError:
            return None
        return out["Body"].read()

    async def save_bytes(self, data: bytes, *, user_id: str, generation_id: str, filename: str) -> str:
        key = self._object_key(user_id=user_id, generation_id=generation_id, filename=filename)
        await asyncio.to_thread(self._put, key, data)
        return key

    def resolve_path(self, file_path: str, *, user_id: Optional[str] = None) -> Optional[str]:
        # MinIO objects are not local files.
        return None

    async def read_bytes(self, file_path: str, *, user_id: Optional[str] = None) -> Optional[bytes]:
        key = self._confined_key(file_path, user_id=user_id)
        if key is None:
            return None
        return await asyncio.to_thread(self._get, key)

    async def delete(self, file_path: str) -> None:
        key = self._confined_key(file_path)
        if key is not None:
            await asyncio.to_thread(self._client.delete_object, Bucket=self._bucket, Key=key)


def _build_store():
    if MEDIA_BACKEND == "minio":
        return MinioMediaStore(
            endpoint_url=MEDIA_S3_ENDPOINT_URL,
            region=MEDIA_S3_REGION,
            bucket=MEDIA_S3_BUCKET,
            access_key_id=MEDIA_S3_ACCESS_KEY_ID,
            secret_access_key=MEDIA_S3_SECRET_ACCESS_KEY,
            use_ssl=MEDIA_S3_USE_SSL,
            key_prefix=MEDIA_S3_PREFIX,
        )
    return LocalMediaStore(MEDIA_DIR)


media_store = _build_store()
