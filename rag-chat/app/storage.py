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
    cleaned = (filename or "").strip()
    if not cleaned:
        return "asset.bin"
    return Path(cleaned).name


class LocalMediaStore:
    def __init__(self, media_dir: str):
        self._media_dir = Path(media_dir)

    def ensure_dirs(self) -> None:
        self._media_dir.mkdir(parents=True, exist_ok=True)

    async def save_bytes(self, data: bytes, *, user_id: str, generation_id: str, filename: str) -> str:
        self.ensure_dirs()
        ext = Path(filename).suffix
        name = f"{uuid.uuid4().hex}{ext}"
        rel_path = f"{user_id}/{generation_id}/{name}"
        full_path = self._media_dir / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_bytes(data)
        return rel_path

    def resolve_path(self, file_path: str) -> Optional[str]:
        p = (self._media_dir / file_path).resolve()
        try:
            p.relative_to(self._media_dir.resolve())
        except ValueError:
            return None
        if not p.exists() or not p.is_file():
            return None
        return str(p)

    async def read_bytes(self, file_path: str) -> Optional[bytes]:
        p = self.resolve_path(file_path)
        if p is None:
            return None
        return Path(p).read_bytes()


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
        try:
            self._client.head_bucket(Bucket=self._bucket)
        except ClientError as e:
            code = str((e.response or {}).get("Error", {}).get("Code", ""))
            if code in {"404", "NoSuchBucket"}:
                self._client.create_bucket(Bucket=self._bucket)
                return
            raise

    def _object_key(self, *, user_id: str, generation_id: str, filename: str) -> str:
        safe_name = _safe_filename(filename)
        key = f"{user_id}/{generation_id}/{uuid.uuid4().hex}-{safe_name}"
        if self._key_prefix:
            return f"{self._key_prefix}/{key}"
        return key

    async def save_bytes(self, data: bytes, *, user_id: str, generation_id: str, filename: str) -> str:
        self.ensure_dirs()
        key = self._object_key(user_id=user_id, generation_id=generation_id, filename=filename)
        self._client.put_object(Bucket=self._bucket, Key=key, Body=data)
        return key

    def resolve_path(self, file_path: str) -> Optional[str]:
        # MinIO objects are not local files.
        return None

    async def read_bytes(self, file_path: str) -> Optional[bytes]:
        try:
            out = self._client.get_object(Bucket=self._bucket, Key=file_path)
        except ClientError:
            return None
        return out["Body"].read()


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
