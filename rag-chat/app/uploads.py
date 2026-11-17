"""Bounded reads of multipart uploads, shared by the routers that buffer them."""

from fastapi import HTTPException, UploadFile


async def read_upload(upload: UploadFile, field: str, limit: int) -> bytes:
    """The upload's bytes, read in chunks: past ``limit`` it is a 413 without reading the rest."""
    if upload.size is not None and upload.size > limit:
        raise HTTPException(413, f"{field} is larger than {limit} bytes")
    chunks, total = [], 0
    while chunk := await upload.read(1024 * 1024):
        total += len(chunk)
        if total > limit:
            raise HTTPException(413, f"{field} is larger than {limit} bytes")
        chunks.append(chunk)
    return b"".join(chunks)
