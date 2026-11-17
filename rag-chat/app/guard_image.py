"""Prepare an image for SingGuard on Ollama.

Ollama's vision path is pickier than Pillow: WebP, TIFF, CMYK JPEG and a few
progressive JPEGs fail there. Cortex already downscales, but it leaves a small
original as-is, so cull scans and darkroom image checks share this rewrite.
"""

from __future__ import annotations

import base64
import binascii
import io

from PIL import Image, ImageOps, UnidentifiedImageError

_MAX_IMAGE_BYTES = 12 * 1024 * 1024
# Match cortex's send cap so already-downscaled frames pass through.
_GUARD_MAX_EDGE = 1536
_GUARD_MAX_BYTES = 4 * 1024 * 1024
# Decoded-size budget, checked from the header before any pixels are decoded: a small,
# highly compressed upload can otherwise claim dimensions that take GBs to decode.
# 64 MP covers full-size phone and camera photos and stays under Pillow's bomb limit.
_GUARD_MAX_PIXELS = 64 * 1024 * 1024


def prepare_guard_image(data: bytes) -> bytes:
    """Re-encode *data* as an RGB JPEG (or PNG with alpha) Ollama will open."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            width, height = im.size
            if width * height > _GUARD_MAX_PIXELS:
                raise ValueError(f"image is too large for the guard ({width}x{height} pixels)")
            fmt = (im.format or "").upper()
            if (
                fmt in {"JPEG", "JPG"}
                and im.mode == "RGB"
                and max(im.size) <= _GUARD_MAX_EDGE
                and len(data) <= _GUARD_MAX_BYTES
                # Rotated photos need exif_transpose; re-encoding also drops their EXIF.
                and not im.getexif()
            ):
                return data
            if fmt in {"JPEG", "JPG"}:
                # Let the JPEG decoder scale down by a power of two (never below the edge cap).
                im.draft(None, (_GUARD_MAX_EDGE, _GUARD_MAX_EDGE))
            im.load()
            im.thumbnail((_GUARD_MAX_EDGE, _GUARD_MAX_EDGE))
            out = ImageOps.exif_transpose(im)
            alpha = out.mode in ("RGBA", "LA", "PA") or (out.mode == "P" and "transparency" in out.info)
            if not alpha:
                out = out.convert("RGB")
            elif out.mode in ("P", "PA"):
                # The PNG encoder cannot write PA; RGBA keeps the transparency.
                out = out.convert("RGBA")
            edge = max(out.size)
            while True:
                buf = io.BytesIO()
                if alpha:
                    out.save(buf, "PNG", optimize=True)
                else:
                    out.save(buf, "JPEG", quality=90)
                if buf.tell() <= _GUARD_MAX_BYTES:
                    return buf.getvalue()
                if edge <= 256:
                    raise ValueError("image could not be shrunk under the guard size limit")
                edge = max(256, edge * 3 // 4)
                out.thumbnail((edge, edge))
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ValueError(f"image could not be decoded for the guard: {exc}") from exc


def image_b64(value: str) -> str:
    """Request `image_base64` (raw or data URL) as the bytes Ollama should see."""
    raw = value.strip()
    if not raw:
        return ""
    if raw.startswith("data:"):
        _, _, raw = raw.partition(",")
    raw = "".join(raw.split())
    try:
        data = base64.b64decode(raw, validate=True)
    except binascii.Error as exc:
        raise ValueError("image is not valid base64") from exc
    if not data:
        raise ValueError("image is empty")
    if len(data) > _MAX_IMAGE_BYTES:
        raise ValueError("image is larger than 12 MB")
    return base64.b64encode(prepare_guard_image(data)).decode("ascii")
