"""Image / video generation endpoints."""

import json
import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import require_user
from app.config import UTC
from app.db import get_db
from app.models.orm import Generation, MediaAsset
from app.models.schemas import (
    AuthContext, GenerationResponse, ImageGenerateRequest,
    MediaAssetOut, VideoGenerateRequest,
)
from app.providers.base import Capability
from app.providers.registry import ProviderRegistry
from app.storage import media_store

log = logging.getLogger("rag-chat.generation")

router = APIRouter(prefix="/api")


def _asset_out(a: MediaAsset) -> MediaAssetOut:
    return MediaAssetOut(
        id=a.id, media_type=a.media_type, mime_type=a.mime_type,
        filename=a.filename, url=f"/api/media/{a.id}",
        width=a.width, height=a.height,
        duration_seconds=a.duration_seconds, created_at=a.created_at,
    )


# ---- Image generation ----

@router.post("/generate/image", response_model=GenerationResponse)
async def generate_image(
    req: ImageGenerateRequest,
    auth: AuthContext = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    """Generate image(s) from a text prompt."""
    gen = Generation(
        user_id=auth.user_id,
        provider_name=req.provider or "auto",
        capability=Capability.IMAGE_GENERATE.value,
        status="running",
        request_params_json=req.model_dump_json(),
    )
    db.add(gen)
    await db.commit()
    await db.refresh(gen)

    try:
        provider = await ProviderRegistry.get_image_provider(db, auth.user_id, req.provider)
        result = await provider.generate_image(
            req.prompt,
            negative_prompt=req.negative_prompt,
            model=req.model,
            size=req.size,
            n=req.n,
            style=req.style,
        )
    except LookupError as e:
        gen.status = "failed"
        gen.error_message = str(e)
        gen.completed_at = datetime.now(tz=UTC)
        await db.commit()
        raise HTTPException(400, str(e))
    except Exception as e:
        gen.status = "failed"
        gen.error_message = str(e)
        gen.completed_at = datetime.now(tz=UTC)
        await db.commit()
        log.error("generate_image failed: %s", e, exc_info=True)
        raise HTTPException(502, f"Generation failed: {e}")

    media_store.ensure_dirs()
    assets = []
    for gm in result.media:
        rel_path = await media_store.save_bytes(
            gm.data, user_id=auth.user_id, generation_id=gen.id, filename=gm.filename,
        )
        asset = MediaAsset(
            user_id=auth.user_id, generation_id=gen.id,
            media_type="image", mime_type=gm.mime_type,
            filename=gm.filename, file_path=rel_path,
            file_size_bytes=len(gm.data), width=gm.width, height=gm.height,
        )
        db.add(asset)
        assets.append(asset)

    gen.status = "completed"
    gen.provider_name = result.provider
    gen.completed_at = datetime.now(tz=UTC)
    gen.result_json = json.dumps({"model": result.model, "count": len(assets)})
    await db.commit()

    for a in assets:
        await db.refresh(a)

    return GenerationResponse(
        id=gen.id, status="completed",
        media=[_asset_out(a) for a in assets],
        provider=result.provider, model=result.model,
    )


# ---- Image editing / inpainting ----

@router.post("/edit/image", response_model=GenerationResponse)
async def edit_image(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    mask: Optional[UploadFile] = File(None),
    provider: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    size: Optional[str] = Form(None),
    auth: AuthContext = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    """Edit an image (inpainting / search-and-replace)."""
    image_bytes = await image.read()
    mask_bytes = await mask.read() if mask else None

    gen = Generation(
        user_id=auth.user_id,
        provider_name=provider or "auto",
        capability=Capability.IMAGE_EDIT.value,
        status="running",
        request_params_json=json.dumps({"prompt": prompt, "provider": provider, "model": model}),
    )
    db.add(gen)
    await db.commit()
    await db.refresh(gen)

    try:
        prov = await ProviderRegistry.get_provider(db, auth.user_id, Capability.IMAGE_EDIT, provider)
        from app.providers.base import ImageProvider
        assert isinstance(prov, ImageProvider)
        result = await prov.edit_image(prompt, image_bytes, mask=mask_bytes, model=model, size=size)
    except LookupError as e:
        gen.status = "failed"
        gen.error_message = str(e)
        gen.completed_at = datetime.now(tz=UTC)
        await db.commit()
        raise HTTPException(400, str(e))
    except Exception as e:
        gen.status = "failed"
        gen.error_message = str(e)
        gen.completed_at = datetime.now(tz=UTC)
        await db.commit()
        log.error("edit_image failed: %s", e, exc_info=True)
        raise HTTPException(502, f"Edit failed: {e}")

    media_store.ensure_dirs()
    assets = []
    for gm in result.media:
        rel_path = await media_store.save_bytes(
            gm.data, user_id=auth.user_id, generation_id=gen.id, filename=gm.filename,
        )
        asset = MediaAsset(
            user_id=auth.user_id, generation_id=gen.id,
            media_type="image", mime_type=gm.mime_type,
            filename=gm.filename, file_path=rel_path,
            file_size_bytes=len(gm.data), width=gm.width, height=gm.height,
        )
        db.add(asset)
        assets.append(asset)

    gen.status = "completed"
    gen.provider_name = result.provider
    gen.completed_at = datetime.now(tz=UTC)
    await db.commit()
    for a in assets:
        await db.refresh(a)

    return GenerationResponse(
        id=gen.id, status="completed",
        media=[_asset_out(a) for a in assets],
        provider=result.provider, model=result.model,
    )


# ---- Video generation ----

@router.post("/generate/video", response_model=GenerationResponse)
async def generate_video(
    prompt: str = Form(...),
    image: Optional[UploadFile] = File(None),
    provider: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    duration: Optional[float] = Form(None),
    aspect_ratio: Optional[str] = Form(None),
    auth: AuthContext = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    """Generate a video from a text prompt (optionally with a first-frame image)."""
    image_bytes = await image.read() if image else None

    gen = Generation(
        user_id=auth.user_id,
        provider_name=provider or "auto",
        capability=Capability.VIDEO_GENERATE.value,
        status="running",
        request_params_json=json.dumps({
            "prompt": prompt, "provider": provider, "model": model,
            "duration": duration, "aspect_ratio": aspect_ratio,
        }),
    )
    db.add(gen)
    await db.commit()
    await db.refresh(gen)

    try:
        prov = await ProviderRegistry.get_video_provider(db, auth.user_id, provider)
        result = await prov.generate_video(
            prompt, image=image_bytes, model=model,
            duration=duration, aspect_ratio=aspect_ratio,
        )
    except LookupError as e:
        gen.status = "failed"
        gen.error_message = str(e)
        gen.completed_at = datetime.now(tz=UTC)
        await db.commit()
        raise HTTPException(400, str(e))
    except Exception as e:
        gen.status = "failed"
        gen.error_message = str(e)
        gen.completed_at = datetime.now(tz=UTC)
        await db.commit()
        log.error("generate_video failed: %s", e, exc_info=True)
        raise HTTPException(502, f"Video generation failed: {e}")

    media_store.ensure_dirs()
    assets = []
    for gm in result.media:
        rel_path = await media_store.save_bytes(
            gm.data, user_id=auth.user_id, generation_id=gen.id, filename=gm.filename,
        )
        asset = MediaAsset(
            user_id=auth.user_id, generation_id=gen.id,
            media_type="video", mime_type=gm.mime_type,
            filename=gm.filename, file_path=rel_path,
            file_size_bytes=len(gm.data),
            duration_seconds=gm.duration_seconds,
        )
        db.add(asset)
        assets.append(asset)

    gen.status = "completed"
    gen.provider_name = result.provider
    gen.completed_at = datetime.now(tz=UTC)
    gen.result_json = json.dumps({"model": result.model, "count": len(assets)})
    await db.commit()
    for a in assets:
        await db.refresh(a)

    return GenerationResponse(
        id=gen.id, status="completed",
        media=[_asset_out(a) for a in assets],
        provider=result.provider, model=result.model,
    )
