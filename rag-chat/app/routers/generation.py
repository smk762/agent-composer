"""Image / video generation endpoints."""

import json
import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import require_user
from app.config import (
    GENERATION_UPLOAD_MAX_BYTES,
    INFERENCE_WORKER_CONCURRENCY,
    OOM_WAIT_INTERVAL_SECONDS,
    OOM_WAIT_MAX_SECONDS,
    UTC,
)
from app.db import get_db
from app.inference_retry import InferenceWorkers, RetryableOomError, run_with_oom_wait
from app.metrics import observe_gpu_oom
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
_RETRYABLE_OOM_JOB_ERROR = "retryable: CUDA out of memory after waiting for capacity"
_inference_workers = InferenceWorkers(INFERENCE_WORKER_CONCURRENCY)


def _asset_out(a: MediaAsset) -> MediaAssetOut:
    return MediaAssetOut(
        id=a.id, media_type=a.media_type, mime_type=a.mime_type,
        filename=a.filename, url=f"/api/media/{a.id}",
        width=a.width, height=a.height,
        duration_seconds=a.duration_seconds, created_at=a.created_at,
    )


async def _read_upload(upload: UploadFile, field: str, max_bytes: Optional[int] = None) -> bytes:
    """The upload's bytes, read in chunks: past ``max_bytes`` it is a 413 without reading the rest."""
    limit = GENERATION_UPLOAD_MAX_BYTES if max_bytes is None else max_bytes
    if upload.size is not None and upload.size > limit:
        raise HTTPException(413, f"{field} is larger than {limit} bytes")
    chunks, total = [], 0
    while chunk := await upload.read(1024 * 1024):
        total += len(chunk)
        if total > limit:
            raise HTTPException(413, f"{field} is larger than {limit} bytes")
        chunks.append(chunk)
    return b"".join(chunks)


async def _store_results(db: AsyncSession, gen: Generation, result, *, user_id: str, media_type: str,
                         record_result: bool = True) -> list:
    """Save the provider's media and mark ``gen`` completed.

    If saving or committing fails, the files written so far are deleted and ``gen`` is marked
    failed (500), so a generation never stays ``running`` with orphaned files."""
    gen_id = gen.id  # read before a rollback expires ``gen``
    saved: list[str] = []
    try:
        media_store.ensure_dirs()
        assets = []
        for gm in result.media:
            rel_path = await media_store.save_bytes(
                gm.data, user_id=user_id, generation_id=gen.id, filename=gm.filename,
            )
            saved.append(rel_path)
            if media_type == "video":
                extra = {"duration_seconds": gm.duration_seconds}
            else:
                extra = {"width": gm.width, "height": gm.height}
            asset = MediaAsset(
                user_id=user_id, generation_id=gen.id,
                media_type=media_type, mime_type=gm.mime_type,
                filename=gm.filename, file_path=rel_path,
                file_size_bytes=len(gm.data), **extra,
            )
            db.add(asset)
            assets.append(asset)

        gen.status = "completed"
        gen.provider_name = result.provider
        gen.completed_at = datetime.now(tz=UTC)
        if record_result:
            gen.result_json = json.dumps({"model": result.model, "count": len(assets)})
        await db.commit()
        for a in assets:
            await db.refresh(a)
        return assets
    except Exception as e:
        log.error("generation %s: storing results failed: %s", gen_id, e, exc_info=True)
        await db.rollback()
        for rel_path in saved:
            try:
                await media_store.delete(rel_path)
            except Exception as cleanup_err:
                log.warning("generation %s: could not delete %s: %s", gen_id, rel_path, cleanup_err)
        gen.status = "failed"
        gen.error_message = f"storing results failed: {e}"
        gen.completed_at = datetime.now(tz=UTC)
        await db.commit()
        raise HTTPException(500, f"Generation succeeded but storing its results failed: {e}")


async def _run_inference_with_oom_retry(operation, *, metric_operation: str):
    return await _inference_workers.run(
        lambda: run_with_oom_wait(
            operation,
            on_oom=lambda: observe_gpu_oom(operation=metric_operation),
            interval_s=OOM_WAIT_INTERVAL_SECONDS,
            max_wait_s=OOM_WAIT_MAX_SECONDS,
        )
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
        result = await _run_inference_with_oom_retry(
            lambda: provider.generate_image(
                req.prompt,
                negative_prompt=req.negative_prompt,
                model=req.model,
                size=req.size,
                n=req.n,
                style=req.style,
            ),
            metric_operation="image_generate",
        )
    except LookupError as e:
        gen.status = "failed"
        gen.error_message = str(e)
        gen.completed_at = datetime.now(tz=UTC)
        await db.commit()
        raise HTTPException(400, str(e))
    except RetryableOomError as e:
        gen.status = "failed"
        gen.error_message = _RETRYABLE_OOM_JOB_ERROR
        gen.completed_at = datetime.now(tz=UTC)
        await db.commit()
        raise HTTPException(503, str(e))
    except Exception as e:
        gen.status = "failed"
        gen.error_message = str(e)
        gen.completed_at = datetime.now(tz=UTC)
        await db.commit()
        log.error("generate_image failed: %s", e, exc_info=True)
        raise HTTPException(502, f"Generation failed: {e}")

    assets = await _store_results(db, gen, result, user_id=auth.user_id, media_type="image")

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
    image_bytes = await _read_upload(image, "image")
    mask_bytes = await _read_upload(mask, "mask") if mask else None

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
        result = await _run_inference_with_oom_retry(
            lambda: prov.edit_image(prompt, image_bytes, mask=mask_bytes, model=model, size=size),
            metric_operation="image_edit",
        )
    except LookupError as e:
        gen.status = "failed"
        gen.error_message = str(e)
        gen.completed_at = datetime.now(tz=UTC)
        await db.commit()
        raise HTTPException(400, str(e))
    except RetryableOomError as e:
        gen.status = "failed"
        gen.error_message = _RETRYABLE_OOM_JOB_ERROR
        gen.completed_at = datetime.now(tz=UTC)
        await db.commit()
        raise HTTPException(503, str(e))
    except Exception as e:
        gen.status = "failed"
        gen.error_message = str(e)
        gen.completed_at = datetime.now(tz=UTC)
        await db.commit()
        log.error("edit_image failed: %s", e, exc_info=True)
        raise HTTPException(502, f"Edit failed: {e}")

    assets = await _store_results(db, gen, result, user_id=auth.user_id, media_type="image",
                                  record_result=False)

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
    image_bytes = await _read_upload(image, "image") if image else None

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
        result = await _run_inference_with_oom_retry(
            lambda: prov.generate_video(
                prompt, image=image_bytes, model=model,
                duration=duration, aspect_ratio=aspect_ratio,
            ),
            metric_operation="video_generate",
        )
    except LookupError as e:
        gen.status = "failed"
        gen.error_message = str(e)
        gen.completed_at = datetime.now(tz=UTC)
        await db.commit()
        raise HTTPException(400, str(e))
    except RetryableOomError as e:
        gen.status = "failed"
        gen.error_message = _RETRYABLE_OOM_JOB_ERROR
        gen.completed_at = datetime.now(tz=UTC)
        await db.commit()
        raise HTTPException(503, str(e))
    except Exception as e:
        gen.status = "failed"
        gen.error_message = str(e)
        gen.completed_at = datetime.now(tz=UTC)
        await db.commit()
        log.error("generate_video failed: %s", e, exc_info=True)
        raise HTTPException(502, f"Video generation failed: {e}")

    assets = await _store_results(db, gen, result, user_id=auth.user_id, media_type="video")

    return GenerationResponse(
        id=gen.id, status="completed",
        media=[_asset_out(a) for a in assets],
        provider=result.provider, model=result.model,
    )
