import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from starlette.responses import Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import require_user
from app.db import get_db
from app.models.orm import Generation, MediaAsset
from app.models.schemas import AuthContext, GenerationStatusResponse, MediaAssetOut
from app.storage import media_store

log = logging.getLogger("rag-chat.media")

router = APIRouter()


def _asset_out(a: MediaAsset) -> MediaAssetOut:
    return MediaAssetOut(
        id=a.id,
        media_type=a.media_type,
        mime_type=a.mime_type,
        filename=a.filename,
        url=f"/api/media/{a.id}",
        width=a.width,
        height=a.height,
        duration_seconds=a.duration_seconds,
        created_at=a.created_at,
    )


@router.get("/api/media/{asset_id}")
async def serve_media(
    asset_id: str,
    auth: AuthContext = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    """Serve a generated media file (auth-gated)."""
    stmt = (
        select(MediaAsset)
        .where(MediaAsset.id == asset_id)
        .where(MediaAsset.user_id == auth.user_id)
    )
    res = await db.execute(stmt)
    asset = res.scalar_one_or_none()
    if not asset:
        raise HTTPException(404, "Media asset not found")

    path = media_store.resolve_path(asset.file_path)
    if path is not None:
        return FileResponse(
            path,
            media_type=asset.mime_type,
            filename=asset.filename,
        )

    blob = await media_store.read_bytes(asset.file_path)
    if blob is None:
        raise HTTPException(404, "Media file missing from storage")
    return Response(content=blob, media_type=asset.mime_type)


@router.get("/api/generations/{generation_id}", response_model=GenerationStatusResponse)
async def get_generation_status(
    generation_id: str,
    auth: AuthContext = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    """Poll status of an async generation job."""
    stmt = (
        select(Generation)
        .where(Generation.id == generation_id)
        .where(Generation.user_id == auth.user_id)
    )
    res = await db.execute(stmt)
    gen = res.scalar_one_or_none()
    if not gen:
        raise HTTPException(404, "Generation not found")

    asset_stmt = select(MediaAsset).where(MediaAsset.generation_id == generation_id)
    asset_res = await db.execute(asset_stmt)
    assets = asset_res.scalars().all()

    return GenerationStatusResponse(
        id=gen.id,
        status=gen.status,
        provider=gen.provider_name,
        capability=gen.capability,
        media=[_asset_out(a) for a in assets],
        error=gen.error_message,
        created_at=gen.created_at,
        completed_at=gen.completed_at,
    )


@router.get("/api/generations", response_model=List[GenerationStatusResponse])
async def list_generations(
    auth: AuthContext = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    """List recent generations for the current user."""
    stmt = (
        select(Generation)
        .where(Generation.user_id == auth.user_id)
        .order_by(Generation.created_at.desc())
        .limit(50)
    )
    res = await db.execute(stmt)
    gens = res.scalars().all()

    out = []
    for gen in gens:
        asset_stmt = select(MediaAsset).where(MediaAsset.generation_id == gen.id)
        asset_res = await db.execute(asset_stmt)
        assets = asset_res.scalars().all()
        out.append(GenerationStatusResponse(
            id=gen.id,
            status=gen.status,
            provider=gen.provider_name,
            capability=gen.capability,
            media=[_asset_out(a) for a in assets],
            error=gen.error_message,
            created_at=gen.created_at,
            completed_at=gen.completed_at,
        ))
    return out
