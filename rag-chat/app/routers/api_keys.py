"""API key management routes: list, create (returns the secret once) and revoke the current user's API keys."""

from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import UTC
from app.db import get_db
from app.auth import issue_api_key, require_user
from app.models.orm import ApiKey
from app.models.schemas import ApiKeyCreate, ApiKeyOut, ApiKeySecret, AuthContext

router = APIRouter()


def _as_out(k: ApiKey) -> ApiKeyOut:
    return ApiKeyOut(
        id=k.id, name=k.name, prefix=k.prefix,
        created_at=k.created_at, last_used_at=k.last_used_at, revoked_at=k.revoked_at,
    )


@router.get("/api-keys", response_model=List[ApiKeyOut])
async def list_api_keys(
    auth: AuthContext = Depends(require_user), db: AsyncSession = Depends(get_db)
):
    stmt = select(ApiKey).where(ApiKey.user_id == auth.user_id).order_by(ApiKey.created_at.desc())
    res = await db.execute(stmt)
    return [_as_out(k) for k in res.scalars().all()]


@router.post("/api-keys", response_model=ApiKeySecret)
async def create_api_key(
    body: ApiKeyCreate,
    auth: AuthContext = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    rec, secret = await issue_api_key(db, auth.user_id, body.name)
    out = _as_out(rec)
    return ApiKeySecret(**out.model_dump(), secret=secret)


@router.delete("/api-keys/{key_id}")
async def revoke_api_key(
    key_id: str,
    auth: AuthContext = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(ApiKey).where(ApiKey.id == key_id).where(ApiKey.user_id == auth.user_id)
    res = await db.execute(stmt)
    rec = res.scalar_one_or_none()
    if not rec:
        raise HTTPException(404, "Not found")
    if rec.revoked_at:
        return {"ok": True, "revoked": True}
    rec.revoked_at = datetime.now(tz=UTC)
    await db.commit()
    return {"ok": True, "revoked": True}
