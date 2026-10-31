import hashlib
import secrets
from datetime import datetime
from typing import Optional

from fastapi import Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import DEV_AUTH_BYPASS, DEV_DEFAULT_USER, UTC
from app.db import get_db
from app.models.orm import ApiKey
from app.models.schemas import AuthContext


def _hash_key(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _generate_secret() -> tuple[str, str]:
    prefix = secrets.token_hex(6)
    secret = f"{prefix}.{secrets.token_urlsafe(24)}"
    return prefix, secret


async def _api_key_from_token(db: AsyncSession, token: str) -> Optional[ApiKey]:
    if not token:
        return None
    if "." in token:
        prefix = token.split(".", 1)[0]
    else:
        prefix = token[:12]
    hashed = _hash_key(token)
    stmt = (
        select(ApiKey)
        .where(ApiKey.prefix == prefix)
        .where(ApiKey.hashed_key == hashed)
        .where(ApiKey.revoked_at.is_(None))
        .limit(1)
    )
    res = await db.execute(stmt)
    obj = res.scalar_one_or_none()
    if obj:
        obj.last_used_at = datetime.now(tz=UTC)
        await db.commit()
    return obj


def _get_header(request: Request, name: str) -> Optional[str]:
    return request.headers.get(name) or request.headers.get(name.lower())


async def require_user(
    request: Request, db: AsyncSession = Depends(get_db)
) -> AuthContext:
    authz = _get_header(request, "Authorization") or ""
    if authz.lower().startswith("bearer "):
        token = authz.split(" ", 1)[1].strip()
        key = await _api_key_from_token(db, token)
        if not key:
            raise HTTPException(401, "Invalid API key")
        return AuthContext(user_id=key.user_id, via="apikey")

    email = _get_header(request, "Cf-Access-Authenticated-User-Email")
    if email:
        return AuthContext(user_id=email, via="access")

    if DEV_AUTH_BYPASS:
        dev_user = _get_header(request, "X-Dev-User") or DEV_DEFAULT_USER
        return AuthContext(user_id=dev_user, via="dev")

    raise HTTPException(401, "Authentication required")
