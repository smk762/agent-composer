"""Request authentication for rag-chat: resolves the user from a bearer API key, a verified
Cloudflare Access JWT, or the DEV_AUTH_BYPASS dev user.

The ``Cf-Access-Authenticated-User-Email`` header is never trusted on its own: anyone who
reaches the origin without passing Cloudflare (a LAN port, another container) can set it.
The identity comes from the ``email`` claim of the Access JWT (``Cf-Access-Jwt-Assertion``
header, else the ``CF_Authorization`` cookie) after checking its signature against
``https://<CF_ACCESS_TEAM_DOMAIN>/cdn-cgi/access/certs``, its ``aud`` (``CF_ACCESS_AUD``),
issuer and expiry. Without both settings, Access identity is refused.
"""

import asyncio
import hashlib
import logging
import secrets
from datetime import datetime
from functools import lru_cache
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import CF_ACCESS_AUD, CF_ACCESS_TEAM_DOMAIN, DEV_AUTH_BYPASS, DEV_DEFAULT_USER, UTC
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


log = logging.getLogger("rag-chat.auth")

_ACCESS_HEADERS = ("Cf-Access-Jwt-Assertion", "Cf-Access-Authenticated-User-Email")


@lru_cache(maxsize=4)
def _jwks_client(team_domain: str) -> jwt.PyJWKClient:
    # Caches the signing keys and refetches on an unknown kid (key rotation).
    return jwt.PyJWKClient(f"https://{team_domain}/cdn-cgi/access/certs", cache_keys=True, lifespan=3600)


def verify_access_jwt(token: str, *, team_domain: str, audience: str) -> str:
    """Email from a valid Cloudflare Access JWT; raises ``jwt.PyJWTError`` otherwise."""
    signing_key = _jwks_client(team_domain).get_signing_key_from_jwt(token)
    claims = jwt.decode(
        token,
        signing_key.key,
        algorithms=["RS256"],
        audience=audience,
        issuer=f"https://{team_domain}",
        options={"require": ["exp", "iat", "aud", "iss"]},
        leeway=30,
    )
    email = claims.get("email")
    if not isinstance(email, str) or not email.strip():
        raise jwt.InvalidTokenError("Access token has no email claim")
    return email.strip()


async def _access_identity(request: Request) -> Optional[str]:
    """Verified Access email, None when no Access credential was sent; 401 on a bad one."""
    token = _get_header(request, "Cf-Access-Jwt-Assertion") or request.cookies.get("CF_Authorization")
    if not token:
        if _get_header(request, "Cf-Access-Authenticated-User-Email"):
            raise HTTPException(401, "Cloudflare Access email without an Access token")
        return None
    if not (CF_ACCESS_TEAM_DOMAIN and CF_ACCESS_AUD):
        raise HTTPException(401, "Cloudflare Access is not configured (CF_ACCESS_TEAM_DOMAIN, CF_ACCESS_AUD)")
    try:
        # The first call (and a key rotation) fetches the certs over the network.
        return await asyncio.to_thread(
            verify_access_jwt, token, team_domain=CF_ACCESS_TEAM_DOMAIN, audience=CF_ACCESS_AUD
        )
    except jwt.PyJWTError as e:
        log.warning("rejected Cloudflare Access token: %s", e)
        raise HTTPException(401, "Invalid Cloudflare Access token") from None


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

    email = await _access_identity(request)
    if email:
        return AuthContext(user_id=email, via="access")

    if DEV_AUTH_BYPASS:
        dev_user = _get_header(request, "X-Dev-User") or DEV_DEFAULT_USER
        return AuthContext(user_id=dev_user, via="dev")

    raise HTTPException(401, "Authentication required")
