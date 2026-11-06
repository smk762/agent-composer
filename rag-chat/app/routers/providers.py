"""Provider management endpoints — list, configure, and test providers."""

import json
import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

import app.providers  # noqa: F401 – auto-register all providers
from app.auth import require_user
from app.config import UTC
from app.db import get_db
from app.models.orm import ProviderConfig
from app.models.schemas import AuthContext, ProviderConfigCreate, ProviderConfigOut, ProviderInfo
from app.providers._crypto import encrypt_value
from app.providers.registry import ProviderRegistry, _PROVIDER_CLASSES

log = logging.getLogger("rag-chat.providers")

router = APIRouter(prefix="/api/providers")


@router.get("", response_model=List[ProviderInfo])
async def list_providers(
    auth: AuthContext = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    """Return all known providers, their capabilities, and whether they're configured."""
    infos = await ProviderRegistry.list_providers(db, auth.user_id)
    return [ProviderInfo(**i) for i in infos]


@router.get("/{provider_name}", response_model=ProviderConfigOut)
async def get_provider_config(
    provider_name: str,
    auth: AuthContext = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    """Get the current configuration for a specific provider."""
    stmt = (
        select(ProviderConfig)
        .where(ProviderConfig.user_id == auth.user_id)
        .where(ProviderConfig.provider_name == provider_name)
    )
    res = await db.execute(stmt)
    pc = res.scalar_one_or_none()
    if not pc:
        raise HTTPException(404, f"Provider '{provider_name}' not configured")

    settings = json.loads(pc.settings_json) if pc.settings_json else None
    return ProviderConfigOut(
        id=pc.id, provider_name=pc.provider_name,
        has_api_key=bool(pc.encrypted_api_key),
        settings=settings, enabled=pc.enabled,
        created_at=pc.created_at, updated_at=pc.updated_at,
    )


@router.post("/{provider_name}/configure", response_model=ProviderConfigOut)
async def configure_provider(
    provider_name: str,
    body: ProviderConfigCreate,
    auth: AuthContext = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    """Store (or update) API key and settings for a provider."""
    if provider_name not in _PROVIDER_CLASSES:
        raise HTTPException(400, f"Unknown provider: {provider_name}")

    stmt = (
        select(ProviderConfig)
        .where(ProviderConfig.user_id == auth.user_id)
        .where(ProviderConfig.provider_name == provider_name)
    )
    res = await db.execute(stmt)
    pc = res.scalar_one_or_none()

    enc_key = encrypt_value(body.api_key) if body.api_key else None
    settings_json = json.dumps(body.settings) if body.settings else None

    if pc:
        if enc_key is not None:
            pc.encrypted_api_key = enc_key
        if settings_json is not None:
            pc.settings_json = settings_json
        pc.enabled = body.enabled
    else:
        pc = ProviderConfig(
            user_id=auth.user_id,
            provider_name=provider_name,
            encrypted_api_key=enc_key,
            settings_json=settings_json,
            enabled=body.enabled,
        )
        db.add(pc)

    await db.commit()
    await db.refresh(pc)

    settings = json.loads(pc.settings_json) if pc.settings_json else None
    return ProviderConfigOut(
        id=pc.id, provider_name=pc.provider_name,
        has_api_key=bool(pc.encrypted_api_key),
        settings=settings, enabled=pc.enabled,
        created_at=pc.created_at, updated_at=pc.updated_at,
    )


@router.post("/{provider_name}/test")
async def test_provider(
    provider_name: str,
    auth: AuthContext = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    """Test if the stored credentials for a provider are valid."""
    if provider_name not in _PROVIDER_CLASSES:
        raise HTTPException(400, f"Unknown provider: {provider_name}")

    stmt = (
        select(ProviderConfig)
        .where(ProviderConfig.user_id == auth.user_id)
        .where(ProviderConfig.provider_name == provider_name)
    )
    res = await db.execute(stmt)
    pc = res.scalar_one_or_none()
    if not pc or not pc.encrypted_api_key:
        return {"ok": False, "error": "No API key configured"}

    from app.providers._crypto import decrypt_value
    api_key = decrypt_value(pc.encrypted_api_key)
    settings = json.loads(pc.settings_json) if pc.settings_json else {}

    cls = _PROVIDER_CLASSES[provider_name]
    instance = cls(api_key=api_key, **settings)

    try:
        valid = await instance.validate_config()
    except Exception as e:
        return {"ok": False, "error": str(e)}

    return {"ok": valid, "error": None if valid else "Validation failed"}


@router.delete("/{provider_name}")
async def remove_provider_config(
    provider_name: str,
    auth: AuthContext = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    """Remove a provider configuration (revoke stored API key)."""
    stmt = (
        select(ProviderConfig)
        .where(ProviderConfig.user_id == auth.user_id)
        .where(ProviderConfig.provider_name == provider_name)
    )
    res = await db.execute(stmt)
    pc = res.scalar_one_or_none()
    if not pc:
        raise HTTPException(404, f"Provider '{provider_name}' not configured")

    await db.delete(pc)
    await db.commit()
    return {"ok": True}
