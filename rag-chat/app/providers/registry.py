"""Provider registry: registers provider classes by name and resolves a configured, enabled provider per user and capability using encrypted stored API keys."""

import json
import logging
from typing import Dict, List, Optional, Type

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.providers.base import BaseProvider, Capability, ImageProvider, VideoProvider
from app.models.orm import ProviderConfig
from app.providers._crypto import decrypt_value

log = logging.getLogger("rag-chat.registry")

_PROVIDER_CLASSES: Dict[str, Type[BaseProvider]] = {}


def register_provider(name: str, cls: Type[BaseProvider]):
    """Register a provider class by name so the registry can instantiate it."""
    _PROVIDER_CLASSES[name] = cls


class ProviderRegistry:
    """Resolve which provider to use for a given capability and user."""

    @staticmethod
    async def list_providers(db: AsyncSession, user_id: str) -> List[dict]:
        """Return info about all known providers and whether the user has configured them."""
        stmt = select(ProviderConfig).where(ProviderConfig.user_id == user_id)
        res = await db.execute(stmt)
        configs = {pc.provider_name: pc for pc in res.scalars().all()}

        out = []
        for name, cls in _PROVIDER_CLASSES.items():
            pc = configs.get(name)
            instance = cls.__new__(cls)
            out.append({
                "name": name,
                "capabilities": [c.value for c in getattr(instance, "capabilities", [])],
                "configured": pc is not None and bool(pc.encrypted_api_key),
                "enabled": pc.enabled if pc else False,
                "models": {
                    cap.value: instance.available_models(cap)
                    for cap in getattr(instance, "capabilities", [])
                },
            })
        return out

    @staticmethod
    async def get_provider(
        db: AsyncSession,
        user_id: str,
        capability: Capability,
        provider_name: Optional[str] = None,
    ) -> BaseProvider:
        """
        Instantiate and return a configured provider for the given capability.

        If *provider_name* is specified, use that provider. Otherwise pick the
        first enabled provider that supports the capability, in a fixed order: the one
        configured first (then by name), so the same request always reaches the same backend.
        """
        stmt = select(ProviderConfig).where(ProviderConfig.user_id == user_id)
        if provider_name:
            stmt = stmt.where(ProviderConfig.provider_name == provider_name)
        stmt = stmt.where(ProviderConfig.enabled.is_(True)).order_by(
            ProviderConfig.created_at, ProviderConfig.provider_name, ProviderConfig.id,
        )
        res = await db.execute(stmt)
        configs = res.scalars().all()

        for pc in configs:
            cls = _PROVIDER_CLASSES.get(pc.provider_name)
            if cls is None:
                continue
            dummy = cls.__new__(cls)
            if not dummy.supports(capability):
                continue

            api_key = decrypt_value(pc.encrypted_api_key) if pc.encrypted_api_key else None
            settings = json.loads(pc.settings_json) if pc.settings_json else {}
            # The key comes from the encrypted column only; a stored "api_key" setting would
            # also collide with the explicit argument (TypeError) and lock the provider out.
            if isinstance(settings, dict) and settings.pop("api_key", None) is not None:
                log.warning("Ignoring 'api_key' in settings_json for provider %s", pc.provider_name)
            elif not isinstance(settings, dict):
                settings = {}
            instance = cls(api_key=api_key, **settings)
            return instance

        available = [n for n, c in _PROVIDER_CLASSES.items() if c.__new__(c).supports(capability)]
        raise LookupError(
            f"No configured provider for {capability.value}. "
            f"Available (unconfigured): {available}"
        )

    @staticmethod
    async def get_image_provider(
        db: AsyncSession, user_id: str, provider_name: Optional[str] = None,
    ) -> ImageProvider:
        p = await ProviderRegistry.get_provider(db, user_id, Capability.IMAGE_GENERATE, provider_name)
        if not isinstance(p, ImageProvider):
            raise LookupError(f"Provider {p.name!r} is not an image provider")
        return p

    @staticmethod
    async def get_video_provider(
        db: AsyncSession, user_id: str, provider_name: Optional[str] = None,
    ) -> VideoProvider:
        p = await ProviderRegistry.get_provider(db, user_id, Capability.VIDEO_GENERATE, provider_name)
        if not isinstance(p, VideoProvider):
            raise LookupError(f"Provider {p.name!r} is not a video provider")
        return p
