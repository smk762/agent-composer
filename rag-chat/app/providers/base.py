"""Abstract media-generation provider interfaces (BaseProvider, ImageProvider, VideoProvider), capabilities and result types."""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


_SIZE_RE = re.compile(r"\s*(\d{1,5})\s*(?:[x×]\s*(\d{1,5}))?\s*", re.IGNORECASE)
_MIN_EDGE, _MAX_EDGE = 64, 4096


def parse_size(size: Optional[str], default: int = 1024) -> tuple[int, int]:
    """``WxH`` (case-insensitive) or ``N`` for a square; empty means ``default`` square.

    Anything else, or a side outside 64-4096, is a LookupError, which the generation
    router answers with a 400 rather than letting ``int()`` raise a 500."""
    if not size or not size.strip():
        return default, default
    m = _SIZE_RE.fullmatch(size)
    if not m:
        raise LookupError(f"size must be WIDTHxHEIGHT, got {size!r}")
    w = int(m.group(1))
    h = int(m.group(2)) if m.group(2) else w
    if not (_MIN_EDGE <= w <= _MAX_EDGE and _MIN_EDGE <= h <= _MAX_EDGE):
        raise LookupError(f"size {size!r} is out of range: each side {_MIN_EDGE}-{_MAX_EDGE}")
    return w, h


class Capability(str, Enum):
    TEXT_CHAT = "text_chat"
    TEXT_EMBEDDING = "text_embedding"
    IMAGE_GENERATE = "image_generate"
    IMAGE_EDIT = "image_edit"
    IMAGE_INPAINT = "image_inpaint"
    VIDEO_GENERATE = "video_generate"


@dataclass
class GeneratedMedia:
    """A single piece of generated media returned by a provider."""
    data: bytes
    mime_type: str
    filename: str
    width: Optional[int] = None
    height: Optional[int] = None
    duration_seconds: Optional[float] = None


@dataclass
class GenerationResult:
    """Result from a provider generation call."""
    media: List[GeneratedMedia] = field(default_factory=list)
    model: str = ""
    provider: str = ""
    raw_response: Optional[dict] = None


class BaseProvider(ABC):
    """Abstract base for all AI providers."""

    name: str = ""
    capabilities: List[Capability] = []

    @abstractmethod
    async def validate_config(self) -> bool:
        """Return True if the provider's configuration (API key, URL, etc.) is valid."""
        ...

    def supports(self, capability: Capability) -> bool:
        return capability in self.capabilities

    def available_models(self, capability: Capability) -> List[str]:
        """Return the model identifiers this provider offers for a given capability."""
        return []


class ImageProvider(BaseProvider):
    """Provider that can generate and optionally edit images."""

    @abstractmethod
    async def generate_image(
        self,
        prompt: str,
        *,
        negative_prompt: Optional[str] = None,
        model: Optional[str] = None,
        size: str = "1024x1024",
        n: int = 1,
        style: Optional[str] = None,
    ) -> GenerationResult:
        ...

    async def edit_image(
        self,
        prompt: str,
        image: bytes,
        *,
        mask: Optional[bytes] = None,
        model: Optional[str] = None,
        size: Optional[str] = None,
    ) -> GenerationResult:
        raise NotImplementedError(f"{self.name} does not support image editing")


class VideoProvider(BaseProvider):
    """Provider that can generate video."""

    @abstractmethod
    async def generate_video(
        self,
        prompt: str,
        *,
        image: Optional[bytes] = None,
        model: Optional[str] = None,
        duration: Optional[float] = None,
        aspect_ratio: Optional[str] = None,
    ) -> GenerationResult:
        ...
