from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


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
