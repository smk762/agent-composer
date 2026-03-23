"""Auto-register all provider adapters on import."""

from app.providers import openai  # noqa: F401
from app.providers import comfyui  # noqa: F401
from app.providers import replicate  # noqa: F401
from app.providers import stability  # noqa: F401
from app.providers import together  # noqa: F401
