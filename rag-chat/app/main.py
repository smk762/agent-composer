from fastapi import FastAPI

from app.db import Base, engine  # noqa: F401 – ensure Base is loaded
import app.models.orm  # noqa: F401 – register all ORM models with Base.metadata
import app.providers  # noqa: F401 – auto-register all provider adapters
from app.metrics import MetricsMiddleware, metrics_router

app = FastAPI(title="RAG Chat API")
app.add_middleware(MetricsMiddleware)

from app.routers import chat, api_keys, ui, generation, media, providers as providers_router, pipeline, repair, runtime, guard, voice  # noqa: E402

app.include_router(chat.router)
app.include_router(api_keys.router)
app.include_router(ui.router)
app.include_router(generation.router)
app.include_router(media.router)
app.include_router(providers_router.router)
app.include_router(pipeline.router)
app.include_router(repair.router)
app.include_router(runtime.router)
app.include_router(guard.router)
app.include_router(voice.router)
app.include_router(metrics_router)
