"""rag-chat FastAPI application: registers ORM models and providers, adds metrics middleware, and mounts all API and UI routers."""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.db import Base, engine  # noqa: F401 – ensure Base is loaded
import app.models.orm  # noqa: F401 – register all ORM models with Base.metadata
import app.providers  # noqa: F401 – auto-register all provider adapters
from app.metrics import MetricsMiddleware, metrics_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Check the RAG collection's embedder in the background: never block startup on Qdrant.
    from app import retrieval

    async def check():
        status = await retrieval.embedder_status(force=True)
        if status.get("error"):
            logging.getLogger("rag-chat").warning("startup embedder check skipped: %s", status["error"])

    task = asyncio.create_task(check())
    yield
    task.cancel()


app = FastAPI(title="RAG Chat API", lifespan=lifespan)
app.add_middleware(MetricsMiddleware)

from app.routers import chat, api_keys, ui, generation, media, providers as providers_router, pipeline, repair, runtime, guard, voice, search  # noqa: E402

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
app.include_router(search.router)
app.include_router(metrics_router)
