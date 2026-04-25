import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException

from .classifier import Classifier
from .model import ModelManager
from .schemas import (ClassifyRequest, ClassifyResponse, EmbedRequest,
                      EmbedResponse, LabelUpdate, ModelStatus)

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

manager = ModelManager()
classifier = Classifier(manager)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if int(os.environ.get("MODERNBERT_PRELOAD", "0")):
        try:
            await manager.ensure_ready()
            await classifier._embed_labels()
            logger.info("Preload complete")
        except Exception as exc:
            logger.warning("Preload failed (%s) — will load on first request", exc)
    yield
    await manager.unload()


app = FastAPI(title="ModernBERT Classifier", lifespan=lifespan)


@app.get("/health")
async def health():
    # Always 200; callers inspect model_state to know if inference is ready.
    return {"status": "ok", "model_state": manager.state, "error": manager.error}


@app.get("/model/status", response_model=ModelStatus)
async def model_status():
    return ModelStatus(
        state=manager.state,
        model_id=manager.model_id,
        device=manager.current_device(),
        last_used=manager._last_used or None,
        keep_alive_gpu=manager.keep_alive_gpu,
        keep_alive_cpu=manager.keep_alive_cpu,
        labels=classifier.labels,
        task=manager.task,
        error=manager.error,
        gpu_capable=manager.effective_device() == "cuda",
        configured_device=manager._target_device,
    )


@app.post("/model/load")
async def load_model():
    await manager.ensure_ready()
    return {"state": manager.state, "device": manager.current_device()}


@app.post("/model/evict")
async def evict_model():
    """GPU → CPU without full unload."""
    await manager.evict()
    return {"state": manager.state}


@app.post("/model/unload")
async def unload_model():
    """Full unload — frees VRAM and RAM."""
    await manager.unload()
    return {"state": manager.state}


@app.get("/labels")
async def get_labels():
    return {"labels": classifier.labels}


@app.put("/labels")
async def update_labels(body: LabelUpdate):
    """Hot-swap the label set; re-embeds prototypes if model is resident."""
    await classifier.update_labels(body.labels)
    return {
        "labels": classifier.labels,
        "embeddings_ready": classifier._label_embeddings is not None,
    }


@app.post("/classify", response_model=ClassifyResponse)
async def classify(req: ClassifyRequest):
    try:
        return await classifier.classify(req)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest):
    try:
        return await classifier.embed(req)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=7998, reload=False)
