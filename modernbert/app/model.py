import asyncio
import gc
import logging
import os
import time
from enum import Enum
from typing import Optional

import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


class ModelState(str, Enum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    CPU = "cpu"
    GPU = "gpu"
    ERROR = "error"


class ModelManager:
    """
    Three-state lifecycle: unloaded → cpu → gpu.

    keep_alive_gpu  seconds on GPU after last use before demoting to CPU  (-1 = forever, 0 = immediate)
    keep_alive_cpu  seconds on CPU after GPU eviction before full unload  (-1 = forever, 0 = immediate)
    """

    def __init__(self) -> None:
        self.model_id: str = os.environ.get("MODERNBERT_MODEL", "answerdotai/ModernBERT-base")
        self.task: str = os.environ.get("MODERNBERT_TASK", "zero_shot")
        self._target_device: str = os.environ.get("MODERNBERT_DEVICE", "auto")
        self.keep_alive_gpu: float = float(os.environ.get("MODERNBERT_KEEP_ALIVE_GPU", "300"))
        self.keep_alive_cpu: float = float(os.environ.get("MODERNBERT_KEEP_ALIVE_CPU", "0"))

        self._state: ModelState = ModelState.UNLOADED
        self._error: Optional[str] = None
        self._model = None
        self._tokenizer = None
        self._last_used: float = 0.0
        self._lock: asyncio.Lock = asyncio.Lock()
        self._evict_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------ props

    @property
    def state(self) -> ModelState:
        return self._state

    @property
    def error(self) -> Optional[str]:
        return self._error

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    def effective_device(self) -> str:
        if self._target_device == "cpu":
            return "cpu"
        if self._target_device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self._target_device

    def current_device(self) -> str:
        if self._model is None:
            return "none"
        try:
            return str(next(self._model.parameters()).device)
        except StopIteration:
            return "unknown"

    # ------------------------------------------------------------------ load

    def _load_sync(self, num_labels: int) -> None:
        logger.info("Loading %s (task=%s, fp16)", self.model_id, self.task)
        tok = AutoTokenizer.from_pretrained(self.model_id)
        if self.task == "classify":
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_id, num_labels=num_labels, torch_dtype=torch.float16
            )
        else:
            model = AutoModel.from_pretrained(self.model_id, torch_dtype=torch.float16)
        model.eval()
        self._tokenizer = tok
        self._model = model  # stays on CPU until _promote

    async def _do_load(self, num_labels: int) -> None:
        self._state = ModelState.LOADING
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, lambda: self._load_sync(num_labels))
            self._state = ModelState.CPU
            self._error = None
            logger.info("Model on CPU — %s", self.model_id)
        except Exception as exc:
            self._state = ModelState.ERROR
            self._error = str(exc)
            logger.error("Load failed: %s", exc)
            raise

    # ------------------------------------------------------------------ move

    async def _promote(self) -> None:
        device = self.effective_device()
        if device == "cpu":
            return  # CPU-only config; CPU counts as ready
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self._model.to(device))
        self._state = ModelState.GPU
        logger.debug("Model → GPU")

    async def _demote(self) -> None:
        if self._model is None:
            return
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self._model.to("cpu"))
        torch.cuda.empty_cache()
        self._state = ModelState.CPU
        logger.debug("Model → CPU")

    # ------------------------------------------------------------------ public

    async def ensure_ready(self, num_labels: int = 2) -> None:
        """Guarantee model is on GPU (or CPU if MODERNBERT_DEVICE=cpu) and reset keep-alive."""
        async with self._lock:
            if self._state == ModelState.GPU:
                pass
            elif self._state == ModelState.CPU:
                await self._promote()
            elif self._state in (ModelState.UNLOADED, ModelState.ERROR):
                await self._do_load(num_labels)
                await self._promote()
            # LOADING is serialised by the lock — can't reach here from outside

            self._last_used = time.time()
            self._arm_evict()

    async def evict(self) -> None:
        """Move GPU → CPU without full unload."""
        async with self._lock:
            if self._state == ModelState.GPU:
                await self._demote()

    async def unload(self) -> None:
        """Full unload; frees both VRAM and RAM."""
        async with self._lock:
            if self._evict_task and not self._evict_task.done():
                self._evict_task.cancel()
            if self._model is not None:
                del self._model
                self._model = None
                gc.collect()
                torch.cuda.empty_cache()
            self._state = ModelState.UNLOADED
            logger.info("Model unloaded")

    # ------------------------------------------------------------------ timer

    def _arm_evict(self) -> None:
        if self._evict_task and not self._evict_task.done():
            self._evict_task.cancel()
        self._evict_task = asyncio.create_task(self._eviction_loop())

    async def _eviction_loop(self) -> None:
        try:
            # --- GPU keep-alive ---
            if self.keep_alive_gpu == 0:
                async with self._lock:
                    if self._state == ModelState.GPU:
                        await self._demote()
            elif self.keep_alive_gpu > 0:
                await asyncio.sleep(self.keep_alive_gpu)
                idle = time.time() - self._last_used
                if idle >= self.keep_alive_gpu:
                    async with self._lock:
                        if self._state == ModelState.GPU:
                            await self._demote()
            # keep_alive_gpu < 0 → stay on GPU forever

            # --- CPU keep-alive ---
            if self.keep_alive_cpu == 0:
                await self.unload()
            elif self.keep_alive_cpu > 0:
                await asyncio.sleep(self.keep_alive_cpu)
                if time.time() - self._last_used >= self.keep_alive_gpu + self.keep_alive_cpu:
                    await self.unload()
            # keep_alive_cpu < 0 → stay on CPU forever
        except asyncio.CancelledError:
            pass
