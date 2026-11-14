"""Lazy ModernBERT model lifecycle (unloaded -> cpu -> gpu) with keep-alive timers
(MODERNBERT_KEEP_ALIVE_GPU / _CPU) that demote to CPU and then unload after idle.
"""

import asyncio
import gc
import logging
import os
import time
from contextlib import asynccontextmanager
from enum import Enum
from typing import AsyncIterator, Optional

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

    Inference must hold a lease (``async with manager.use(): ...``) for as long as it touches
    ``model``/``tokenizer``. Demotion and unload wait until no lease is held, and the idle timer
    only starts when the last lease is released.
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
        self._in_use: int = 0
        self._idle: asyncio.Event = asyncio.Event()
        self._idle.set()

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

    async def _make_ready(self, num_labels: int) -> None:
        # Caller holds self._lock.
        if self._state == ModelState.GPU:
            pass
        elif self._state == ModelState.CPU:
            await self._promote()
        elif self._state in (ModelState.UNLOADED, ModelState.ERROR):
            await self._do_load(num_labels)
            await self._promote()
        # LOADING is serialised by the lock — can't reach here from outside

    async def ensure_ready(self, num_labels: int = 2) -> None:
        """Guarantee model is on GPU (or CPU if MODERNBERT_DEVICE=cpu) and reset keep-alive.

        Does not protect a following inference; use ``use()`` for that.
        """
        async with self._lock:
            await self._make_ready(num_labels)
            self._last_used = time.time()
            if self._in_use == 0:
                self._arm_evict()

    @asynccontextmanager
    async def use(self, num_labels: int = 2) -> AsyncIterator["ModelManager"]:
        """Make the model ready and hold it for the whole block.

        While any lease is held the model is neither demoted nor unloaded. Don't call
        ``ensure_ready``/``evict``/``unload`` inside the block: they take the lock that a
        waiting demote/unload holds, which would deadlock.
        """
        async with self._lock:
            await self._make_ready(num_labels)
            self._in_use += 1
            self._idle.clear()
            if self._evict_task and not self._evict_task.done():
                self._evict_task.cancel()
        try:
            yield self
        finally:
            self._in_use -= 1
            self._last_used = time.time()
            if self._in_use == 0:
                self._idle.set()
                self._arm_evict()

    async def evict(self) -> None:
        """Move GPU → CPU without full unload (waits for in-flight inference)."""
        async with self._lock:
            await self._idle.wait()
            if self._state == ModelState.GPU:
                await self._demote()

    def _unload_locked(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            gc.collect()
            torch.cuda.empty_cache()
        self._state = ModelState.UNLOADED
        logger.info("Model unloaded")

    async def unload(self) -> None:
        """Full unload; frees both VRAM and RAM (waits for in-flight inference)."""
        async with self._lock:
            await self._idle.wait()
            if (self._evict_task and not self._evict_task.done()
                    and self._evict_task is not asyncio.current_task()):
                self._evict_task.cancel()
            self._unload_locked()

    # ------------------------------------------------------------------ timer

    def _arm_evict(self) -> None:
        if self._evict_task and not self._evict_task.done():
            self._evict_task.cancel()
        self._evict_task = asyncio.create_task(self._eviction_loop())

    async def _eviction_loop(self) -> None:
        # Armed when the model goes idle (last lease released, or ensure_ready with no lease);
        # any new lease cancels it. A stage that finds a lease held just stops: its release re-arms.
        try:
            # --- GPU keep-alive ---
            if self.keep_alive_gpu < 0:
                return  # stay resident forever
            # On a CPU-only device the model lives in CPU state, which is its resident
            # state: it gets the same keep-alive before the CPU stage below.
            resident_cpu = self._state == ModelState.CPU and self.effective_device() == "cpu"
            if self._state == ModelState.GPU or resident_cpu:
                if self.keep_alive_gpu > 0:
                    await asyncio.sleep(self.keep_alive_gpu)
                async with self._lock:
                    if self._in_use:
                        return
                    if self._state == ModelState.GPU:
                        await self._demote()
                    elif not (resident_cpu and self._state == ModelState.CPU):
                        return
            elif self._state != ModelState.CPU:
                return  # unloaded / error: nothing to evict

            # --- CPU keep-alive --- (after a GPU→CPU demotion, or CPU-resident to begin with)
            if self.keep_alive_cpu < 0:
                return  # stay on CPU forever
            if self.keep_alive_cpu > 0:
                await asyncio.sleep(self.keep_alive_cpu)
            async with self._lock:
                if self._state != ModelState.CPU or self._in_use:
                    return
                self._unload_locked()
        except asyncio.CancelledError:
            pass
