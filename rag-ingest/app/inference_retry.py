from __future__ import annotations

import asyncio
import gc
import time
from contextlib import asynccontextmanager
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")
_OOM_MESSAGE = "CUDA out of memory after waiting for capacity"

try:
    import torch
except Exception:  # pragma: no cover - torch is optional
    torch = None  # type: ignore[assignment]


class RetryableOomError(RuntimeError):
    """Raised when OOM retry budget is exhausted."""


class InferenceWorkers:
    def __init__(self, max_workers: int) -> None:
        self._semaphore = asyncio.Semaphore(max(1, int(max_workers)))

    @asynccontextmanager
    async def acquire(self):
        async with self._semaphore:
            yield

    async def run(self, fn: Callable[[], Awaitable[T]]) -> T:
        async with self._semaphore:
            return await fn()


def _is_oom_error(exc: Exception) -> bool:
    if torch is not None:
        out_of_memory_error = getattr(torch, "OutOfMemoryError", None)
        if out_of_memory_error and isinstance(exc, out_of_memory_error):
            return True

    if not isinstance(exc, RuntimeError):
        return False

    msg = str(exc).lower()
    return (
        "out of memory" in msg
        or "cuda oom" in msg
        or "cuda error: out of memory" in msg
        or "cublas_status_alloc_failed" in msg
        or "cuda error: memory allocation" in msg
    )


def _clear_cuda_memory() -> None:
    gc.collect()
    if torch is None:
        return
    cuda = getattr(torch, "cuda", None)
    if cuda is not None and cuda.is_available():
        cuda.empty_cache()


async def run_with_oom_wait(
    fn: Callable[[], Awaitable[T]],
    *,
    max_wait_s: int,
    interval_s: int,
    on_oom: Callable[[], None],
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    clear_memory: Callable[[], None] = _clear_cuda_memory,
) -> T:
    deadline = monotonic() + max(0, max_wait_s)

    while True:
        try:
            return await fn()
        except Exception as exc:
            if not _is_oom_error(exc):
                raise

            on_oom()
            clear_memory()

            if monotonic() >= deadline:
                raise RetryableOomError(_OOM_MESSAGE) from exc

            wait_s = min(max(0, interval_s), max(0.0, deadline - monotonic()))
            if wait_s > 0:
                await sleep(wait_s)
