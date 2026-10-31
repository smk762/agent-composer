from __future__ import annotations

import re
import time

from fastapi import APIRouter, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware

metrics_router = APIRouter()

_HTTP_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30)
_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.I,
)
_INT_SEGMENT_RE = re.compile(r"(?<=/)\d+(?=/|$)")
_SKIP_PATHS = {"/metrics", "/health", "/ready"}

REQ_COUNT = Counter(
    "http_requests_total",
    "HTTP requests by method, normalized path, and status code",
    ["method", "path", "status_code"],
)

REQ_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["method", "path"],
    buckets=_HTTP_BUCKETS,
)

GPU_OOM_COUNT = Counter(
    "gpu_oom_events_total",
    "GPU out-of-memory events during inference",
    ["operation"],
)


def _normalize_path(path: str) -> str:
    path = _UUID_RE.sub("{id}", path)
    path = _INT_SEGMENT_RE.sub("{id}", path)
    return path


def observe_request(*, method: str, path: str, status_code: int, duration_s: float) -> None:
    norm_path = _normalize_path(path)
    REQ_COUNT.labels(method=method, path=norm_path, status_code=str(status_code)).inc()
    REQ_LATENCY.labels(method=method, path=norm_path).observe(duration_s)


def observe_gpu_oom(*, operation: str) -> None:
    GPU_OOM_COUNT.labels(operation=operation).inc()


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path in _SKIP_PATHS:
            return await call_next(request)

        start = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            observe_request(
                method=request.method,
                path=path,
                status_code=status_code,
                duration_s=time.perf_counter() - start,
            )


@metrics_router.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
