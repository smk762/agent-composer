import asyncio
import os
import signal
import time
from typing import Dict, List, Optional

import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response


LISTEN_HOST = os.getenv("INFINITY_MANAGER_HOST", "0.0.0.0")
LISTEN_PORT = int(os.getenv("INFINITY_MANAGER_PORT", "7997"))
CHILD_HOST = os.getenv("INFINITY_CHILD_HOST", "127.0.0.1")
CHILD_PORT = int(os.getenv("INFINITY_CHILD_PORT", "7998"))
IDLE_TIMEOUT_S = float(os.getenv("INFINITY_IDLE_TIMEOUT", "900"))
STARTUP_TIMEOUT_S = float(os.getenv("INFINITY_STARTUP_TIMEOUT", "180"))
REQUEST_TIMEOUT_S = float(os.getenv("INFINITY_REQUEST_TIMEOUT", "300"))

MODEL_IDS = [
    item.strip()
    for item in os.getenv("INFINITY_MODEL_IDS", "nomic-ai/CodeRankEmbed,BAAI/bge-reranker-v2-m3").split(",")
    if item.strip()
]
ENGINE = os.getenv("INFINITY_ENGINE", "torch")
DEVICE = os.getenv("INFINITY_DEVICE", "cuda")
EXTRA_ARGS = [item for item in os.getenv("INFINITY_EXTRA_ARGS", "").split(" ") if item]
CHILD_CMD = os.getenv("INFINITY_CHILD_CMD", "").strip()

UPSTREAM = f"http://{CHILD_HOST}:{CHILD_PORT}"
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


app = FastAPI(title="Infinity Idle Manager")

_lock = asyncio.Lock()
_process: Optional[asyncio.subprocess.Process] = None
_last_used = 0.0
_active_requests = 0


def _base_command() -> List[str]:
    if CHILD_CMD:
        return CHILD_CMD.split()

    command = ["infinity_emb", "v2"]
    for model_id in MODEL_IDS:
        command.extend(["--model-id", model_id])
    command.extend(["--port", str(CHILD_PORT), "--engine", ENGINE, "--device", DEVICE])
    command.extend(EXTRA_ARGS)
    return command


async def _is_ready() -> bool:
    try:
        timeout = aiohttp.ClientTimeout(total=2)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f"{UPSTREAM}/health") as response:
                return response.status == 200
    except Exception:
        return False


async def _start_locked() -> None:
    global _process
    if _process and _process.returncode is None:
        return

    command = _base_command()
    _process = await asyncio.create_subprocess_exec(
        *command,
        start_new_session=True,
    )

    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    while time.monotonic() < deadline:
        if _process.returncode is not None:
            raise RuntimeError(f"Infinity exited during startup with code {_process.returncode}")
        if await _is_ready():
            return
        await asyncio.sleep(1)

    await _stop_locked()
    raise RuntimeError("Infinity did not become healthy before startup timeout")


async def _ensure_started() -> None:
    global _last_used
    async with _lock:
        await _start_locked()
        _last_used = time.time()


async def _stop_locked() -> bool:
    global _process
    if not _process or _process.returncode is not None:
        _process = None
        return False

    pid = _process.pid
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        _process = None
        return False

    try:
        await asyncio.wait_for(_process.wait(), timeout=20)
    except asyncio.TimeoutError:
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await _process.wait()

    _process = None
    return True


async def _idle_reaper() -> None:
    while True:
        await asyncio.sleep(5)
        if IDLE_TIMEOUT_S < 0:
            continue
        async with _lock:
            if not _process or _process.returncode is not None or _active_requests:
                continue
            if time.time() - _last_used >= IDLE_TIMEOUT_S:
                await _stop_locked()


@app.on_event("startup")
async def startup() -> None:
    if os.getenv("INFINITY_PRELOAD", "1") == "1":
        await _ensure_started()
    asyncio.create_task(_idle_reaper())


@app.on_event("shutdown")
async def shutdown() -> None:
    async with _lock:
        await _stop_locked()


@app.get("/health")
async def health() -> Dict[str, float | str | bool | None]:
    running = bool(_process and _process.returncode is None)
    return {
        "status": "ok",
        "state": "running" if running else "evicted",
        "managed": True,
        "idle_timeout": IDLE_TIMEOUT_S,
        "last_used": _last_used or None,
    }


@app.get("/admin/status")
async def admin_status() -> Dict[str, object]:
    running = bool(_process and _process.returncode is None)
    return {
        "state": "running" if running else "evicted",
        "managed": True,
        "pid": _process.pid if running and _process else None,
        "models": MODEL_IDS,
        "idle_timeout": IDLE_TIMEOUT_S,
        "last_used": _last_used or None,
        "upstream": UPSTREAM,
    }


@app.post("/admin/wake")
async def admin_wake() -> Dict[str, object]:
    await _ensure_started()
    return await admin_status()


@app.post("/admin/evict")
async def admin_evict() -> Dict[str, object]:
    async with _lock:
        stopped = await _stop_locked()
    status = await admin_status()
    status["evicted"] = stopped
    return status


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def proxy(path: str, request: Request) -> Response:
    global _active_requests, _last_used
    if path.startswith("admin/"):
        raise HTTPException(404, detail="Not Found")

    await _ensure_started()
    _active_requests += 1
    try:
        body = await request.body()
        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in HOP_BY_HOP_HEADERS and key.lower() != "host"
        }
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.request(
                request.method,
                f"{UPSTREAM}/{path}",
                params=request.query_params,
                data=body,
                headers=headers,
            ) as upstream_response:
                content = await upstream_response.read()
                response_headers = {
                    key: value
                    for key, value in upstream_response.headers.items()
                    if key.lower() not in HOP_BY_HOP_HEADERS
                }
                content_type = upstream_response.headers.get("content-type")
                status_code = upstream_response.status
        return Response(
            content=content,
            status_code=status_code,
            headers=response_headers,
            media_type=content_type,
        )
    finally:
        _last_used = time.time()
        _active_requests -= 1


if __name__ == "__main__":
    uvicorn.run(app, host=LISTEN_HOST, port=LISTEN_PORT)
