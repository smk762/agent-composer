"""Info-leak guard fixture: responses the rule must accept (scripts/tests/test_info_leak_guard.py). Never imported."""

import logging
import sys
import traceback

import httpx
from fastapi import HTTPException
from fastapi.responses import JSONResponse, Response

log = logging.getLogger(__name__)


async def search(q):
    try:
        return await q.run()
    except Exception:
        log.exception("search failed")
        raise HTTPException(502, "Search failed (details in the log)")  # fixed message


async def validate(payload):
    try:
        return parse(payload)
    except ValueError as exc:
        raise HTTPException(422, f"Invalid payload: {exc}")  # a 4xx: our own validation message


async def busy(op):
    try:
        return await op()
    except HTTPException as exc:
        raise HTTPException(exc.status_code, detail=exc.detail)  # our own error's client-facing fields


async def oom(model):
    try:
        return await model.run()
    except MemoryError as exc:
        log.error("oom: %s", exc)
        return JSONResponse(status_code=503, content={"error": "GPU OOM"})


async def relay(client, url):
    try:
        r = await client.get(url)
        r.raise_for_status()
    except httpx.HTTPStatusError as exc:
        # info-leak: a 4xx body is the service's own message about this request, for the client
        raise HTTPException(exc.response.status_code, detail=exc.response.text)


def convert(raw: str) -> int:
    try:
        return int(raw)
    except Exception:
        log.error("conversion failed:\n%s", traceback.format_exc())  # the trace goes to the log
        raise HTTPException(status_code=500, detail="conversion failed (details in the log)") from None


def bad_input(raw: str):
    try:
        return int(raw)
    except ValueError:
        # a 4xx: our own validation message, reached through sys.exc_info()
        return JSONResponse({"error": f"not a number: {sys.exc_info()[1]}"}, status_code=422)


def ok() -> Response:
    return Response(content=b"ok", media_type="text/plain")


def parse(payload):
    return payload
