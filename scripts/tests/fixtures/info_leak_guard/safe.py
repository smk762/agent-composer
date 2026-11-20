"""Info-leak guard fixture: responses the rule must accept (scripts/tests/test_info_leak_guard.py). Never imported."""

import logging

import httpx
from fastapi import HTTPException
from fastapi.responses import JSONResponse

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


def parse(payload):
    return payload
