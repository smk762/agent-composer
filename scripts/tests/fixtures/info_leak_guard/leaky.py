"""Info-leak guard fixture: responses the rule must flag (scripts/tests/test_info_leak_guard.py). Never imported."""

import httpx
from fastapi import HTTPException
from fastapi.responses import JSONResponse


async def search(q):
    try:
        return await q.run()
    except Exception as e:
        raise HTTPException(502, f"Search failed: {e}")  # exception text in a 5xx


async def generate(model):
    try:
        return await model.run()
    except RuntimeError as exc:
        return JSONResponse(status_code=503, content={"error": "GPU OOM", "detail": str(exc)})  # exception text


async def relay(client, url):
    try:
        r = await client.get(url)
        r.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, detail=exc.response.text)  # upstream body


async def forward(client, url):
    response = await client.get(url)
    if response.status_code >= 400:
        raise HTTPException(response.status_code, detail=response.text)  # upstream body


async def dynamic(op, status):
    try:
        return await op()
    except Exception as err:
        # info-leak:
        raise HTTPException(status, detail=repr(err))  # status unknown, and a marker with no reason
