"""Info-leak guard fixture: responses the rule must flag (scripts/tests/test_info_leak_guard.py). Never imported."""

import io
import sys
import traceback

import httpx
from fastapi import HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse, Response


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


def convert(raw: str) -> int:
    try:
        return int(raw)
    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc()) from None  # traceback


def debug_dump(op):
    try:
        return op()
    except Exception:
        buf = io.StringIO()
        traceback.print_exc(file=buf)
        return PlainTextResponse(buf.getvalue(), status_code=500)  # traceback, through print_exc's buffer


def bad_input(raw: str):
    try:
        return int(raw)
    except ValueError:
        trace = "".join(traceback.format_exception(*sys.exc_info()))
        raise HTTPException(400, detail=trace)  # traceback, even in a 4xx


def no_name(op):
    try:
        return op()
    except Exception:
        return JSONResponse({"error": str(sys.exc_info()[1])}, status_code=500)  # exception text, no `as`


def unpacked(op):
    try:
        return op()
    except Exception:
        _, err, _ = sys.exc_info()
        return Response(content=f"failed: {err}", status_code=503)  # exception text in a plain Response
