"""The client-facing error for a failed upstream call (ai-code-auditor pack-class-info-leak).

An upstream service's error body can name its hosts, paths, versions and internals, so a
5xx keeps its status but gets a fixed detail, and the body goes to the log. A 4xx is the
service's own message about this request (an unknown voice, a bad label set), written for
the client, so it is relayed, capped. ``scripts/tests/test_info_leak_guard.py`` fails CI on
an upstream ``.text`` in any other client error, which steers new code here.
"""
from __future__ import annotations

import logging

import httpx
from fastapi import HTTPException

log = logging.getLogger("rag-chat.upstream")

_DETAIL_CAP = 500


def upstream_error(response: httpx.Response, service: str) -> HTTPException:
    """``HTTPException`` with ``response``'s status: its body for a 4xx, a fixed message otherwise."""
    status = response.status_code
    if 400 <= status < 500:
        # info-leak: a 4xx body is the service's own message about this request, for the client
        return HTTPException(status, detail=response.text[:_DETAIL_CAP])
    log.warning("%s answered %s: %s", service, status, response.text[:_DETAIL_CAP])
    return HTTPException(status, detail=f"{service} failed ({status})")
