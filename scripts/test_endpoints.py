#!/usr/bin/env python3
"""
Minimal endpoint smoke test for rag-chat and rag-ingest.
Dependency-free (stdlib only).
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Tuple


CHAT_BASE = os.environ.get("CHAT_URL", "http://127.0.0.1:9150")
INGEST_BASE = os.environ.get("INGEST_URL", "http://127.0.0.1:9050")


def _request(
    url: str, method: str = "GET", body: bytes | None = None, headers: dict[str, str] | None = None
) -> Tuple[int, Any]:
    req = urllib.request.Request(url, data=body, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read()
            ctype = resp.headers.get("Content-Type", "")
            if "application/json" in ctype:
                return resp.status, json.loads(raw.decode("utf-8") or "{}")
            return resp.status, raw.decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", errors="replace")
        return e.code, msg
    except Exception as e:
        return 0, str(e)


def get_json(url: str) -> Tuple[int, Any]:
    return _request(url, "GET", None, {"Accept": "application/json"})


def post_json(url: str, payload: dict[str, Any]) -> Tuple[int, Any]:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return _request(
        url,
        "POST",
        body,
        {"Accept": "application/json", "Content-Type": "application/json"},
    )


def check_health(name: str, url: str) -> bool:
    status, data = get_json(url)
    ok = status == 200
    print(f"{name}: {'OK' if ok else 'FAIL'} (status {status}) -> {data}")
    return ok


def check_chat() -> bool:
    payload = {"messages": [{"role": "user", "content": "Hello, test message."}]}
    status, data = post_json(f"{CHAT_BASE}/chat", payload)
    ok = status == 200
    snippet = data if isinstance(data, str) else data.get("content")
    print(f"rag-chat /chat: {'OK' if ok else 'FAIL'} (status {status}) -> {snippet}")
    return ok


def main() -> None:
    ok = True
    ok &= check_health("rag-chat /health", f"{CHAT_BASE}/health")
    ok &= check_chat()
    ok &= check_health("rag-ingest /health", f"{INGEST_BASE}/health")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
