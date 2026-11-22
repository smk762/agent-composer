"""SSRF guard fixture: outbound clients the rule must accept (tests/test_ssrf_guard.py). Never imported."""

import urllib.parse
import urllib.request

import httpx
import requests

from app.repair.url_policy import audit_client

OLLAMA_URL = "http://ollama:11434"


async def fetch_pack(audit_api_base: str, pack_id: str):
    async with audit_client(timeout=10) as client:  # the pinned helper
        return await client.get(f"{audit_api_base}/audit/packs/{pack_id}")


async def generate(payload: dict):
    # outbound: OLLAMA_URL is operator config, never request data
    async with httpx.AsyncClient(timeout=60) as client:
        return await client.post(f"{OLLAMA_URL}/api/generate", json=payload)


def classify(exc: Exception) -> bool:
    return isinstance(exc, (httpx.HTTPError, httpx.TimeoutException))  # types, not calls


def ollama_tags() -> bytes:
    # outbound: OLLAMA_URL is operator config, never request data
    with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as resp:
        return resp.read()


def same_line() -> int:
    return requests.get(f"{OLLAMA_URL}/api/version", timeout=5).status_code  # outbound: operator-configured OLLAMA_URL


def host_of(url: str) -> str:
    return urllib.parse.urlsplit(url).hostname or ""  # parsing a URL opens nothing


def is_net_error(exc: Exception) -> bool:
    return isinstance(exc, requests.RequestException)  # a type, not a call
