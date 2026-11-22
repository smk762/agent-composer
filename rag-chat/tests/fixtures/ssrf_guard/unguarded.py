"""SSRF guard fixture: outbound clients the rule must flag (tests/test_ssrf_guard.py). Never imported."""

import http.client
import urllib.request
from urllib import request as urllib_request

import aiohttp
import httpx
import requests
from httpx import AsyncClient
from requests import Session


async def fetch_pack(audit_api_base: str, pack_id: str):
    async with httpx.AsyncClient(timeout=10) as client:  # client-supplied base, no pinning
        return await client.get(f"{audit_api_base}/audit/packs/{pack_id}")


async def callback(url: str):
    async with AsyncClient() as client:  # imported by name
        return await client.post(url)


def probe(url: str):
    return httpx.get(url)  # module-level helper


async def unexplained(url: str):
    # outbound:
    async with httpx.AsyncClient() as client:  # a marker without a reason doesn't count
        return await client.get(url)


def fetch_url(payload: dict) -> bytes:
    with urllib.request.urlopen(payload["url"], timeout=10) as resp:  # stdlib, request-supplied URL
        return resp.read()


def build(url: str):
    return urllib_request.Request(url, method="POST")  # `from urllib import request`


def sync_get(url: str):
    return requests.get(url, timeout=10)  # requests


def session_get(url: str):
    with Session() as s:  # requests.Session imported by name
        return s.get(url)


def raw(host: str):
    return http.client.HTTPSConnection(host, timeout=10)  # http.client


async def aio(url: str):
    async with aiohttp.ClientSession() as session:  # aiohttp
        return await session.get(url)
