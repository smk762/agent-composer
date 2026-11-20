"""SSRF guard fixture: outbound clients the rule must flag (tests/test_ssrf_guard.py). Never imported."""

import httpx
from httpx import AsyncClient


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
