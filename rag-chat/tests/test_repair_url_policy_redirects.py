"""PinnedTransport pins the socket address without rewriting the caller's request URL,
so redirects, same-origin checks and response.url keep the hostname. No network."""

import unittest
from unittest.mock import patch

import httpx

from app.repair import url_policy


# pack-agent-composer-c43c7f824d
class PinnedTransportRedirectTests(unittest.IsolatedAsyncioTestCase):
    async def test_relative_redirect_stays_on_the_name_and_is_pinned_again(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            if request.url.path == "/old":
                return httpx.Response(302, headers={"Location": "/new"})
            return httpx.Response(200, json={})

        async def resolve(host, port):
            return ["192.0.2.10"]

        with patch.object(url_policy, "_resolve", resolve):
            client = url_policy.audit_client(
                transport=url_policy.PinnedTransport(httpx.MockTransport(handler)), follow_redirects=True,
            )
            async with client:
                r = await client.get("https://audit.example:8765/old", headers={"Authorization": "Bearer t"})

        observed = {
            "response_url": str(r.url),
            "request_url": str(r.request.url),
            "hops": [(q.url.host, q.url.path, q.headers.get("host"), q.extensions.get("sni_hostname"),
                      q.headers.get("authorization")) for q in seen],
        }
        self.assertEqual(observed, {
            "response_url": "https://audit.example:8765/new",
            "request_url": "https://audit.example:8765/new",
            "hops": [
                ("192.0.2.10", "/old", "audit.example:8765", "audit.example", "Bearer t"),
                ("192.0.2.10", "/new", "audit.example:8765", "audit.example", "Bearer t"),
            ],
        })


if __name__ == "__main__":
    unittest.main()
