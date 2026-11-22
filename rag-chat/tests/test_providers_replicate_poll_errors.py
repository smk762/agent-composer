"""Replicate: any poll failure cancels the prediction; a prediction that ended on its own is not cancelled."""

import asyncio
import unittest
from unittest import mock

import httpx

from app.providers import replicate

CANCEL = f"{replicate._API_BASE}/predictions/pred-1/cancel"


def _handler(poll_response):
    calls = []

    def handle(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, str(request.url)))
        if request.method == "POST" and request.url.path.endswith("/cancel"):
            return httpx.Response(200, json={})
        if request.method == "POST":
            return httpx.Response(201, json={"id": "pred-1"})
        return poll_response(request)

    return handle, calls


# pack-agent-composer-64b9b8f72d
class ReplicatePollErrorTests(unittest.TestCase):
    def _run(self, poll_response):
        handle, calls = _handler(poll_response)
        real = httpx.AsyncClient

        def factory(*a, **kw):
            kw["transport"] = httpx.MockTransport(handle)
            return real(*a, **kw)

        provider = replicate.ReplicateProvider(api_key="k")
        with mock.patch.object(replicate.httpx, "AsyncClient", factory), \
                mock.patch.object(replicate, "_POLL_INTERVAL", 0):
            try:
                asyncio.run(provider.generate_video("a cat"))
            except Exception as exc:  # noqa: BLE001 - the error type is asserted by the caller
                return type(exc), calls
        return None, calls

    def test_poll_failures_cancel_the_prediction(self) -> None:
        def raise_transport(request):
            raise httpx.ConnectError("blip", request=request)

        outcomes = [
            self._run(lambda r: httpx.Response(503, json={})),
            self._run(lambda r: httpx.Response(200, content=b"not json")),
            self._run(raise_transport),
        ]
        self.assertTrue(all(err is not None for err, _ in outcomes))
        self.assertEqual([("POST", CANCEL) in calls for _, calls in outcomes], [True, True, True])

    def test_failed_prediction_is_not_cancelled(self) -> None:
        err, calls = self._run(lambda r: httpx.Response(200, json={"status": "failed", "error": "x"}))
        self.assertTrue(issubclass(err, RuntimeError))
        self.assertNotIn(("POST", CANCEL), calls)


if __name__ == "__main__":
    unittest.main()
