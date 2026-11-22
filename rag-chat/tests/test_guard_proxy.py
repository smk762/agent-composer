"""/api/guard/check proxies to argus-retina's SingGuard: field mapping, defaults, error mapping.

No network: httpx.AsyncClient is swapped for one over an httpx.MockTransport.
"""

import json
import unittest
from unittest.mock import patch

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers import guard

UPSTREAM = "http://retina-singguard:8080"
MODEL = "inclusionAI/SingGuard-4b"
_RealAsyncClient = httpx.AsyncClient

UNSAFE = {
    "safe": False,
    "verdict": "unsafe",
    "rule": "B. Real-World Crimes & Public Safety",
    "explanation": "[Step 1] The user asks how to build a weapon.",
    "categories": [
        {
            "code": "B. Real-World Crimes & Public Safety",
            "label": "B. Real-World Crimes & Public Safety",
            "why": "[Step 1] The user asks how to build a weapon.",
        }
    ],
    "fast_verdict": "unsafe",
    "model": MODEL,
    "thinking": "fast-slow",
    "elapsed_ms": 812,
}
SAFE = {
    "safe": True,
    "verdict": "safe",
    "rule": "Safe",
    "explanation": "",
    "categories": [],
    "fast_verdict": "safe",
    "model": MODEL,
    "thinking": "fast",
    "elapsed_ms": 90,
}


class GuardProxyTests(unittest.TestCase):
    def setUp(self) -> None:
        app = FastAPI()
        app.include_router(guard.router)
        self.client = TestClient(app)
        self.sent: list[httpx.Request] = []
        for name, value in {"GUARD_URL": UPSTREAM, "GUARD_THINKING": "fast-slow", "_upstream_model": None}.items():
            p = patch.object(guard, name, value)
            p.start()
            self.addCleanup(p.stop)

    def _upstream(self, handler) -> None:
        sent = self.sent

        def record(request: httpx.Request) -> httpx.Response:
            sent.append(request)
            return handler(request)

        class _Client(_RealAsyncClient):
            def __init__(self, *a, **kw):
                kw["transport"] = httpx.MockTransport(record)
                super().__init__(*a, **kw)

        p = patch.object(guard.httpx, "AsyncClient", _Client)
        p.start()
        self.addCleanup(p.stop)

    def _answer(self, status: int, body) -> None:
        self._upstream(lambda request: httpx.Response(status, json=body))

    def _check(self, **body):
        return self.client.post("/api/guard/check", json={"input": "hello", **body})

    def _forwarded(self) -> dict:
        checks = [r for r in self.sent if r.url.path == "/api/guard/check"]
        self.assertEqual(len(checks), 1)
        return json.loads(checks[0].content)

    # --- mapping -----------------------------------------------------------------------------

    def test_an_unsafe_answer_keeps_rag_chats_shape(self) -> None:
        self._answer(200, UNSAFE)
        r = self._check(thinking="fast-slow")
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertFalse(data["safe"])
        self.assertEqual(data["verdict"], "unsafe")
        self.assertEqual(data["rule"], "B. Real-World Crimes & Public Safety")
        # The rule's text from the default policy, not the explanation again.
        self.assertIn("violent crime, weapons", data["rule_text"])
        self.assertEqual(data["explanation"], UNSAFE["explanation"])
        self.assertEqual(data["categories"], UNSAFE["categories"])
        self.assertEqual(data["raw"], "")
        self.assertEqual(data["model"], MODEL)
        self.assertEqual(data["thinking"], "fast-slow")
        self.assertEqual(data["fast_verdict"], "unsafe")
        self.assertNotIn("fast_rule", data)
        self.assertIsInstance(data["elapsed_ms"], int)

    def test_rule_text_falls_back_to_the_category_why(self) -> None:
        body = dict(UNSAFE, rule="X. Not in the policy")
        body["categories"] = [{"code": "X. Not in the policy", "label": "X", "why": "because"}]
        self._answer(200, body)
        self.assertEqual(self._check().json()["rule_text"], "because")

    def test_rule_text_comes_from_the_requests_policy(self) -> None:
        policy = "### B. Weapons\n  - guns and knives\n\n### Safe\n  - anything else"
        body = dict(UNSAFE, rule="B. Weapons")
        body["categories"] = [{"code": "B. Weapons", "label": "B. Weapons", "why": "it asks for a gun"}]
        self._answer(200, body)
        r = self._check(policy=policy)
        self.assertEqual(r.json()["rule_text"], "- guns and knives")
        self.assertEqual(self._forwarded()["policy"], policy)

    def test_a_safe_answer(self) -> None:
        self._answer(200, SAFE)
        data = self._check(thinking="fast").json()
        self.assertTrue(data["safe"])
        self.assertEqual(data["rule"], "Safe")
        self.assertEqual(data["rule_text"], "- Content that does not match any risk category.")
        self.assertEqual(data["categories"], [])

    def test_a_cascade_passes_fast_rule_through(self) -> None:
        self._answer(200, dict(UNSAFE, thinking="slow", fast_rule="B. Real-World Crimes & Public Safety"))
        data = self._check(thinking="cascade").json()
        self.assertEqual(data["thinking"], "slow")
        self.assertEqual(data["fast_rule"], "B. Real-World Crimes & Public Safety")
        self.assertEqual(self._forwarded()["thinking"], "cascade")

    # --- the forwarded request ---------------------------------------------------------------

    def test_blank_thinking_is_rag_chats_default(self) -> None:
        self._answer(200, UNSAFE)
        with patch.object(guard, "GUARD_THINKING", "slow"):
            self._check()
        self.assertEqual(self._forwarded()["thinking"], "slow")

    def test_the_request_is_forwarded(self) -> None:
        self._answer(200, UNSAFE)
        self._check(
            input="a reply",
            input_role="assistant",
            image_base64="data:image/png;base64,aGk=",
            conversation=[{"role": "user", "content": "hi"}],
            thinking="SLOW",
        )
        self.assertEqual(self.sent[0].url, httpx.URL(f"{UPSTREAM}/api/guard/check"))
        self.assertEqual(
            self._forwarded(),
            {
                "input": "a reply",
                "image_base64": "data:image/png;base64,aGk=",
                "input_role": "assistant",
                "conversation": [{"role": "user", "content": "hi"}],
                "policy": None,  # blank: retina's default rules
                "thinking": "slow",
            },
        )

    def test_bad_thinking_is_a_400_without_an_upstream_call(self) -> None:
        self._answer(200, UNSAFE)
        r = self._check(thinking="bogus")
        self.assertEqual(r.status_code, 400)
        self.assertEqual(self.sent, [])

    # --- model override ----------------------------------------------------------------------

    def _with_health(self, check_status: int = 200, check_body=None) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/health":
                return httpx.Response(200, json={"model_state": "gpu", "busy": False, "model": MODEL})
            return httpx.Response(check_status, json=check_body or UNSAFE)

        self._upstream(handler)

    def test_another_model_is_refused(self) -> None:
        self._with_health()
        r = self._check(model="hf.co/inclusionAI/SingGuard-8b-GGUF:Q4_K_M")
        self.assertEqual(r.status_code, 400)
        self.assertIn(MODEL, r.json()["detail"])
        self.assertEqual([req.url.path for req in self.sent], ["/health"])

    def test_the_upstreams_own_model_is_accepted(self) -> None:
        self._with_health()
        r = self._check(model=MODEL)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertNotIn("model", self._forwarded())

    # --- errors ------------------------------------------------------------------------------

    def test_unset_url_is_a_503(self) -> None:
        self._answer(200, UNSAFE)
        with patch.object(guard, "GUARD_URL", ""):
            r = self._check()
        self.assertEqual(r.status_code, 503)
        self.assertIn("not configured", r.json()["detail"])
        self.assertEqual(self.sent, [])

    def test_upstream_400_is_relayed(self) -> None:
        self._answer(400, {"detail": "image_base64 is not an image"})
        r = self._check()
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.json()["detail"], "image_base64 is not an image")

    def test_upstream_422_is_a_400(self) -> None:
        self._answer(422, {"detail": [{"loc": ["body", "input"], "msg": "String should have at most 32768 characters"}]})
        r = self._check()
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.json()["detail"], "input: String should have at most 32768 characters")

    def test_upstream_503_is_a_503(self) -> None:
        self._answer(503, {"detail": "SingGuard is loading"})
        r = self._check()
        self.assertEqual(r.status_code, 503)
        self.assertIn("SingGuard is loading", r.json()["detail"])

    def test_upstream_500_is_a_502(self) -> None:
        self._answer(500, {"detail": "SingGuard returned no verdict"})
        r = self._check()
        self.assertEqual(r.status_code, 502)
        self.assertEqual(r.json()["detail"], "guard returned no verdict")

    def test_another_upstream_status_is_a_502_not_relayed(self) -> None:
        self._answer(404, {"detail": "Not Found"})
        r = self._check()
        self.assertEqual(r.status_code, 502)
        self.assertNotEqual(r.status_code, 404)

    def test_an_unusable_body_is_a_502(self) -> None:
        self._upstream(lambda request: httpx.Response(200, content=b"<html>proxy</html>"))
        r = self._check()
        self.assertEqual(r.status_code, 502)
        self.assertNotIn("html", r.json()["detail"])

    def _raising(self, exc: Exception):
        def handler(request: httpx.Request) -> httpx.Response:
            raise exc

        self._upstream(handler)
        return self._check()

    def test_timeout_is_a_504(self) -> None:
        r = self._raising(httpx.ReadTimeout("slow"))
        self.assertEqual(r.status_code, 504)
        self.assertIn("timed out", r.json()["detail"])

    def test_unreachable_is_a_503(self) -> None:
        self.assertEqual(self._raising(httpx.ConnectError("down")).status_code, 503)
        self.assertEqual(self._raising(httpx.RemoteProtocolError("restarted")).status_code, 503)


class GuardPolicyRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        app = FastAPI()
        app.include_router(guard.router)
        self.client = TestClient(app)

    def test_policy_and_categories_need_no_upstream(self) -> None:
        with patch.object(guard, "GUARD_URL", ""):
            policy = self.client.get("/api/guard/policy").json()
            categories = self.client.get("/api/guard/categories").json()
        self.assertIn("### G. Animal Abuse", policy["policy"])
        self.assertEqual(policy["thinking_modes"], ["fast", "fast-slow", "slow", "cascade"])
        self.assertEqual(categories[0]["title"], "A. Sexual Content Risk")


if __name__ == "__main__":
    unittest.main()
