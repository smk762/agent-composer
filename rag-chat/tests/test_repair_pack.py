"""Pack-driven repair: pack fetch, context injection, repo resolution. No network, no model."""

import json
import unittest
from unittest.mock import patch

import httpx

from app.repair import multi_model
from app.repair.context_builder import build_repair_context
from app.repair.models import RepairRequest
from app.repair.pack_client import (
    build_pack_block,
    fetch_pack_context,
    report_pack_status,
    resolve_repo,
    validate_pack_id,
)

PACK_ID = "ext-retry-core-001"
BASE = "http://audit.test:8765"

PACK = {
    "work_item_id": PACK_ID,
    "kind": "extraction",
    "priority": {"severity": "HIGH", "score": 0.9},
    "repo": {"name": "argus-cortex"},
    "location": {"file_path": "cortex/retry.py", "line": 42, "symbol": "with_retry"},
    "narrative": {
        "title": "Extract shared retry helper",
        "recommendation": "Move retry logic into a shared module.",
        "acceptance_criteria": ["Retry helper lives in one module", "No behaviour change"],
    },
    "verification": {"suggested_test_commands": ["pytest -q tests/test_retry.py"]},
    "governance": {"architecture_rules_hit": [{"rule_name": "no-cross-repo-import", "evidence": "x imports y"}]},
}
BRIEF = "# Agent Work Pack: Extract shared retry helper\n\nDo not auto-merge."

real_async_client = httpx.AsyncClient


def _patch_http(handler):
    """Route every httpx.AsyncClient through a MockTransport."""
    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_async_client(*args, **kwargs)

    return patch("httpx.AsyncClient", factory)


def _ok_handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == f"/audit/packs/{PACK_ID}":
        return httpx.Response(200, json=PACK)
    if path == f"/audit/packs/{PACK_ID}/brief":
        return httpx.Response(200, text=BRIEF)
    return httpx.Response(404, json={"detail": "nope"})


class PackBlockTests(unittest.TestCase):
    def test_block_has_key_fields(self) -> None:
        ctx = build_pack_block(PACK, BRIEF)
        self.assertTrue(ctx.ok)
        self.assertEqual(ctx.repo, "argus-cortex")
        for needle in (
            PACK_ID,
            "Extract shared retry helper",
            "Retry helper lives in one module",
            "pytest -q tests/test_retry.py",
            "no-cross-repo-import",
            "cortex/retry.py:42",
            "Do not auto-merge.",
        ):
            self.assertIn(needle, ctx.block)

    def test_pack_id_validation(self) -> None:
        self.assertTrue(validate_pack_id("pack-world-ecosystem"))
        for bad in ("", "../etc/passwd", "a/b", "a b", "x" * 300):
            self.assertFalse(validate_pack_id(bad))

    def test_resolve_repo(self) -> None:
        self.assertEqual(resolve_repo("", "argus-cortex", write_ops=True), ("argus-cortex", "", ""))
        self.assertEqual(resolve_repo("foo", "", write_ops=True), ("foo", "", ""))
        self.assertEqual(resolve_repo("Argus-Cortex", "argus-cortex", write_ops=True)[2], "")
        repo, warn, err = resolve_repo("foo", "argus-cortex", write_ops=False)
        self.assertEqual((repo, bool(warn), err), ("argus-cortex", True, ""))
        repo, warn, err = resolve_repo("foo", "argus-cortex", write_ops=True)
        self.assertEqual((repo, warn), ("foo", ""))
        self.assertIn("refusing", err)


class PackFetchTests(unittest.IsolatedAsyncioTestCase):
    async def test_fetch_success(self) -> None:
        with _patch_http(_ok_handler):
            ctx = await fetch_pack_context(PACK_ID, BASE)
        self.assertTrue(ctx.ok)
        self.assertEqual(ctx.title, "Extract shared retry helper")
        self.assertEqual(ctx.acceptance_criteria[0], "Retry helper lives in one module")
        self.assertIn("### Pack brief", ctx.block)

    async def test_fetch_not_found_fails_soft(self) -> None:
        with _patch_http(lambda r: httpx.Response(404, json={})):
            ctx = await fetch_pack_context(PACK_ID, BASE)
        self.assertFalse(ctx.ok)
        self.assertEqual(ctx.block, "")
        self.assertIn("not found", ctx.warning)

    async def test_fetch_timeout_fails_soft(self) -> None:
        def boom(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("slow", request=request)

        with _patch_http(boom):
            ctx = await fetch_pack_context(PACK_ID, BASE, timeout_s=0.1)
        self.assertFalse(ctx.ok)
        self.assertIn("failed", ctx.warning)

    async def test_json_down_brief_up_still_usable(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/brief"):
                return httpx.Response(200, text=BRIEF)
            return httpx.Response(500)

        with _patch_http(handler):
            ctx = await fetch_pack_context(PACK_ID, BASE)
        self.assertTrue(ctx.ok)
        self.assertIn("Do not auto-merge.", ctx.block)
        self.assertTrue(ctx.warning)

    async def test_bad_inputs_make_no_request(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("no request expected")

        with _patch_http(handler):
            self.assertFalse((await fetch_pack_context("../x", BASE)).ok)
            self.assertFalse((await fetch_pack_context(PACK_ID, "file:///etc")).ok)

    async def test_status_callback_soft_when_endpoint_missing(self) -> None:
        with _patch_http(lambda r: httpx.Response(404)):
            err = await report_pack_status(PACK_ID, BASE, "resolved")
        self.assertIn("unavailable", err)


class RepairContextTests(unittest.IsolatedAsyncioTestCase):
    def test_context_without_pack_unchanged(self) -> None:
        ctx = build_repair_context(diff="--- a\n+++ b\n")
        self.assertTrue(ctx.startswith("## Diff under review"))
        self.assertNotIn("Agent work pack", ctx)

    def test_context_contains_pack_before_diff(self) -> None:
        block = build_pack_block(PACK, BRIEF).block
        ctx = build_repair_context(diff="--- a\n+++ b\n", pack_context=block)
        self.assertIn(PACK_ID, ctx)
        self.assertIn("Extract shared retry helper", ctx)
        self.assertIn("No behaviour change", ctx)
        self.assertLess(ctx.index("Agent work pack"), ctx.index("Diff under review"))

    async def test_repair_loop_prompt_includes_pack(self) -> None:
        prompts: list[str] = []

        async def fake_generate(prompt: str, model: str, system: str) -> str:
            prompts.append(prompt)
            return ""

        req = RepairRequest(
            repo="argus-cortex",
            diff="--- a\n+++ b\n",
            pack_context=build_pack_block(PACK, BRIEF).block,
            audit_api_url="",
        )
        with patch.object(multi_model, "_generate", fake_generate):
            async for _ in multi_model.repair_loop(req):
                pass
        self.assertEqual(len(prompts), 1)
        self.assertIn(PACK_ID, prompts[0])
        self.assertIn("Extract shared retry helper", prompts[0])
        self.assertIn("Retry helper lives in one module", prompts[0])


class RepairEndpointTests(unittest.TestCase):
    """POST /api/repair/run with pack_id, HTTP to the audit API mocked."""

    def setUp(self) -> None:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from app.routers import repair as repair_router

        self.router_mod = repair_router
        app = FastAPI()
        app.include_router(repair_router.router)
        self.client = TestClient(app)
        self.prompts: list[tuple[str, str]] = []  # (repo, prompt)
        self._cur_repo = ""

        async def fake_generate(prompt: str, model: str, system: str) -> str:
            self.prompts.append((self._cur_repo, prompt))
            return ""

        self._patches = [
            patch.object(multi_model, "_generate", fake_generate),
            patch.object(repair_router, "AUDIT_API_URL", ""),
            patch.object(repair_router, "QDRANT_URL", ""),
        ]
        for p in self._patches:
            p.start()
            self.addCleanup(p.stop)

    def _post(self, body: dict, handler=_ok_handler):
        with _patch_http(handler):
            r = self.client.post("/api/repair/run", json={"validate_tests": False, "audit_api_base": BASE, **body})
        events: list[tuple[str, dict]] = []
        event = ""
        for line in r.text.splitlines():
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                events.append((event, json.loads(line[5:].strip())))
        return r, events

    def test_pack_id_injects_context_and_adopts_repo(self) -> None:
        r, events = self._post({"pack_id": PACK_ID, "diff": "--- a\n+++ b\n"})
        self.assertEqual(r.status_code, 200)
        names = [e for e, _ in events]
        self.assertIn("pack_loaded", names)
        self.assertEqual(dict(events)["pack_loaded"]["repo"], "argus-cortex")
        self.assertEqual(names[-1], "complete")
        prompt = self.prompts[0][1]
        self.assertIn(PACK_ID, prompt)
        self.assertIn("Extract shared retry helper", prompt)
        self.assertIn("Retry helper lives in one module", prompt)

    def test_pack_only_request_is_accepted(self) -> None:
        r, events = self._post({"pack_id": PACK_ID})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(events[-1][0], "complete")

    def test_pack_fetch_failure_continues_with_warning(self) -> None:
        r, events = self._post(
            {"pack_id": PACK_ID, "repo": "argus-cortex", "diff": "--- a\n+++ b\n"},
            handler=lambda req: httpx.Response(503),
        )
        names = [e for e, _ in events]
        self.assertIn("warning", names)
        self.assertNotIn("pack_loaded", names)
        self.assertEqual(names[-1], "complete")
        self.assertNotIn("Agent work pack", self.prompts[0][1])

    def test_pack_failure_without_repo_errors(self) -> None:
        _, events = self._post({"pack_id": PACK_ID}, handler=lambda req: httpx.Response(404))
        self.assertEqual(events[-1][0], "error")

    def test_repo_mismatch_readonly_prefers_pack_repo(self) -> None:
        _, events = self._post({"pack_id": PACK_ID, "repo": "other", "diff": "--- a\n+++ b\n"})
        self.assertEqual(dict(events)["pack_loaded"]["repo"], "argus-cortex")
        self.assertTrue(any("differs" in d.get("message", "") for e, d in events if e == "warning"))

    def test_repo_mismatch_with_apply_is_rejected(self) -> None:
        _, events = self._post(
            {"pack_id": PACK_ID, "repo": "other", "diff": "--- a\n+++ b\n", "apply": True, "mode": "auto_fix"}
        )
        self.assertEqual(events[-1][0], "error")
        self.assertEqual(self.prompts, [])

    def test_without_pack_id_unchanged(self) -> None:
        r, events = self._post({"repo": "argus-cortex", "diff": "--- a\n+++ b\n"})
        names = [e for e, _ in events]
        self.assertNotIn("pack_loaded", names)
        self.assertNotIn("warning", names)
        self.assertNotIn("Agent work pack", self.prompts[0][1])

    def test_requires_something_to_repair(self) -> None:
        r = self.client.post("/api/repair/run", json={"repo": "argus-cortex"})
        self.assertEqual(r.status_code, 422)
        r = self.client.post("/api/repair/run", json={"diff": "x"})
        self.assertEqual(r.status_code, 422)


if __name__ == "__main__":
    unittest.main()
