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

    async def test_status_callback_payload_and_token(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json={"status": "resolved"})

        with _patch_http(handler):
            err = await report_pack_status(PACK_ID, BASE, "resolved", note="done", api_token="tok")
        self.assertEqual(err, "")
        req = seen[0]
        self.assertEqual(req.url.path, f"/audit/packs/{PACK_ID}/status")
        self.assertEqual(json.loads(req.content), {"status": "resolved", "note": "done"})
        self.assertEqual(req.headers["x-api-token"], "tok")


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

        from app.db import get_db
        from app.models.schemas import AuthContext

        self.router_mod = repair_router
        app = FastAPI()
        app.include_router(repair_router.router)

        async def no_db():
            yield None

        async def fake_user(request, db):
            return AuthContext(user_id="test", via="dev")

        app.dependency_overrides[get_db] = no_db
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
            patch.object(repair_router, "require_user", fake_user),
            patch.object(repair_router, "AUDIT_API_ALLOWED_HOSTS", "audit.test:8765"),
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

    def test_applied_auto_fix_reports_resolved(self) -> None:
        posts: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                posts.append(request)
                return httpx.Response(200, json={})
            return _ok_handler(request)

        async def fake_generate(prompt: str, model: str, system: str) -> str:
            return "```diff\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n```"

        async def fake_path(repo: str):
            return "/tmp/repo"

        with patch.object(multi_model, "_generate", fake_generate), \
                patch.object(self.router_mod, "_resolve_repo_path", fake_path), \
                patch.object(self.router_mod, "apply_inplace", lambda path, patch_text: None):
            _, events = self._post(
                {"pack_id": PACK_ID, "diff": "--- a\n+++ b\n", "apply": True, "mode": "auto_fix"},
                handler=handler,
            )
        self.assertEqual(events[-1][0], "complete")
        self.assertTrue(events[-1][1]["applied"])
        self.assertEqual(len(posts), 1)
        self.assertEqual(posts[0].url.path, f"/audit/packs/{PACK_ID}/status")
        self.assertEqual(json.loads(posts[0].content)["status"], "resolved")

    def test_rejected_patch_is_never_applied_or_resolved(self) -> None:
        """Iteration 1 proposes a patch the audit rejects; iteration 2 proposes nothing."""
        posts: list[httpx.Request] = []
        applied: list[str] = []
        replies = iter(["```diff\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n```", ""])

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                posts.append(request)
                return httpx.Response(200, json={})
            return _ok_handler(request)

        async def fake_generate(prompt: str, model: str, system: str) -> str:
            return next(replies, "")

        async def rejected(patch_text: str, repo: str, url: str):
            return False, ["audit: new HIGH finding"]

        async def fake_path(repo: str):
            return "/tmp/repo"

        with patch.object(multi_model, "_generate", fake_generate), \
                patch.object(multi_model, "_validate_via_api", rejected), \
                patch.object(self.router_mod, "_resolve_repo_path", fake_path), \
                patch.object(self.router_mod, "apply_inplace", lambda path, p: applied.append(p)):
            _, events = self._post(
                {"pack_id": PACK_ID, "diff": "--- a\n+++ b\n", "apply": True, "mode": "auto_fix",
                 "max_iterations": 2, "audit_api_url": "http://audit.test:8765"},
                handler=handler,
            )
        done = events[-1][1]
        self.assertEqual(events[-1][0], "complete")
        self.assertEqual((done["applied"], done["patch_validated"], done["iterations"]), (False, False, 2))
        self.assertIn("No patch passed validation", done["error"])
        self.assertEqual(applied, [])
        self.assertFalse(any(p.url.path.endswith("/status") for p in posts))

    def test_job_survives_client_disconnect(self) -> None:
        import asyncio

        from app.repair.models import RepairRequest as Req

        async def scenario():
            resp = await self.router_mod.run_repair(
                Req(repo="argus-cortex", diff="--- a\n+++ b\n", validate_tests=False), None, None)
            it = resp.body_iterator
            first = await it.__anext__()
            job_id = json.loads(first.split("data:", 1)[1])["job_id"]
            await it.aclose()  # the client goes away after the first event
            await asyncio.gather(*list(self.router_mod._tasks))
            return self.router_mod._get_job(job_id)

        job = asyncio.run(scenario())
        self.assertEqual(job["status"], "completed")

    def test_full_store_evicts_finished_jobs_never_running_ones(self) -> None:
        from app.repair.models import RepairResult

        mod = self.router_mod
        with patch.object(mod, "_jobs", {}), patch.object(mod, "_MAX_JOBS", 2):
            def job(i):
                return RepairResult(job_id=f"j{i}", repo="r", status="pending", mode="review", created_at="t")
            self.assertTrue(mod._store_job(job(1)))
            self.assertTrue(mod._store_job(job(2)))
            self.assertFalse(mod._store_job(job(3)))  # both still running
            mod._update_job("j2", status="completed")
            self.assertTrue(mod._store_job(job(3)))
            self.assertEqual(sorted(mod._jobs), ["j1", "j3"])

    def test_review_mode_does_not_report_status(self) -> None:
        posts: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                posts.append(request)
            return _ok_handler(request)

        self._post({"pack_id": PACK_ID, "diff": "--- a\n+++ b\n"}, handler=handler)
        self.assertEqual(posts, [])

    def test_audit_urls_outside_the_allowlist_are_refused(self) -> None:
        for field in ("audit_api_url", "audit_api_base"):
            for url in ("http://evil.test:8765", "http://audit.test:6379", "http://169.254.169.254/", "file:///etc/passwd"):
                r = self.client.post("/api/repair/run", json={"repo": "r", "diff": "x", field: url})
                self.assertEqual(r.status_code, 422, (field, url, r.text))
                self.assertIn(field, r.text)
        self.assertEqual(self.prompts, [])  # refused before any work

    def test_requires_something_to_repair(self) -> None:
        r = self.client.post("/api/repair/run", json={"repo": "argus-cortex"})
        self.assertEqual(r.status_code, 422)
        r = self.client.post("/api/repair/run", json={"diff": "x"})
        self.assertEqual(r.status_code, 422)


if __name__ == "__main__":
    unittest.main()


class RepairAuthGateTest(unittest.TestCase):
    """apply / git_ops need an authenticated caller; read-only runs stay open."""

    def setUp(self) -> None:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from app import auth
        from app.db import get_db
        from app.routers import repair as repair_router

        async def no_db():
            yield None

        async def fake_generate(prompt: str, model: str, system: str) -> str:
            return ""

        app = FastAPI()
        app.include_router(repair_router.router)
        app.dependency_overrides[get_db] = no_db
        self.client = TestClient(app)
        for p in (
            patch.object(auth, "DEV_AUTH_BYPASS", False),
            patch.object(multi_model, "_generate", fake_generate),
            patch.object(repair_router, "AUDIT_API_URL", ""),
            patch.object(repair_router, "QDRANT_URL", ""),
        ):
            p.start()
            self.addCleanup(p.stop)

    def _run(self, **extra):
        body = {"repo": "argus-cortex", "diff": "--- a\n+++ b\n", "validate_tests": False, **extra}
        return self.client.post("/api/repair/run", json=body)

    def test_apply_requires_auth(self) -> None:
        self.assertEqual(self._run(apply=True, mode="auto_fix").status_code, 401)

    def test_git_ops_requires_auth(self) -> None:
        self.assertEqual(self._run(git_ops={"enabled": True}).status_code, 401)

    def test_access_identity_allows_apply(self) -> None:
        r = self.client.post(
            "/api/repair/run",
            json={"repo": "argus-cortex", "diff": "--- a\n+++ b\n", "validate_tests": False, "apply": True},
            headers={"Cf-Access-Authenticated-User-Email": "ops@example.com"},
        )
        self.assertEqual(r.status_code, 200)

    def test_read_only_runs_stay_open(self) -> None:
        self.assertEqual(self._run(mode="review").status_code, 200)
        self.assertEqual(self._run(git_ops={"enabled": False}).status_code, 200)


class AuditUrlPolicyTests(unittest.TestCase):
    """Issue #28: client-supplied audit URLs must be configured or allow-listed origins."""

    def check(self, url, allowlist="", configured=("http://host.docker.internal:8765",)):
        from app.repair.url_policy import check_audit_url
        return check_audit_url(url, "audit_api_url", allowlist=allowlist, configured=configured)

    def test_configured_origin_only_by_default(self) -> None:
        self.assertIsNone(self.check("http://host.docker.internal:8765/"))
        self.assertIsNone(self.check("HTTP://Host.Docker.Internal:8765"))
        self.assertIn("not allowed", self.check("http://host.docker.internal:6379"))  # same host, other service
        self.assertIn("not allowed", self.check("http://192.168.1.50:8765"))

    def test_allowlist_entries(self) -> None:
        self.assertIsNone(self.check("http://audit.lan:8765", allowlist=" audit.lan:8765 , other "))
        self.assertIsNotNone(self.check("http://audit.lan:9000", allowlist="audit.lan:8765"))
        self.assertIsNone(self.check("https://other/x", allowlist="audit.lan:8765,other"))  # bare host: any port
        self.assertIsNone(self.check("http://[::1]:8765", allowlist="[::1]:8765"))
        self.assertIsNone(self.check("http://anything:1", allowlist="*"))
        self.assertIsNotNone(self.check("http://audit.lan:8765", allowlist="http://audit.lan/path"))  # bad entry ignored

    def test_never_link_local_or_non_http(self) -> None:
        for url in ("http://169.254.169.254/latest", "http://[fe80::1]:80", "http://[::ffff:169.254.169.254]/"):
            self.assertIn("link-local", self.check(url, allowlist="*"))
        for url in ("file:///etc/passwd", "ftp://audit:21", "http://", "not a url", "http://h:99999"):
            self.assertIn("http(s)", self.check(url, allowlist="*"))
