"""Repair pipeline hardening: ref names, fail-closed validation, loop bookkeeping,
patch extraction, in-place apply and the audit URL policy. No network, no model."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from pydantic import ValidationError

from app.repair import multi_model, url_policy
from app.repair.models import RepairRequest
from app.repair.patch_applier import apply_inplace, extract_patch

real_async_client = httpx.AsyncClient
DIFF = "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b"


def _patch_http(handler):
    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_async_client(*args, **kwargs)

    return patch("httpx.AsyncClient", factory)


# ── compare_branch (pack-agent-composer-5171e7789d) ─────────────────────────────


class CompareBranchTests(unittest.TestCase):
    def test_option_like_compare_branch_is_refused(self) -> None:
        for value in ("--output=/tmp/pwned", "-p", "--no-index"):
            with self.assertRaises(ValidationError, msg=value):
                RepairRequest(repo="r", compare_branch=value)

    def test_range_and_whitespace_compare_branch_is_refused(self) -> None:
        for value in ("main..dev", "dev main", "dev\n--output=x", "HEAD@{1}", "a:b", "x~1"):
            with self.assertRaises(ValidationError, msg=value):
                RepairRequest(repo="r", compare_branch=value)

    def test_plain_refs_are_accepted(self) -> None:
        for value in ("", "dev", "origin/dev", "feature/x-y_z", "v1.2.3", "0123abcd"):
            self.assertEqual(RepairRequest(repo="r", compare_branch=value).compare_branch, value)


# ── fail-closed validation (b0814d8fb0, ae95e194f4) ──────────────────────────────


class DiffAuditFailsClosedTests(unittest.IsolatedAsyncioTestCase):
    async def _check(self, handler):
        with _patch_http(handler):
            return await multi_model._validate_via_api(DIFF, "repo", "http://audit")

    async def test_server_error_is_not_a_pass(self) -> None:
        passed, errors = await self._check(lambda req: httpx.Response(500, text="boom"))
        self.assertFalse(passed)
        self.assertIn("audit unavailable", errors[0])

    async def test_connection_error_is_not_a_pass(self) -> None:
        def handler(req):
            raise httpx.ConnectError("refused", request=req)

        passed, errors = await self._check(handler)
        self.assertFalse(passed)
        self.assertIn("ConnectError", errors[0])

    async def test_malformed_payload_is_not_a_pass(self) -> None:
        for body in ([], {"findings": "none"}, {"no": "findings"}, {"findings": ["x"]}):
            passed, _ = await self._check(lambda req, body=body: httpx.Response(200, json=body))
            self.assertFalse(passed, body)

    async def test_clean_audit_still_passes(self) -> None:
        passed, errors = await self._check(lambda req: httpx.Response(200, json={"findings": [{"severity": "LOW"}]}))
        self.assertTrue(passed)
        self.assertEqual(errors, [])


class TestValidationFailsClosedTests(unittest.IsolatedAsyncioTestCase):
    async def test_client_timeout_is_not_a_pass(self) -> None:
        def handler(req):
            raise httpx.ReadTimeout("slow", request=req)

        with _patch_http(handler):
            passed, errors = await multi_model._validate_via_tests(DIFF, "repo", "http://audit")
        self.assertFalse(passed)
        self.assertIn("timed out", errors[0])

    async def test_client_waits_longer_than_the_server_budget(self) -> None:
        seen = {}

        def handler(req):
            import json
            seen["server_budget"] = json.loads(req.content)["timeout_s"]
            seen["client_read"] = req.extensions["timeout"]["read"]
            return httpx.Response(200, json={"passed": True})

        with _patch_http(handler):
            await multi_model._validate_via_tests(DIFF, "repo", "http://audit", timeout=60)
        self.assertEqual(seen["server_budget"], 60)
        self.assertGreaterEqual(seen["client_read"], seen["server_budget"] + 30)

    async def test_unreachable_or_erroring_endpoint_is_not_a_pass(self) -> None:
        def refused(req):
            raise httpx.ConnectError("refused", request=req)

        for handler in (refused, lambda req: httpx.Response(404), lambda req: httpx.Response(502)):
            with _patch_http(handler):
                passed, _ = await multi_model._validate_via_tests(DIFF, "repo", "http://audit")
            self.assertFalse(passed)

    async def test_passing_and_skipped_runs_still_pass(self) -> None:
        for body in ({"passed": True}, {"skipped_reason": "no test runner"}):
            with _patch_http(lambda req, body=body: httpx.Response(200, json=body)):
                passed, _ = await multi_model._validate_via_tests(DIFF, "repo", "http://audit")
            self.assertTrue(passed, body)


# ── repair_loop bookkeeping (e426de652e, 7e21ad516e) ────────────────────────────


class RepairLoopTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, responses, request, api_result=(True, []), test_result=(True, [])):
        prompts: list[str] = []
        replies = iter(responses)

        async def fake_generate(prompt, model, system):
            prompts.append(prompt)
            return next(replies, "")

        async def fake_api(patch_, repo, url):
            return api_result

        async def fake_tests(patch_, repo, url, timeout=180):
            return test_result

        with patch.object(multi_model, "_generate", fake_generate), \
                patch.object(multi_model, "_validate_via_api", fake_api), \
                patch.object(multi_model, "_validate_via_tests", fake_tests):
            iterations = [it async for it in multi_model.repair_loop(request)]
        return iterations, prompts

    async def test_without_validation_the_loop_runs_max_iterations(self) -> None:
        fenced = f"```diff\n{DIFF}\n```"
        request = RepairRequest(repo="r", diff=DIFF, mode="auto_fix", max_iterations=3, audit_api_url="")
        iterations, _ = await self._run([fenced] * 3, request)
        self.assertEqual(len(iterations), 3)
        self.assertTrue(all(it.validation_passed is None for it in iterations))

    async def test_validated_patch_still_stops_the_loop(self) -> None:
        fenced = f"```diff\n{DIFF}\n```"
        request = RepairRequest(repo="r", diff=DIFF, mode="auto_fix", max_iterations=3, audit_api_url="http://audit")
        iterations, _ = await self._run([fenced] * 3, request)
        self.assertEqual(len(iterations), 1)
        self.assertTrue(iterations[0].validation_passed)

    async def test_an_iteration_without_a_patch_keeps_the_errors(self) -> None:
        fenced = f"```diff\n{DIFF}\n```"
        request = RepairRequest(repo="r", diff=DIFF, mode="auto_fix", max_iterations=3, audit_api_url="http://audit")
        iterations, prompts = await self._run(
            [fenced, "no idea", "still nothing"], request,
            api_result=(False, ["HIGH x.py:1 — sql injection in query()"]),
        )
        self.assertEqual(len(iterations), 3)
        self.assertEqual(iterations[1].patch, "")
        self.assertIn("sql injection in query()", prompts[2])  # iteration 3 still sees the error


    async def test_skipped_critic_does_not_carry_old_notes(self) -> None:
        fenced = f"```diff\n{DIFF}\n```"
        request = RepairRequest(
            repo="r", diff=DIFF, mode="auto_fix", max_iterations=3, audit_api_url="",
            critique_model="critic",
        )
        # iteration 1: fixer patch + critic note; iteration 2: no patch (critic skipped).
        iterations, prompts = await self._run([fenced, "STALE CRITIC NOTE", "no idea", "nothing"], request)
        self.assertEqual(iterations[1].patch, "")
        self.assertIn("STALE CRITIC NOTE", prompts[2])      # iteration 2 sees iteration 1's notes
        self.assertNotIn("STALE CRITIC NOTE", prompts[3])   # iteration 3 does not


class GenerateFailureTests(unittest.IsolatedAsyncioTestCase):
    """A failed model call raises; it is never the empty string a model can legitimately return."""

    async def _generate_with(self, handler):
        with _patch_http(handler):
            return await multi_model._generate("p", "m", "s")

    async def test_failures_raise_and_empty_answers_do_not(self) -> None:
        def down(request):
            raise httpx.ConnectError("refused", request=request)

        cases = {
            "unreachable": down,
            "answered 500": lambda r: httpx.Response(500, text="boom"),
            "JSONDecodeError": lambda r: httpx.Response(200, text="not json"),
        }
        for why, handler in cases.items():
            with self.assertRaisesRegex(multi_model.GenerationFailed, why):
                await self._generate_with(handler)
        self.assertEqual(await self._generate_with(lambda r: httpx.Response(200, json={"response": ""})), "")

    async def test_a_failed_call_ends_the_loop_instead_of_iterating_on_nothing(self) -> None:
        calls = []

        async def failing(prompt, model, system):
            calls.append(prompt)
            raise multi_model.GenerationFailed("model m: Ollama unreachable")

        request = RepairRequest(repo="r", diff=DIFF, mode="auto_fix", max_iterations=3, audit_api_url="")
        with patch.object(multi_model, "_generate", failing):
            with self.assertRaises(multi_model.GenerationFailed):
                [it async for it in multi_model.repair_loop(request)]
        self.assertEqual(len(calls), 1)


# ── extract_patch (8e2687b000, a03eeabf52) ──────────────────────────────────────


class ExtractPatchTests(unittest.TestCase):
    def test_diff_block_after_a_non_diff_block(self) -> None:
        response = f"Before:\n```python\nprint(1)\n```\nThen apply this:\n```diff\n{DIFF}\n```\nThat's it."
        self.assertEqual(extract_patch(response), DIFF + "\n")

    def test_inline_diff_stops_at_trailing_fence_and_prose(self) -> None:
        response = f"Fix:\n{DIFF}\n```\n\nThis renames a to b."
        self.assertEqual(extract_patch(response), DIFF + "\n")

    def test_inline_diff_stops_at_prose(self) -> None:
        response = f"{DIFF}\nThis renames a to b."
        self.assertEqual(extract_patch(response), DIFF + "\n")

    def test_plain_fenced_diff(self) -> None:
        self.assertEqual(extract_patch(f"```diff\n{DIFF}\n```"), DIFF + "\n")
        self.assertEqual(extract_patch("nothing here"), "")

    def test_crlf_fenced_diff(self) -> None:
        response = f"```diff\n{DIFF}\n```\nThis renames a to b.".replace("\n", "\r\n")
        self.assertEqual(extract_patch(response), DIFF + "\n")

    def test_trailing_blank_context_line_is_kept(self) -> None:
        diff = "--- a/x.py\n+++ b/x.py\n@@ -1,2 +1,2 @@\n-a\n+b\n \n"
        self.assertEqual(extract_patch(f"```diff\n{diff}```"), diff)
        self.assertEqual(extract_patch(f"{diff}\n"), diff)


# ── apply_inplace (4234183cf8) ──────────────────────────────────────────────────


@unittest.skipUnless(shutil.which("git"), "git not installed")
class ApplyInplaceTests(unittest.TestCase):
    def test_a_planted_symlink_does_not_redirect_the_patch(self) -> None:
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        repo, victim = root / "repo", root / "victim.txt"
        repo.mkdir()
        victim.write_text("precious\n")
        (repo / "x.py").write_text("a\n")
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        os.symlink(victim, repo / ".ai_repair_patch.tmp")
        os.symlink(victim, repo / ".ai_repair_write_probe")

        apply_inplace(repo, DIFF + "\n")

        self.assertEqual(victim.read_text(), "precious\n")
        self.assertEqual((repo / "x.py").read_text(), "b\n")


# ── audit URL policy (5ef6904310, 169d3e6421) ───────────────────────────────────


class NonCanonicalLiteralTests(unittest.TestCase):
    def check(self, url):
        return url_policy.check_audit_url(url, "audit_api_url", allowlist="192.168.1.198:8765,[::1]:8765,audit.lan:8765", configured=())

    def test_decimal_and_hex_ipv4_are_link_local(self) -> None:
        for url in ("http://2852039166:8765/", "http://0xA9FEA9FE/", "http://0251.0376.0251.0376/"):
            self.assertIn("link-local", self.check(url) or "", url)

    def test_short_and_trailing_dot_ipv4_are_link_local(self) -> None:
        for url in ("http://169.254.43518/", "http://169.254.169.254./"):
            self.assertIn("link-local", self.check(url) or "", url)

    def test_ipv4_inside_ipv6_is_link_local(self) -> None:
        for url in ("http://[::a9fe:a9fe]/", "http://[64:ff9b::a9fe:a9fe]/", "http://[2002:a9fe:a9fe::]/"):
            self.assertIn("link-local", self.check(url) or "", url)

    def test_lan_and_loopback_stay_allowed_by_the_allowlist(self) -> None:
        for url in ("http://192.168.1.198:8765", "http://[::1]:8765", "http://audit.lan:8765"):
            self.assertIsNone(self.check(url), url)


class ResolvedHostTests(unittest.TestCase):
    """An allow-listed name is resolved; one pointing at link-local is a 422."""

    def setUp(self) -> None:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from app.db import get_db
        from app.routers import repair as repair_router

        async def no_db():
            yield None

        async def fake_generate(prompt, model, system):
            return ""

        async def resolve(host, port):
            return {"metadata.evil.test": ["10.0.0.5", "169.254.169.254"],
                    "audit.lan": ["192.168.1.198"]}.get(host) or []

        app = FastAPI()
        app.include_router(repair_router.router)
        app.dependency_overrides[get_db] = no_db
        self.client = TestClient(app)
        for p in (
            patch.object(multi_model, "_generate", fake_generate),
            patch.object(repair_router, "AUDIT_API_URL", ""),
            patch.object(repair_router, "QDRANT_URL", ""),
            patch.object(repair_router, "AUDIT_API_ALLOWED_HOSTS",
                         "metadata.evil.test:8765,nowhere.evil.test:8765,audit.lan:8765"),
            patch.object(url_policy, "_resolve", resolve, create=True),
        ):
            p.start()
            self.addCleanup(p.stop)

    def _post(self, url):
        return self.client.post("/api/repair/run", json={"repo": "r", "diff": "x", "validate_tests": False,
                                                          "audit_api_url": url})

    def test_name_resolving_to_metadata_is_refused(self) -> None:
        r = self._post("http://metadata.evil.test:8765")
        self.assertEqual(r.status_code, 422, r.text)
        self.assertIn("link-local", r.text)

    def test_name_that_does_not_resolve_is_refused(self) -> None:
        self.assertEqual(self._post("http://nowhere.evil.test:8765").status_code, 422)

    def test_lan_name_is_accepted(self) -> None:
        self.assertEqual(self._post("http://audit.lan:8765").status_code, 200)


if __name__ == "__main__":
    unittest.main()


class PinnedTransportTests(unittest.IsolatedAsyncioTestCase):
    """The fetch connects to the address it vetted, so a name re-pointed after
    ``check_resolved`` (DNS rebinding) cannot reach link-local (d881588efb)."""

    async def _get(self, url, answers):
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json={})

        async def resolve(host, port):
            return answers

        with patch.object(url_policy, "_resolve", resolve):
            client = url_policy.audit_client(transport=url_policy.PinnedTransport(httpx.MockTransport(handler)))
            async with client:
                await client.get(url)
        return seen

    async def test_connects_to_the_vetted_address_keeping_host_and_sni(self) -> None:
        seen = await self._get("https://audit.example:8765/audit/packs/x", ["192.0.2.10"])
        self.assertEqual(seen[0].url.host, "192.0.2.10")
        self.assertEqual(seen[0].headers["host"], "audit.example:8765")
        self.assertEqual(seen[0].extensions["sni_hostname"], "audit.example")

    async def test_name_rebound_to_metadata_is_refused_at_connect_time(self) -> None:
        for answers in (["169.254.169.254"], ["192.0.2.10", "fe80::1"], []):
            with self.assertRaises(httpx.ConnectError, msg=answers):
                await self._get("http://rebind.example:8765/", answers)

    async def test_link_local_literal_is_refused(self) -> None:
        with self.assertRaises(httpx.ConnectError):
            await self._get("http://169.254.169.254/latest/meta-data", [])

    async def test_pack_fetch_through_a_rebound_name_fails_soft_without_reaching_it(self) -> None:
        from app.repair import pack_client

        async def resolve(host, port):
            return ["169.254.169.254"]

        with patch.object(url_policy, "_resolve", resolve):
            ctx = await pack_client.fetch_pack_context("pack-x", "http://rebind.example:8765", timeout_s=2)
        self.assertFalse(ctx.ok)
        self.assertIn("link-local", ctx.warning)
