"""Git ops stop before commit/push when the branch endpoint names no branch. No network, no model."""

import unittest
from unittest.mock import patch

import httpx

from app.repair.models import GitOpsConfig

real_async_client = httpx.AsyncClient
DIFF = "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b"


def _patch_http(handler):
    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_async_client(*args, **kwargs)

    return patch("httpx.AsyncClient", factory)


# pack-agent-composer-b239656881
class GitOpsEmptyBranchTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_branch_stops_before_commit_and_push(self) -> None:
        from app.routers import repair as repair_router

        seen = []

        def handler(req):
            seen.append(req.url.path)
            if req.url.path.endswith("/audit/git/branch"):
                return httpx.Response(200, json={"branch": ""})
            return httpx.Response(200, json={"commit_sha": "abc", "push_url": "u"})

        cfg = GitOpsConfig(enabled=True, commit=True, push=True)
        with _patch_http(handler):
            branch, sha, push_url, err = await repair_router._run_git_ops(
                job_id="repair-abc", final_patch="patch", git_ops=cfg, repo="repo",
                audit_api_url="http://audit",
            )
        self.assertEqual(seen, ["/audit/git/branch"])
        self.assertEqual((branch, sha, push_url), ("", "", ""))
        self.assertIn("no branch", err)


if __name__ == "__main__":
    unittest.main()
