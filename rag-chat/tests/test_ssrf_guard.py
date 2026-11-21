"""SSRF class guard (ai-code-auditor pack-class-ssrf).

The repair loop is where request data names a server-side fetch target (``audit_api_url`` /
``audit_api_base``), and the GPU broker names the node every Ollama call of a lease goes to.
This file is the guard for that class:

* **Rule.** In ``app/repair/`` and ``app/routers/repair.py``, an outbound httpx client is
  ``url_policy.audit_client`` (pinned, link-local refused at connect time). A plain
  ``httpx.AsyncClient`` / ``httpx.Client`` / ``httpx.get``… is allowed only with a
  ``# outbound: <reason>`` comment on the call's line or the line above, saying why its URL
  can't come from a request. The rule is checked on positive and negative fixtures, then on
  the real tree, so a new unpinned client there fails CI.
* **Helpers.** ``AUDIT_API_ALLOWED_HOSTS`` has no wildcard (``*`` is ignored), and a broker's
  node URL must be a bare http(s) origin whose host isn't link-local
  (``gpu_lease.vet_node_url``).
"""

import ast
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # rag-chat/
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "ssrf_guard"
SCOPE = [ROOT / "app" / "repair", ROOT / "app" / "routers" / "repair.py"]
HELPER = ROOT / "app" / "repair" / "url_policy.py"  # audit_client itself builds the one plain client

_CLIENT_ATTRS = {"AsyncClient", "Client", "get", "post", "put", "patch", "delete", "head", "options",
                 "request", "stream"}
_MARKER = re.compile(r"#\s*outbound:\s*(.*)$")
MIN_REASON = 10


def _marked(lines: list[str], lineno: int) -> bool:
    for i in (lineno - 1, lineno - 2):  # the call's line, then the line above
        if 0 <= i < len(lines):
            m = _MARKER.search(lines[i])
            if m and len(m.group(1).strip()) >= MIN_REASON:
                return True
    return False


def unguarded_clients(path: Path) -> list[str]:
    """``file:line`` of every unmarked outbound httpx call in *path*."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    lines = source.splitlines()
    by_name = {alias.asname or alias.name
               for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module == "httpx"
               for alias in node.names if alias.name in _CLIENT_ATTRS}
    modules = {alias.asname or alias.name  # `import httpx as hx` names the module hx
               for node in ast.walk(tree) if isinstance(node, ast.Import)
               for alias in node.names if alias.name == "httpx"}
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        direct = (isinstance(f, ast.Attribute) and f.attr in _CLIENT_ATTRS
                  and isinstance(f.value, ast.Name) and f.value.id in modules)
        imported = isinstance(f, ast.Name) and f.id in by_name
        if (direct or imported) and not _marked(lines, node.lineno):
            hits.append(f"{path.name}:{node.lineno}")
    return hits


def _scope_files() -> list[Path]:
    out = []
    for p in SCOPE:
        out += sorted(p.rglob("*.py")) if p.is_dir() else [p]
    return [p for p in out if p.resolve() != HELPER.resolve()]


class RuleTests(unittest.TestCase):
    def test_fires_on_the_positive_fixture(self) -> None:
        hits = unguarded_clients(FIXTURES / "unguarded.py")
        self.assertEqual(len(hits), 4, hits)  # AsyncClient, imported AsyncClient, httpx.get, reason-less marker

    def test_fires_on_an_aliased_module_import(self) -> None:
        import tempfile
        src = "import httpx as hx\n\nasync def f(url):\n    async with hx.AsyncClient() as c:\n        return await c.get(url)\n"
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "aliased.py"
            path.write_text(src, encoding="utf-8")
            self.assertEqual(unguarded_clients(path), ["aliased.py:4"])

    def test_silent_on_the_negative_fixture(self) -> None:
        self.assertEqual(unguarded_clients(FIXTURES / "guarded.py"), [])

    def test_repair_code_has_no_unguarded_client(self) -> None:
        files = _scope_files()
        self.assertGreater(len(files), 3, "the rule's scope moved: update SCOPE")
        hits = [h for p in files for h in unguarded_clients(p)]
        self.assertEqual(hits, [], "use url_policy.audit_client, or add '# outbound: <why the URL is config>'")


class AllowlistTests(unittest.TestCase):
    def check(self, url, allowlist):
        from app.repair.url_policy import check_audit_url
        return check_audit_url(url, "audit_api_url", allowlist=allowlist, configured=())

    def test_wildcard_allows_nothing(self) -> None:
        for allowlist in ("*", " * ", "*:8765", "audit.lan,*"):
            self.assertIn("not allowed", self.check("http://127.0.0.1:6379", allowlist) or "", allowlist)
            self.assertIn("not allowed", self.check("http://192.168.1.20:8765", allowlist) or "", allowlist)

    def test_named_hosts_still_work_beside_a_wildcard(self) -> None:
        self.assertIsNone(self.check("http://audit.lan:8765", "audit.lan,*"))


class BrokerNodeUrlTests(unittest.TestCase):
    def vet(self, url):
        from app.gpu_lease import vet_node_url
        return vet_node_url(url)

    def test_bare_origins_pass(self) -> None:
        self.assertEqual(self.vet("http://jarrah:11434/"), "http://jarrah:11434")
        self.assertEqual(self.vet("https://192.168.1.109:11434"), "https://192.168.1.109:11434")

    def test_link_local_is_refused_in_any_form(self) -> None:
        from app.gpu_lease import OllamaLeaseUnavailable
        for url in ("http://169.254.169.254", "http://2852039166:11434", "http://0xA9FEA9FE/",
                    "http://[::ffff:169.254.169.254]:11434", "http://[fe80::1]:11434", "http://169.254.169.254.:80"):
            with self.assertRaises(OllamaLeaseUnavailable, msg=url):
                self.vet(url)

    def test_paths_credentials_and_other_schemes_are_refused(self) -> None:
        from app.gpu_lease import OllamaLeaseUnavailable
        for url in ("http://jarrah:11434/api/../admin", "http://user:pw@jarrah:11434", "http://jarrah:11434?x=1",
                    "file:///etc/passwd", "gopher://jarrah:70", "http://", "http://jarrah:99999", ""):
            with self.assertRaises(OllamaLeaseUnavailable, msg=url):
                self.vet(url)


if __name__ == "__main__":
    unittest.main()
