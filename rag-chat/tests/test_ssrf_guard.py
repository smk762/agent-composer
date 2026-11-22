"""SSRF class guard (ai-code-auditor pack-class-ssrf).

The repair loop is where request data names a server-side fetch target (``audit_api_url`` /
``audit_api_base``), and the GPU broker names the node every Ollama call of a lease goes to.
This file is the guard for that class:

* **Rule.** In ``app/repair/`` and ``app/routers/repair.py``, an outbound HTTP client is
  ``url_policy.audit_client`` (pinned, link-local refused at connect time). Any other client is
  allowed only with a ``# outbound: <reason>`` comment on the call's line or the line above,
  saying why its URL can't come from a request. "Any other client" is every way the stdlib and
  the usual libraries open a request: ``httpx`` (``AsyncClient``, ``Client``, ``get``…),
  ``urllib.request`` (``urlopen``, ``Request``, ``build_opener``), ``requests`` (``get``…,
  ``Session``), ``http.client`` (``HTTPConnection``, ``HTTPSConnection``) and ``aiohttp``
  (``ClientSession``, ``request``), however they are imported (``import urllib.request``,
  ``from urllib import request``, ``from requests import get``, an ``as`` alias). The rule is
  checked on positive and negative fixtures, then on the real tree, so a new unpinned client
  there fails CI.
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

_HTTP_VERBS = {"get", "post", "put", "patch", "delete", "head", "options", "request"}
# Fully qualified callables that open an outbound request (or build the client that does).
OUTBOUND = (
    {f"httpx.{a}" for a in _HTTP_VERBS | {"AsyncClient", "Client", "stream"}}
    | {f"urllib.request.{a}" for a in ("urlopen", "Request", "build_opener")}
    | {f"requests.{a}" for a in _HTTP_VERBS | {"Session", "session"}}
    | {f"requests.api.{a}" for a in _HTTP_VERBS}
    | {f"http.client.{a}" for a in ("HTTPConnection", "HTTPSConnection")}
    | {f"aiohttp.{a}" for a in ("ClientSession", "request")}
)
_MARKER = re.compile(r"#\s*outbound:\s*(.*)$")
MIN_REASON = 10


def _marked(lines: list[str], lineno: int) -> bool:
    for i in (lineno - 1, lineno - 2):  # the call's line, then the line above
        if 0 <= i < len(lines):
            m = _MARKER.search(lines[i])
            if m and len(m.group(1).strip()) >= MIN_REASON:
                return True
    return False


def _bindings(tree: ast.AST) -> dict[str, str]:
    """Local name -> the fully qualified module or object it is bound to by an import."""
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:  # `import urllib.request as ur` binds ur to urllib.request
                    out[alias.asname] = alias.name
                else:  # `import urllib.request` binds urllib
                    top = alias.name.split(".", 1)[0]
                    out[top] = top
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for alias in node.names:
                out[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return out


def _dotted(node: ast.AST) -> list[str] | None:
    """``a.b.c`` as ``["a", "b", "c"]``, or None when it isn't a plain dotted name."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    return [node.id, *reversed(parts)]


def qualified_call(func: ast.AST, bindings: dict[str, str]) -> str | None:
    """The fully qualified name *func* refers to through the file's imports, if it does."""
    parts = _dotted(func)
    if not parts or parts[0] not in bindings:
        return None
    return ".".join([bindings[parts[0]], *parts[1:]])


def unguarded_clients(path: Path) -> list[str]:
    """``file:line`` of every unmarked outbound HTTP call in *path*."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    lines = source.splitlines()
    bindings = _bindings(tree)
    hits = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and qualified_call(node.func, bindings) in OUTBOUND
                and not _marked(lines, node.lineno)):
            hits.append(f"{path.name}:{node.lineno}")
    return sorted(hits, key=lambda h: int(h.rsplit(":", 1)[1]))


def _scope_files() -> list[Path]:
    out = []
    for p in SCOPE:
        out += sorted(p.rglob("*.py")) if p.is_dir() else [p]
    return [p for p in out if p.resolve() != HELPER.resolve()]


class RuleTests(unittest.TestCase):
    def test_fires_on_the_positive_fixture(self) -> None:
        hits = unguarded_clients(FIXTURES / "unguarded.py")
        # httpx: AsyncClient, imported AsyncClient, httpx.get, reason-less marker; then urlopen,
        # urllib Request, requests.get, requests Session, http.client, aiohttp
        self.assertEqual(len(hits), 10, hits)

    def hits_in(self, src: str) -> list[str]:
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "case.py"
            path.write_text(src, encoding="utf-8")
            return unguarded_clients(path)

    def test_fires_on_an_aliased_module_import(self) -> None:
        src = "import httpx as hx\n\nasync def f(url):\n    async with hx.AsyncClient() as c:\n        return await c.get(url)\n"
        self.assertEqual(self.hits_in(src), ["case.py:4"])

    def test_fires_on_stdlib_and_other_clients_however_imported(self) -> None:
        for src in (
            "import urllib.request\n\ndef f(p):\n    with urllib.request.urlopen(p['url'], timeout=10) as r:\n        return r.read()\n",
            "import urllib.request as ur\n\ndef f(p):\n    with ur.urlopen(p['url']) as r:\n        return r.read()\n",
            "from urllib.request import urlopen\n\ndef f(p):\n    with urlopen(p['url']) as r:\n        return r.read()\n",
            "from urllib import request\n\ndef f(p):\n    with request.urlopen(p['url']) as r:\n        return r.read()\n",
            "import requests as rq\n\ndef f(p):\n    return [rq.post(p['url'])]\n",
            "from http import client\n\ndef f(p):\n    return [client.HTTPConnection(p['host'])]\n",
        ):
            self.assertEqual(len(self.hits_in(src)), 1, src)

    def test_a_reasoned_marker_or_a_non_request_call_is_quiet(self) -> None:
        self.assertEqual(self.hits_in(
            "import urllib.request\n\ndef f():\n    # outbound: OLLAMA_URL is operator config\n"
            "    return urllib.request.urlopen(OLLAMA_URL)\n"), [])
        self.assertEqual(self.hits_in("import urllib.parse\n\ndef f(u):\n    return urllib.parse.urlsplit(u)\n"), [])

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
            self.assertIn("not allowed", self.check("http://192.168.200.20:8765", allowlist) or "", allowlist)

    def test_named_hosts_still_work_beside_a_wildcard(self) -> None:
        self.assertIsNone(self.check("http://audit.lan:8765", "audit.lan,*"))


class BrokerNodeUrlTests(unittest.TestCase):
    def vet(self, url):
        from app.gpu_lease import vet_node_url
        return vet_node_url(url)

    def test_bare_origins_pass(self) -> None:
        self.assertEqual(self.vet("http://jarrah:11434/"), "http://jarrah:11434")
        self.assertEqual(self.vet("https://192.168.200.109:11434"), "https://192.168.200.109:11434")

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
