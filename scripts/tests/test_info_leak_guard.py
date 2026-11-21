"""Info-leak class guard (ai-code-auditor pack-class-info-leak). Stdlib only:

    python3 -m unittest discover -s scripts/tests

An error response is built for the client, not for us. In the services' code, an
``HTTPException`` or ``JSONResponse`` must not carry:

* **an upstream body**: any ``<x>.text`` in its detail or content (``exc.response.text``,
  ``r.text``). An upstream service's error page can name its hosts, paths, versions and
  internals; relay its status with a fixed message and log the body.
* **exception text in a server error**: the bound name of an enclosing ``except … as exc``
  (``str(exc)``, ``f"…{exc}"``, ``exc.args``) in a response whose status is 5xx or not a
  literal. Exception messages carry paths, hosts, SQL and stack internals. A 4xx may echo the
  exception: those are our own validation messages, raised for the client to read.
  ``exc.status_code``, ``exc.detail`` and ``exc.retry_after`` are fine: they are the fields
  our own ``HTTPException``/error types set for the client.

A deliberate one is marked ``# info-leak: <why this text is safe to show>`` on the call's
line or the line above. The rule fires on ``fixtures/info_leak_guard/leaky.py``, stays quiet
on ``safe.py``, and finds nothing in the services.
"""

import ast
import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "info_leak_guard"
SERVICES = ["rag-chat/app", "rag-ingest/app", "modernbert/app", "voice-stack", "infinity-manager"]
SKIP_DIRS = {"tests", "fixtures", "__pycache__", ".venv", "venv"}
SAFE_ATTRS = {"status_code", "detail", "retry_after"}
_MARKER = re.compile(r"#\s*info-leak:\s*(.*)$")
MIN_REASON = 10


def _marked(lines: list[str], lineno: int) -> bool:
    for i in (lineno - 1, lineno - 2):
        if 0 <= i < len(lines):
            m = _MARKER.search(lines[i])
            if m and len(m.group(1).strip()) >= MIN_REASON:
                return True
    return False


def _sink(call: ast.Call) -> tuple[ast.AST | None, ast.AST | None] | None:
    """(status, payload) of an HTTPException / JSONResponse call, else None."""
    name = call.func.id if isinstance(call.func, ast.Name) else getattr(call.func, "attr", None)
    kw = {k.arg: k.value for k in call.keywords if k.arg}
    if name == "HTTPException":
        status = kw.get("status_code", call.args[0] if call.args else None)
        return status, kw.get("detail", call.args[1] if len(call.args) > 1 else None)
    if name == "JSONResponse":
        status = kw.get("status_code", call.args[1] if len(call.args) > 1 else ast.Constant(200))
        return status, kw.get("content", call.args[0] if call.args else None)
    return None


def _server_error(status: ast.AST | None) -> bool:
    """5xx, or a status we can't read statically (it might be one)."""
    if isinstance(status, ast.Constant) and isinstance(status.value, int):
        return status.value >= 500
    return True


def _uses(payload: ast.AST, names: set[str]) -> bool:
    """Does *payload* use an exception name other than through a client-facing attribute?"""
    safe = {id(n.value) for n in ast.walk(payload) if isinstance(n, ast.Attribute) and n.attr in SAFE_ATTRS}
    return any(isinstance(n, ast.Name) and n.id in names and id(n) not in safe for n in ast.walk(payload))


def findings(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    out: list[str] = []

    def visit(node: ast.AST, bound: set[str]) -> None:
        if isinstance(node, ast.ExceptHandler) and node.name:
            bound = bound | {node.name}
        if isinstance(node, ast.Call) and (sink := _sink(node)) and sink[1] is not None:
            status, payload = sink
            upstream = any(isinstance(n, ast.Attribute) and n.attr == "text" for n in ast.walk(payload))
            exc_text = _server_error(status) and _uses(payload, bound)
            if (upstream or exc_text) and not _marked(lines, node.lineno):
                out.append(f"{path.relative_to(REPO)}:{node.lineno} {'upstream body' if upstream else 'exception text'}")
        for child in ast.iter_child_nodes(node):
            visit(child, bound)

    visit(ast.parse(source, filename=str(path)), set())
    return out


def service_files() -> list[Path]:
    files: list[Path] = []
    for d in SERVICES:
        root = REPO / d
        files += [p for p in root.rglob("*.py") if not SKIP_DIRS.intersection(p.relative_to(root).parts[:-1])]
    return sorted(files)


class InfoLeakGuardTests(unittest.TestCase):
    def test_fires_on_the_leaky_fixture(self) -> None:
        kinds = sorted(h.split(" ", 1)[1] for h in findings(FIXTURES / "leaky.py"))
        self.assertEqual(kinds, ["exception text", "exception text", "exception text",
                                 "upstream body", "upstream body"], findings(FIXTURES / "leaky.py"))

    def test_silent_on_the_safe_fixture(self) -> None:
        self.assertEqual(findings(FIXTURES / "safe.py"), [])

    def test_jsonresponse_positional_status_is_read(self) -> None:
        call = ast.parse('JSONResponse({"error": str(exc)}, 500)').body[0].value
        status, _ = _sink(call)
        self.assertTrue(_server_error(status))
        call = ast.parse('JSONResponse({"ok": 1})').body[0].value
        self.assertFalse(_server_error(_sink(call)[0]))

    def test_services_leak_nothing(self) -> None:
        files = service_files()
        self.assertGreater(len(files), 30, "the scan found too few files: SERVICES moved?")
        hits = [h for f in files for h in findings(f)]
        self.assertEqual(hits, [], "send a fixed message and log the detail, or mark '# info-leak: <reason>'")


if __name__ == "__main__":
    unittest.main()
