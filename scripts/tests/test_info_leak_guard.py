"""Info-leak class guard (ai-code-auditor pack-class-info-leak). Stdlib only:

    python3 -m unittest discover -s scripts/tests

An error response is built for the client, not for us. In the services' code, an
``HTTPException``, or a ``JSONResponse`` / ``Response`` (and the other starlette response
classes) must not carry:

* **an upstream body**: any ``<x>.text`` in its detail or content (``exc.response.text``,
  ``r.text``). An upstream service's error page can name its hosts, paths, versions and
  internals; relay its status with a fixed message and log the body.
* **exception text in a server error**: the bound name of an enclosing ``except … as exc``
  (``str(exc)``, ``f"…{exc}"``, ``exc.args``) in a response whose status is 5xx or not a
  literal. Exception messages carry paths, hosts, SQL and stack internals. A 4xx may echo the
  exception: those are our own validation messages, raised for the client to read.
  ``exc.status_code``, ``exc.detail`` and ``exc.retry_after`` are fine: they are the fields
  our own ``HTTPException``/error types set for the client. Exception text reached without an
  ``as`` name counts the same: ``sys.exc_info()`` / ``sys.exception()``, directly or through a
  name assigned from them (``_, err, _ = sys.exc_info()``).
* **a traceback**, at any status: anything from the ``traceback`` module (``format_exc()``,
  ``format_exception(…)``, ``TracebackException…``), directly or through a name assigned from
  it, or the buffer a ``print_exc(file=buf)`` / ``print_exception(…, file=buf)`` wrote to. A
  stack trace is never a message for the client.

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
SERVICES = ["rag-chat/app", "rag-ingest/app", "modernbert/app", "infinity-manager"]
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


RESPONSES = {"JSONResponse", "Response", "PlainTextResponse", "HTMLResponse", "ORJSONResponse", "UJSONResponse"}


def _sink(call: ast.Call) -> tuple[ast.AST | None, ast.AST | None] | None:
    """(status, payload) of an HTTPException / response-class call, else None."""
    name = call.func.id if isinstance(call.func, ast.Name) else getattr(call.func, "attr", None)
    kw = {k.arg: k.value for k in call.keywords if k.arg}
    if name == "HTTPException":
        status = kw.get("status_code", call.args[0] if call.args else None)
        return status, kw.get("detail", call.args[1] if len(call.args) > 1 else None)
    if name in RESPONSES:
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


_PRINTERS = {"print_exc", "print_exception", "print_tb", "print_stack", "print_last"}
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


def _own_nodes(scope: ast.AST):
    """Every node in *scope* except those inside a nested function or class (their own scope)."""
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        n = stack.pop()
        yield n
        if not isinstance(n, _SCOPES):
            stack.extend(ast.iter_child_nodes(n))


class _Sources:
    """Where a file's traceback text and handler-less exception text come from."""

    def __init__(self, tree: ast.AST) -> None:
        self.tb_names: set[str] = set()   # names bound to the traceback module or its functions
        self.sys_names: set[str] = set()  # names bound to sys
        self.exc_info: set[str] = set()   # `from sys import exc_info` / `exception`
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name == "traceback":
                        self.tb_names.add(a.asname or a.name)
                    elif a.name == "sys":
                        self.sys_names.add(a.asname or a.name)
            elif isinstance(node, ast.ImportFrom) and node.module == "traceback":
                self.tb_names.update(a.asname or a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module == "sys":
                self.exc_info.update(a.asname or a.name for a in node.names if a.name in ("exc_info", "exception"))

    def traceback(self, expr: ast.AST, tainted: set[str]) -> bool:
        """Does *expr* use the traceback module, or a name holding its text?"""
        return any(isinstance(n, ast.Name) and (n.id in self.tb_names or n.id in tainted) for n in ast.walk(expr))

    def exc_info_call(self, expr: ast.AST) -> bool:
        """Does *expr* call ``sys.exc_info()`` / ``sys.exception()``?"""
        for n in ast.walk(expr):
            if not isinstance(n, ast.Call):
                continue
            f = n.func
            if isinstance(f, ast.Name) and f.id in self.exc_info:
                return True
            if (isinstance(f, ast.Attribute) and f.attr in ("exc_info", "exception")
                    and isinstance(f.value, ast.Name) and f.value.id in self.sys_names):
                return True
        return False

    def tainted(self, scope: ast.AST) -> tuple[set[str], set[str]]:
        """(traceback names, exception-text names) assigned in *scope*, flow-insensitively."""
        tb: set[str] = set()
        exc: set[str] = set()
        for _ in range(3):  # a name assigned from a tainted name is tainted (a few hops is plenty)
            for n in _own_nodes(scope):
                if isinstance(n, ast.Call):  # print_exc(file=buf) writes a traceback into buf
                    f = n.func
                    name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
                    if name in _PRINTERS and self.traceback(f, set()):
                        tb.update(k.value.id for k in n.keywords if k.arg == "file" and isinstance(k.value, ast.Name))
                value = getattr(n, "value", None)
                if not isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr)) or value is None:
                    continue
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                names = {t.id for tgt in targets for t in ast.walk(tgt) if isinstance(t, ast.Name)}
                if self.traceback(value, tb):
                    tb |= names
                elif self.exc_info_call(value) or _uses(value, exc):
                    exc |= names
        return tb, exc


def findings(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source, filename=str(path))
    src = _Sources(tree)
    out: list[str] = []

    def visit(node: ast.AST, bound: set[str], tb: set[str]) -> None:
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef)):
            tb_names, exc_names = src.tainted(node)
            tb, bound = tb | tb_names, bound | exc_names
        if isinstance(node, ast.ExceptHandler) and node.name:
            bound = bound | {node.name}
        if isinstance(node, ast.Call) and (sink := _sink(node)) and sink[1] is not None:
            status, payload = sink
            upstream = any(isinstance(n, ast.Attribute) and n.attr == "text" for n in ast.walk(payload))
            trace = src.traceback(payload, tb)
            exc_text = _server_error(status) and (_uses(payload, bound) or src.exc_info_call(payload))
            kind = "upstream body" if upstream else "traceback" if trace else "exception text" if exc_text else ""
            if kind and not _marked(lines, node.lineno):
                out.append(f"{path.relative_to(REPO)}:{node.lineno} {kind}")
        for child in ast.iter_child_nodes(node):
            visit(child, bound, tb)

    visit(tree, set(), set())
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
        self.assertEqual(kinds, ["exception text"] * 5 + ["traceback"] * 3 + ["upstream body"] * 2,
                         findings(FIXTURES / "leaky.py"))

    def hits_in(self, src: str) -> list[str]:
        import tempfile
        with tempfile.TemporaryDirectory(dir=REPO / "scripts" / "tests") as d:
            path = Path(d) / "case.py"
            path.write_text(src, encoding="utf-8")
            return [h.split(":", 1)[1] for h in findings(path)]

    def test_traceback_text_without_an_as_name(self) -> None:
        head = "import traceback\nfrom fastapi import HTTPException\n\ndef f(raw):\n    try:\n        return int(raw)\n    except Exception:\n"
        self.assertEqual(self.hits_in(head + "        raise HTTPException(status_code=500, detail=traceback.format_exc()) from None\n"),
                         ["8 traceback"])
        self.assertEqual(self.hits_in(head.replace("import traceback", "from traceback import format_exc as fe")
                                      + "        msg = fe()\n        raise HTTPException(500, msg) from None\n"),
                         ["9 traceback"])

    def test_names_assigned_in_one_function_do_not_taint_another(self) -> None:
        src = ("import sys\nfrom fastapi import HTTPException\n\ndef a():\n    err = sys.exc_info()[1]\n    return err\n\n"
               "def b(err):\n    raise HTTPException(500, detail=f'{err.code}')\n")
        self.assertEqual(self.hits_in(src), [])

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
