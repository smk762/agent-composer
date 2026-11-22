"""Silent-failure class guard (ai-code-auditor pack-class-silent-failure). Stdlib only:

    python3 -m unittest discover -s scripts/tests

CI runs ruff ``B905,S110,S112`` over the repo. This test is the same rule in stdlib, plus one
thing ruff doesn't check: a ``noqa`` for it must give a reason.

* ``zip()`` with arguments says ``strict=``: ``True`` when the lengths must match (a mismatch
  raises), ``False`` when shorter is intended.
* An ``except`` (bare, ``Exception`` or ``BaseException``) whose whole body is ``pass`` or
  ``continue`` swallows the error. Log it, narrow it, or mark the ``except`` line
  ``# noqa: S110 <why nothing is lost>`` (``S112`` for ``continue``).
* So does one whose whole body is a single ``return`` of an empty value: a bare ``return``, or
  ``{}``, ``[]``, ``()``, ``""``, ``set()``, ``dict()``, ``list()``, ``0``, ``False`` or ``None``.
  A corrupt state file read as ``{}`` looks like an empty state, not a failure. Ruff has no rule
  for it, so it is reported as ``S110`` and takes the same reasoned ``# noqa: S110 <reason>``.

It fires on ``fixtures/silent_failure_guard/unguarded.py``, stays quiet on ``guarded.py``, and
finds nothing in the repo's own code.
"""

import ast
import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "silent_failure_guard"
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", "fixtures", "build", "dist"}
MIN_REASON = 10
_BROAD = {"Exception", "BaseException"}


def _reasoned_noqa(line: str, code: str) -> bool:
    """A ``# noqa:`` naming *code* whose text after the whole code list is a real reason."""
    m = re.search(r"#\s*noqa:\s*((?:[A-Z]+[0-9]+[,\s]*)+)(.*)$", line)
    if not m or code not in re.findall(r"[A-Z]+[0-9]+", m.group(1)):
        return False
    return len(m.group(2).strip()) >= MIN_REASON


def _broad(handler: ast.ExceptHandler) -> bool:
    """Bare ``except``, a broad name, or a tuple containing one."""
    t = handler.type
    if t is None:
        return True
    elts = t.elts if isinstance(t, ast.Tuple) else [t]
    return any(isinstance(e, ast.Name) and e.id in _BROAD for e in elts)


def _empty_return(stmt: ast.stmt) -> bool:
    """``return`` alone, or ``return`` of an empty literal / empty builtin constructor / 0 / False / None."""
    if not isinstance(stmt, ast.Return):
        return False
    v = stmt.value
    if v is None:
        return True
    if isinstance(v, ast.Constant):
        x = v.value
        return x is None or (isinstance(x, str) and x == "") or (type(x) in (int, bool) and x == 0)
    if isinstance(v, ast.Dict):
        return not v.keys
    if isinstance(v, (ast.List, ast.Tuple)):
        return not v.elts
    return (isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id in ("set", "dict", "list")
            and not v.args and not v.keywords)


def findings(path: Path) -> list[str]:
    """``file:line code`` for each silent-failure pattern in *path* without a reasoned noqa."""
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    out = []

    def hit(lineno: int, code: str) -> None:
        if not _reasoned_noqa(lines[lineno - 1], code):
            out.append(f"{path.relative_to(REPO)}:{lineno} {code}")

    for node in ast.walk(ast.parse(source, filename=str(path))):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "zip"
                and node.args and not any(k.arg == "strict" for k in node.keywords)):
            hit(node.lineno, "B905")
        elif isinstance(node, ast.ExceptHandler) and len(node.body) == 1:
            broad = _broad(node)
            if broad and isinstance(node.body[0], ast.Pass):
                hit(node.lineno, "S110")
            elif broad and isinstance(node.body[0], ast.Continue):
                hit(node.lineno, "S112")
            elif broad and _empty_return(node.body[0]):
                hit(node.lineno, "S110")
    return out


def repo_files() -> list[Path]:
    return sorted(p for p in REPO.rglob("*.py")
                  if not SKIP_DIRS.intersection(p.relative_to(REPO).parts[:-1]))


class SilentFailureGuardTests(unittest.TestCase):
    def test_fires_on_the_positive_fixture(self) -> None:
        codes = sorted(h.rsplit(" ", 1)[1] for h in findings(FIXTURES / "unguarded.py"))
        # zip, except-continue, bare except-pass, a noqa that gives no reason, return {} and a bare return
        self.assertEqual(codes, ["B905", "S110", "S110", "S110", "S110", "S112"], findings(FIXTURES / "unguarded.py"))

    def test_empty_returns(self) -> None:
        empty = ("return", "return {}", "return []", "return ()", 'return ""', "return set()", "return dict()",
                 "return list()", "return 0", "return False", "return None")
        for stmt in empty:
            self.assertTrue(_empty_return(ast.parse(stmt).body[0]), stmt)
        for stmt in ("return {'ok': False}", "return [0]", "return 1", "return True", "return 'x'", "return 0.0",
                     "return b''", "return list(xs)", "return dict(a=1)", "return fallback", "x = {}"):
            self.assertFalse(_empty_return(ast.parse(stmt).body[0]), stmt)

    def test_silent_on_the_negative_fixture(self) -> None:
        self.assertEqual(findings(FIXTURES / "guarded.py"), [])

    def test_trailing_codes_are_not_a_reason(self) -> None:
        self.assertFalse(_reasoned_noqa("except Exception:  # noqa: BLE001, S110, E722", "S110"))
        self.assertFalse(_reasoned_noqa("except Exception:  # noqa: S110, BLE001, E722", "S110"))
        self.assertTrue(_reasoned_noqa("except Exception:  # noqa: S110 best-effort cleanup only", "S110"))
        self.assertFalse(_reasoned_noqa("x  # noqa: S112 best-effort cleanup only", "S110"))

    def test_tuple_with_a_broad_type_is_broad(self) -> None:
        for src in ("try:\n    f()\nexcept (Exception, OSError):\n    pass\n",
                    "try:\n    f()\nexcept (Exception,):\n    pass\n"):
            self.assertTrue(_broad(ast.parse(src).body[0].handlers[0]), src)
        self.assertFalse(_broad(ast.parse("try:\n    f()\nexcept (OSError, ValueError):\n    pass\n").body[0].handlers[0]))

    def test_repo_has_no_silent_failure(self) -> None:
        files = repo_files()
        self.assertGreater(len(files), 50, "the scan found too few files: REPO moved?")
        hits = [h for f in files for h in findings(f)]
        self.assertEqual(hits, [], "say zip(strict=...), log or narrow the except, or add '# noqa: S110 <reason>'")


if __name__ == "__main__":
    unittest.main()
