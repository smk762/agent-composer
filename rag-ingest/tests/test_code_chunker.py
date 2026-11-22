"""Smoke tests for AST-aware code chunking.

We avoid network/HF downloads — these tests are fully offline. We do require
``tree-sitter`` + ``tree-sitter-language-pack`` to be importable; if they
aren't, the AST tests are skipped (the fallback window path is still
exercised).
"""
from __future__ import annotations

import unittest
from unittest import mock

from app import code_chunker
from app.code_chunker import (
    _fallback_window,
    chunk_code,
    language_for_path,
)


def _covered(text: str, chunks) -> str:
    """Every non-blank line of ``text`` that no chunk's line range includes."""
    lines = text.splitlines()
    seen = set()
    for c in chunks:
        seen.update(range(c.start_line, c.end_line + 1))
    return "\n".join(line for n, line in enumerate(lines, 1) if line.strip() and n not in seen)


def _have_treesitter() -> bool:
    try:
        from tree_sitter_language_pack import get_parser  # noqa: F401
        return True
    except ImportError:
        return False


class LanguageDetectionTests(unittest.TestCase):
    def test_extension_to_language(self) -> None:
        self.assertEqual(language_for_path("foo/bar.py"), "python")
        self.assertEqual(language_for_path("FOO/BAR.PY"), "python")
        self.assertEqual(language_for_path("a/b.tsx"), "tsx")
        self.assertEqual(language_for_path("a/b.go"), "go")

    def test_unknown_extension(self) -> None:
        self.assertIsNone(language_for_path(""))
        self.assertIsNone(language_for_path("README"))
        self.assertIsNone(language_for_path("notes.txt"))


class FallbackWindowTests(unittest.TestCase):
    def test_unsupported_language_falls_back_to_window(self) -> None:
        text = "alpha\nbeta\ngamma\n" * 200
        chunks = chunk_code(text, "klingon")
        self.assertGreater(len(chunks), 0)
        self.assertTrue(all(c.node_type == "window" for c in chunks))
        self.assertTrue(all(c.symbol is None for c in chunks))

    def test_empty_text(self) -> None:
        self.assertEqual(chunk_code("", "python"), [])

    def test_short_final_window_is_not_dropped(self) -> None:
        # MAX=2000 -> overlap 200; a 400-char tail is shorter than MIN=500 and
        # the previous window reaches only 200 chars into it.
        text = "".join(f"{i:04d}\n" for i in range(440))  # 2200 chars
        with mock.patch.object(code_chunker, "CODE_CHUNK_MAX_CHARS", 2000), \
                mock.patch.object(code_chunker, "CODE_CHUNK_MIN_CHARS", 500):
            chunks = _fallback_window(text, "text")
        self.assertTrue(chunks[-1].text.endswith("0439\n"))
        self.assertEqual(chunks[-1].end_line, text.count("\n"))  # the 440th line ends on the last \n
        self.assertEqual(_covered(text, chunks), "")

    def test_window_ending_on_a_newline_ends_on_that_line(self) -> None:
        chunks = _fallback_window("a\nb\n", "text")
        self.assertEqual([(c.start_line, c.end_line) for c in chunks], [(1, 2)])

    def test_line_numbers_match_a_recount(self) -> None:
        text = "".join(f"line {i}\n" * (i % 3) + "x" * (i % 7) for i in range(400))
        with mock.patch.object(code_chunker, "CODE_CHUNK_MAX_CHARS", 300), \
                mock.patch.object(code_chunker, "CODE_CHUNK_MIN_CHARS", 50):
            chunks = _fallback_window(text, "text", line_offset=5)
        self.assertGreater(len(chunks), 5)
        pos = 0
        for c in chunks:
            i = text.index(c.text, max(0, pos - 300))
            pos = i + len(c.text)
            self.assertEqual(c.start_line, 5 + text.count("\n", 0, i) + 1)
            self.assertEqual(c.end_line, 5 + text.count("\n", 0, pos - 1) + 1)

    def test_line_numbers_are_not_rescanned_from_the_start(self) -> None:
        scanned = [0]

        class CountingStr(str):
            """Tallies the chars each scan or slice touches, however the chunker counts lines."""

            def count(self, sub, start=0, end=None):  # noqa: A003
                stop = len(self) if end is None else end
                scanned[0] += max(0, stop - start)
                return str.count(self, sub, start, end)

            def find(self, sub, start=0, end=None):
                pos = str.find(self, sub, start, end)
                stop = (len(self) if end is None else end) if pos == -1 else pos + len(sub)
                scanned[0] += max(0, stop - start)
                return pos

            def __getitem__(self, key):
                piece = str.__getitem__(self, key)
                scanned[0] += len(piece)  # a slice copies its chars: text[:i].count() rescans
                return piece

        text = CountingStr("abcdefghi\n" * 20_000)  # 200k chars, ~1100 windows at MAX=200
        with mock.patch.object(code_chunker, "CODE_CHUNK_MAX_CHARS", 200), \
                mock.patch.object(code_chunker, "CODE_CHUNK_MIN_CHARS", 20):
            chunks = _fallback_window(text, "text")
        self.assertEqual(chunks[-1].end_line, 20_000)
        self.assertGreater(scanned[0], 0, "the tally saw no work: the hook was bypassed")
        self.assertLessEqual(scanned[0], 4 * len(text))  # linear, not O(n^2 / step)

    def test_fully_covered_tail_is_not_repeated(self) -> None:
        text = "x" * 2000
        with mock.patch.object(code_chunker, "CODE_CHUNK_MAX_CHARS", 2000), \
                mock.patch.object(code_chunker, "CODE_CHUNK_MIN_CHARS", 100):
            chunks = _fallback_window(text, "text")
        self.assertEqual(len(chunks), 1)


@unittest.skipUnless(_have_treesitter(), "tree-sitter-language-pack not importable")
class PythonAstTests(unittest.TestCase):
    SOURCE = '''\
"""Module docstring."""

import os


def add(a, b):
    """Add two numbers."""
    return a + b


def multiply(a, b):
    return a * b


class Greeter:
    def __init__(self, name):
        self.name = name

    def greet(self):
        return f"Hello, {self.name}!"
'''

    def test_emits_function_and_class_chunks(self) -> None:
        chunks = chunk_code(self.SOURCE, "python")
        self.assertGreater(len(chunks), 0)

        symbols = {c.symbol for c in chunks if c.symbol}
        self.assertIn("add", symbols)
        self.assertIn("multiply", symbols)
        self.assertIn("Greeter", symbols)

        for c in chunks:
            self.assertEqual(c.language, "python")
            self.assertGreaterEqual(c.start_line, 1)
            self.assertGreaterEqual(c.end_line, c.start_line)
            self.assertNotEqual(c.node_type, "window")

    def test_lines_resolve_to_real_source_ranges(self) -> None:
        chunks = chunk_code(self.SOURCE, "python")
        lines = self.SOURCE.splitlines()
        for c in chunks:
            self.assertLessEqual(c.end_line, len(lines))
            joined = "\n".join(lines[c.start_line - 1 : c.end_line])
            self.assertIn(c.text.splitlines()[0].strip(), joined)


    def test_module_level_code_is_indexed_alongside_functions(self) -> None:
        source = (
            "import os\n\nSETTINGS = {'retries': 3, 'backoff': 'exponential'}\n\n\n"
            "def add(a, b):\n    return a + b\n\n\n"
            "if __name__ == '__main__':\n    print(add(1, 2))\n"
        )
        chunks = chunk_code(source, "python")
        self.assertIn("add", {c.symbol for c in chunks})
        self.assertEqual(_covered(source, chunks), "")
        joined = "".join(c.text for c in chunks)
        self.assertIn("SETTINGS", joined)
        self.assertIn("print(add(1, 2))", joined)
        module = [c for c in chunks if c.node_type == "module"]
        self.assertTrue(module and all(c.symbol is None for c in module))

    def test_oversized_function_without_nested_splits_is_windowed(self) -> None:
        body = "".join(f"    total += {i}  # step {i}\n" for i in range(60))
        source = "def small():\n    return 1\n\n\ndef big():\n    total = 0\n" + body + "    return total\n"
        with mock.patch.object(code_chunker, "CODE_CHUNK_MAX_CHARS", 600), \
                mock.patch.object(code_chunker, "CODE_CHUNK_MIN_CHARS", 50):
            chunks = chunk_code(source, "python")
        big = [c for c in chunks if c.symbol == "big"]
        self.assertGreater(len(big), 1)
        self.assertTrue(all(c.node_type == "function_definition" for c in big))
        self.assertTrue(all(len(c.text) <= 600 for c in chunks))
        self.assertIn("return total", big[-1].text)
        self.assertEqual(_covered(source, chunks), "")

    def test_oversized_class_keeps_fields_between_methods(self) -> None:
        methods = "".join(
            f"    def m{i}(self):\n        return {i}  # padding padding padding padding\n\n" for i in range(12)
        )
        source = "class Big:\n    LIMIT = 10\n    name: str = 'big'\n\n" + methods
        with mock.patch.object(code_chunker, "CODE_CHUNK_MAX_CHARS", 400), \
                mock.patch.object(code_chunker, "CODE_CHUNK_MIN_CHARS", 50):
            chunks = chunk_code(source, "python")
        self.assertIn("m0", {c.symbol for c in chunks})
        self.assertTrue(any("LIMIT = 10" in c.text for c in chunks))
        self.assertEqual(_covered(source, chunks), "")

    def test_deeply_nested_expression_does_not_hit_the_recursion_limit(self) -> None:
        import sys

        # ``x = 1 + 1 + ...`` parses as a left-nested chain one node deep per term.
        depth = sys.getrecursionlimit() * 3
        source = "def f():\n    return 1\n\nx = " + " + ".join(["1"] * depth) + "\n"
        chunks = chunk_code(source, "python")
        self.assertTrue(any(c.symbol == "f" for c in chunks))
        self.assertEqual(_covered(source, chunks), "")


if __name__ == "__main__":
    unittest.main()
