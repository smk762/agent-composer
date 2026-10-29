"""Smoke tests for AST-aware code chunking.

We avoid network/HF downloads — these tests are fully offline. We do require
``tree-sitter`` + ``tree-sitter-language-pack`` to be importable; if they
aren't, the AST tests are skipped (the fallback window path is still
exercised).
"""
from __future__ import annotations

import unittest

from app.code_chunker import (
    chunk_code,
    language_for_path,
)


def _have_treesitter() -> bool:
    try:
        from tree_sitter_language_pack import get_parser  # noqa: F401
        return True
    except Exception:
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


if __name__ == "__main__":
    unittest.main()
