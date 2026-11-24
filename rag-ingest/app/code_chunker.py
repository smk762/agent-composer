"""Tree-sitter AST-aware chunking for code documents.

For code we want one chunk per function/class/method (with a fall-through for
files where no top-level splittable nodes exist) rather than fixed character
windows. The result preserves enough metadata that the chat side can render
useful citations like ``repo/path:fn_name (lines 42-87)``.

For unknown languages we fall back to a sliding character window so we never
silently drop a document.
"""
from __future__ import annotations

import os
from bisect import bisect_left
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# Soft caps to keep chunks within the embedder's reasonable input window.
CODE_CHUNK_MAX_CHARS = int(os.getenv("CODE_CHUNK_MAX_CHARS", "3000"))
CODE_CHUNK_MIN_CHARS = int(os.getenv("CODE_CHUNK_MIN_CHARS", "120"))

# Path-extension -> tree-sitter language id. Extend as needed.
EXT_TO_LANG: Dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".jsx": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
    ".cs": "csharp",
    ".swift": "swift",
    ".scala": "scala",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".sql": "sql",
    ".lua": "lua",
}

# Per-language tree-sitter node types worth emitting as standalone chunks.
SPLIT_TYPES: Dict[str, Set[str]] = {
    "python": {"function_definition", "class_definition", "decorated_definition"},
    "javascript": {
        "function_declaration",
        "class_declaration",
        "method_definition",
        "lexical_declaration",  # const/let with arrow funcs
    },
    "typescript": {
        "function_declaration",
        "class_declaration",
        "method_definition",
        "interface_declaration",
        "type_alias_declaration",
        "lexical_declaration",
    },
    "tsx": {
        "function_declaration",
        "class_declaration",
        "method_definition",
        "interface_declaration",
        "type_alias_declaration",
        "lexical_declaration",
    },
    "go": {"function_declaration", "method_declaration", "type_declaration"},
    "rust": {
        "function_item",
        "impl_item",
        "struct_item",
        "enum_item",
        "trait_item",
        "mod_item",
    },
    "java": {"method_declaration", "class_declaration", "interface_declaration"},
    "kotlin": {"function_declaration", "class_declaration", "object_declaration"},
    "ruby": {"method", "class", "module", "singleton_method"},
    "php": {"function_definition", "method_declaration", "class_declaration"},
    "c": {"function_definition", "struct_specifier"},
    "cpp": {"function_definition", "class_specifier", "struct_specifier", "namespace_definition"},
    "csharp": {"method_declaration", "class_declaration", "interface_declaration"},
    "swift": {"function_declaration", "class_declaration", "protocol_declaration"},
    "scala": {"function_definition", "class_definition", "trait_definition", "object_definition"},
}


@dataclass
class CodeChunk:
    text: str
    language: str
    node_type: str               # e.g. "function_definition" or "window" (fallback)
    symbol: Optional[str]        # extracted function/class name when available
    start_line: int              # 1-based, inclusive
    end_line: int                # 1-based, inclusive
    extras: Dict[str, str] = field(default_factory=dict)


def language_for_path(path: str) -> Optional[str]:
    """Return the tree-sitter language id for ``path`` or None."""
    if not path:
        return None
    _, ext = os.path.splitext(path.lower())
    return EXT_TO_LANG.get(ext)


def _safe_decode(source: bytes, start: int, end: int) -> str:
    return source[start:end].decode("utf-8", errors="replace")


def _node_symbol(node, source: bytes) -> Optional[str]:
    """Best-effort extraction of a function/class name from a tree-sitter node."""
    try:
        name = node.child_by_field_name("name")
    except Exception:  # noqa: S110 falls back to scanning the children below
        name = None
    if name is not None:
        return _safe_decode(source, name.start_byte, name.end_byte)

    for child in node.children:
        if child.type in ("identifier", "type_identifier", "field_identifier", "constant"):
            return _safe_decode(source, child.start_byte, child.end_byte)
    return None


def _emit(node, source: bytes, language: str, out: List[CodeChunk]) -> bool:
    text = _safe_decode(source, node.start_byte, node.end_byte)
    # AST nodes (a whole function/class) are inherently meaningful, so we
    # don't apply the lower-bound filter here — that's only useful for the
    # window-fallback path. We DO enforce the upper bound so a giant class
    # gets recursed into instead of dumped as one huge embedding.
    if len(text) > CODE_CHUNK_MAX_CHARS:
        return False
    out.append(
        CodeChunk(
            text=text,
            language=language,
            node_type=node.type,
            symbol=_node_symbol(node, source),
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
        )
    )
    return True


def chunk_code(text: str, language: str) -> List[CodeChunk]:
    """Walk the AST and emit one chunk per function/class.

    * Files with no splittable nodes (or in unsupported languages) fall back to
      a sliding character window.
    * Splittable nodes that exceed ``CODE_CHUNK_MAX_CHARS`` are recursed into so
      large classes still produce per-method chunks; one with nothing
      splittable inside is windowed whole instead of being dropped.
    * Code outside every emitted chunk (imports, constants, class fields) is
      emitted as ``"module"`` windows, so no non-blank part of a file is lost.
    """
    if not text:
        return []
    if language not in SPLIT_TYPES:
        return _fallback_window(text, language or "text")

    try:
        # Imported lazily so the module is importable in environments where
        # tree-sitter wheels aren't built (e.g. docs-only dev setups).
        from tree_sitter_language_pack import get_parser
    except Exception:
        return _fallback_window(text, language)

    try:
        parser = get_parser(language)
    except Exception:
        return _fallback_window(text, language)

    source = text.encode("utf-8")
    tree = parser.parse(source)
    splits = SPLIT_TYPES[language]
    chunks: List[CodeChunk] = []
    # Byte ranges [start, end) already inside some chunk, so whatever is left
    # (imports, constants, class fields, an oversized body) can be indexed too.
    covered: List[Tuple[int, int]] = []

    # Iterative depth-first walk: tree-sitter nests long expression chains
    # (``a + b + c ...``, deeply nested literals) one level per term, deep
    # enough to blow Python's recursion limit. A ``(node, before)`` entry is
    # the post-visit of an oversized split node, run once its subtree is done.
    stack: List[Tuple[object, Optional[int]]] = [(tree.root_node, None)]
    while stack:
        node, before = stack.pop()
        if before is not None:
            # A big node with nothing splittable inside (a long function) is
            # windowed whole instead of dropped.
            if len(chunks) == before:
                chunks.extend(
                    _window_bytes(
                        source, node.start_byte, node.end_byte, language, node.type, _node_symbol(node, source)
                    )
                )
                covered.append((node.start_byte, node.end_byte))
            continue
        if node.type in splits:
            if _emit(node, source, language, chunks):
                covered.append((node.start_byte, node.end_byte))
                continue  # don't descend into already-emitted nodes
            # Too big: descend to find finer-grained splits inside (e.g. the
            # methods of a giant class), then check in the post-visit.
            stack.append((node, len(chunks)))
        stack.extend((child, None) for child in reversed(node.children))

    if not chunks:
        return _fallback_window(text, language)

    # Whatever sits outside every emitted chunk (module-level statements,
    # imports, class bodies between methods) goes in as "module" windows, so
    # every non-blank byte of the file lands in some chunk.
    pos = 0
    for start, end in sorted(covered):
        if start > pos:
            chunks.extend(_window_bytes(source, pos, start, language, "module", None))
        pos = max(pos, end)
    if pos < len(source):
        chunks.extend(_window_bytes(source, pos, len(source), language, "module", None))
    chunks.sort(key=lambda c: (c.start_line, c.end_line))
    return chunks


def _window_bytes(
    source: bytes, start: int, end: int, language: str, node_type: str, symbol: Optional[str]
) -> List[CodeChunk]:
    """Window ``source[start:end]``, with line numbers relative to the whole file.

    Leading/trailing blank lines are trimmed; a blank range yields nothing.
    """
    raw = source[start:end]
    stripped = raw.strip()
    if not stripped:
        return []
    start += raw.index(stripped)
    end = start + len(stripped)
    first_line = source.count(b"\n", 0, start)
    out = _fallback_window(_safe_decode(source, start, end), language, line_offset=first_line)
    for chunk in out:
        chunk.node_type = node_type
        chunk.symbol = symbol
    return out


def _fallback_window(text: str, language: str, *, line_offset: int = 0) -> List[CodeChunk]:
    out: List[CodeChunk] = []
    n = len(text)
    if n == 0:
        return out
    # A small overlap helps avoid splitting an identifier between chunks.
    overlap = min(200, max(0, CODE_CHUNK_MAX_CHARS // 10))
    step = max(1, CODE_CHUNK_MAX_CHARS - overlap)
    # Newline offsets, found once: line numbers are then a bisect per window
    # rather than a rescan from the start of ``text`` (quadratic on big files).
    newlines: List[int] = []
    pos = text.find("\n")
    while pos != -1:
        newlines.append(pos)
        pos = text.find("\n", pos + 1)
    i = 0
    while i < n:
        slice_text = text[i : i + CODE_CHUNK_MAX_CHARS]
        if i > 0 and len(slice_text) < CODE_CHUNK_MIN_CHARS:
            # A short tail is never dropped: unless the previous window already
            # reaches the end, pull this last one back so it ends at ``n`` and
            # is at least MIN chars long.
            if i - step + CODE_CHUNK_MAX_CHARS >= n:
                break
            i = max(0, n - CODE_CHUNK_MIN_CHARS)
            slice_text = text[i:]
        out.append(
            CodeChunk(
                text=slice_text,
                language=language,
                node_type="window",
                symbol=None,
                start_line=line_offset + bisect_left(newlines, i) + 1,
                # Newlines before the window's last char: one it ends on
                # closes its last line rather than starting another.
                end_line=line_offset + bisect_left(newlines, i + len(slice_text) - 1) + 1,
            )
        )
        if i + len(slice_text) >= n:
            break
        i += step
    return out
