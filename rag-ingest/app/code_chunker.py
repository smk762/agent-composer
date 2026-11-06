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
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

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
    except Exception:
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
      large classes still produce per-method chunks instead of being dropped.
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

    def walk(node) -> None:
        if node.type in splits:
            if _emit(node, source, language, chunks):
                return  # don't recurse into already-emitted nodes
            # Too big or too small. If too big, recurse to find finer-grained
            # splits inside (e.g. methods of a giant class). If too small, just
            # skip — the parent walk will still consider sibling nodes.
            if (node.end_byte - node.start_byte) > CODE_CHUNK_MAX_CHARS:
                for child in node.children:
                    walk(child)
            return
        for child in node.children:
            walk(child)

    walk(tree.root_node)

    if not chunks:
        return _fallback_window(text, language)
    return chunks


def _fallback_window(text: str, language: str) -> List[CodeChunk]:
    out: List[CodeChunk] = []
    n = len(text)
    if n == 0:
        return out
    # A small overlap helps avoid splitting an identifier between chunks.
    overlap = min(200, max(0, CODE_CHUNK_MAX_CHARS // 10))
    step = max(1, CODE_CHUNK_MAX_CHARS - overlap)
    i = 0
    while i < n:
        slice_text = text[i : i + CODE_CHUNK_MAX_CHARS]
        if len(slice_text) >= CODE_CHUNK_MIN_CHARS or i == 0:
            out.append(
                CodeChunk(
                    text=slice_text,
                    language=language,
                    node_type="window",
                    symbol=None,
                    start_line=text.count("\n", 0, i) + 1,
                    end_line=text.count("\n", 0, i + len(slice_text)) + 1,
                )
            )
        i += step
    return out
