"""Ingest a repo's source files into the code collection (QDRANT_COLLECTION_CODE) via signed /ingest.

    python3 scripts/ingest_code.py rag-chat/app rag-ingest/app
    python3 scripts/ingest_code.py --root /home/smk/voluptas --repo voluptas src

Each file is sent as a ``kind: code`` doc with ``repo``, repo-relative ``path`` and the
checkout's ``commit_sha``; rag-ingest splits it per function/class (tree-sitter) and embeds
it with Infinity. Docs are identified by repo+path, so re-running replaces a file's old
chunks instead of duplicating them (files deleted from the repo are not retired: use
``POST /delete`` with ``kind: code``, ``repo`` and ``paths``).

Env: INGEST_URL (base URL, default http://127.0.0.1:9050) and INGEST_SHARED_SECRET (else
read from <agent-composer>/.env). Stdlib only.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import secrets
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DEFAULT_EXTS = ".py,.js,.ts,.tsx,.go,.rs"
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build"}


def shared_secret() -> str:
    if os.environ.get("INGEST_SHARED_SECRET"):
        return os.environ["INGEST_SHARED_SECRET"]
    env = HERE / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("INGEST_SHARED_SECRET="):
                return line.split("=", 1)[1].strip().strip("'\"")
    sys.exit("INGEST_SHARED_SECRET is not set (env or .env)")


def post(url: str, secret: str, payload: dict) -> tuple[int, object]:
    """Sign as rag-ingest expects: HMAC-SHA256(sha256(secret), "{ts}.{nonce}.{sha256(body)}")."""
    body = json.dumps(payload).encode()
    ts, nonce = str(int(time.time())), secrets.token_hex(16)
    msg = f"{ts}.{nonce}.{hashlib.sha256(body).hexdigest()}".encode()
    sig = hmac.new(hashlib.sha256(secret.encode()).digest(), msg, hashlib.sha256).hexdigest()
    req = urllib.request.Request(f"{url}/ingest", data=body, headers={
        "Content-Type": "application/json", "X-Timestamp": ts, "X-Nonce": nonce, "X-Signature": sig})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")[:400]


def source_files(root: Path, dirs: list[str], exts: set[str]) -> list[Path]:
    out = []
    for d in dirs:
        base = (root / d).resolve()
        candidates = [base] if base.is_file() else base.rglob("*")
        for p in candidates:
            if p.is_file() and p.suffix in exts and p.stat().st_size > 0 \
                    and not SKIP_DIRS.intersection(p.relative_to(root).parts):
                out.append(p)
    return sorted(set(out))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dirs", nargs="*", default=["."], help="dirs/files under --root to ingest (default: all)")
    ap.add_argument("--root", default=str(HERE), help="repo checkout (default: this repo)")
    ap.add_argument("--repo", help="repo name stored on each chunk (default: --root's dir name)")
    ap.add_argument("--source", help="source stored on each chunk (default: --repo)")
    ap.add_argument("--exts", default=DEFAULT_EXTS, help=f"comma-separated suffixes (default {DEFAULT_EXTS})")
    ap.add_argument("--batch", type=int, default=8, help="files per request")
    ap.add_argument("--url", default=os.environ.get("INGEST_URL", "http://127.0.0.1:9050"))
    ap.add_argument("--dry-run", action="store_true", help="list the files, send nothing")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    repo = args.repo or root.name
    exts = {e if e.startswith(".") else f".{e}" for e in args.exts.split(",") if e.strip()}
    files = source_files(root, args.dirs, exts)
    if not files:
        print("no matching files", file=sys.stderr)
        return 1
    if args.dry_run:
        print("\n".join(str(p.relative_to(root)) for p in files))
        print(f"{len(files)} files", file=sys.stderr)
        return 0

    try:
        sha = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        sha = None
    secret, url = shared_secret(), args.url.rstrip("/")
    chunks = 0
    for i in range(0, len(files), args.batch):
        batch = files[i:i + args.batch]
        docs = [{"kind": "code", "repo": repo, "path": str(p.relative_to(root)), "text": p.read_text(errors="replace"),
                 **({"commit_sha": sha} if sha else {})} for p in batch]
        status, res = post(url, secret, {"docs": docs, "source": args.source or repo, "tags": [f"repo:{repo}"]})
        if status != 200 or not isinstance(res, dict):
            print(f"HTTP {status} on {docs[0]['path']}..: {res}", file=sys.stderr)
            return 1
        chunks += res.get("upserted", 0)
        print(json.dumps({"files": [d["path"] for d in docs], "upserted": res.get("upserted"),
                          "pruned": res.get("pruned"), "collection": res.get("collection")}), flush=True)
    print(json.dumps({"repo": repo, "files": len(files), "chunks": chunks, "commit_sha": sha}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
