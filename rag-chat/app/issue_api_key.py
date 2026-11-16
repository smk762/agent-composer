"""Issue a rag-chat API key from the command line, for a service account such as argus-iris.

``POST /api-keys`` needs a credential already (an API key, Cloudflare Access, or
DEV_AUTH_BYPASS). This writes the key straight to the database, so it works on any
deployment. Run it inside the container, which has DATABASE_URL:

    docker exec rag-chat python -m app.issue_api_key --user argus-iris --name "iris broker"

It prints the secret once; only its hash is stored. Revoke it from /ui/api-keys while
acting as that user, or by setting ``revoked_at`` on its ``api_keys`` row.
"""

import argparse
import asyncio
import sys

from app.auth import issue_api_key
from app.db import SessionLocal


async def _main(user: str, name: str | None) -> str:
    async with SessionLocal() as db:
        rec, secret = await issue_api_key(db, user, name)
    print(f"issued key {rec.id} (prefix {rec.prefix}) for user {user!r}", file=sys.stderr)
    return secret


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m app.issue_api_key", description=__doc__.splitlines()[0])
    ap.add_argument("--user", required=True, help="user id the key acts as, e.g. argus-iris")
    ap.add_argument("--name", default=None, help="label shown in /ui/api-keys")
    args = ap.parse_args(argv)
    print(asyncio.run(_main(args.user, args.name)))


if __name__ == "__main__":
    main()
