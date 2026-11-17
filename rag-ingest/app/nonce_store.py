"""Replay-protection nonce stores for signed ingest requests: SQLite (atomic INSERT claim) and Redis (SET NX EX) backends with TTL expiry."""

import logging
import os
import random
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass

import redis

log = logging.getLogger("rag-ingest.nonce")


@dataclass(frozen=True)
class NonceStoreConfig:
    db_path: str
    # How long to keep nonces around (seconds), counted from the later of the signed ``ts``
    # and the first sighting. Must be >= the max timestamp skew window: a request stays
    # acceptable until ``ts + skew``, so its nonce must outlive that.
    nonce_ttl_s: int
    # Randomly perform cleanup ~1/cleanup_chance_denom requests.
    cleanup_chance_denom: int = 50


class NonceStore:
    """
    Minimal persistent nonce store for replay protection.

    Uses SQLite with a UNIQUE primary key on nonce, so we can atomically
    "claim" a nonce via INSERT and reject duplicates reliably.
    """

    def __init__(self, cfg: NonceStoreConfig):
        self._cfg = cfg
        os.makedirs(os.path.dirname(cfg.db_path) or ".", exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        # Short-lived connections keep things simple and safe under uvicorn workers.
        conn = sqlite3.connect(self._cfg.db_path, timeout=5, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        # ``with conn`` only commits/rolls back; closing() actually releases the connection.
        with closing(self._connect()) as conn, conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS nonces (
                    nonce TEXT PRIMARY KEY,
                    ts INTEGER NOT NULL,
                    created_at INTEGER NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_nonces_ts ON nonces(ts)")
            # cleanup() filters on created_at (and ts): index it so eviction isn't a full scan.
            conn.execute("CREATE INDEX IF NOT EXISTS idx_nonces_created_at ON nonces(created_at)")

    def claim_or_reject(self, nonce: str, ts: int) -> None:
        """
        Claim a nonce; raise ValueError if the nonce was already used.
        """
        now = int(time.time())
        created_at = now

        try:
            with closing(self._connect()) as conn, conn:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    "INSERT INTO nonces (nonce, ts, created_at) VALUES (?, ?, ?)",
                    (nonce, ts, created_at),
                )
                conn.execute("COMMIT")
        except sqlite3.IntegrityError as e:
            # Duplicate nonce: replay
            raise ValueError("Nonce already used") from e

        # Opportunistic cleanup to bound DB size. The claim is committed, so housekeeping
        # (e.g. "database is locked" under concurrent workers) must not fail the request:
        # a retry would be rejected as a replay.
        if self._cfg.cleanup_chance_denom > 0 and random.randint(1, self._cfg.cleanup_chance_denom) == 1:
            try:
                self.cleanup(now=now)
            except Exception as e:
                log.warning("nonce cleanup skipped: %s", e)

    def cleanup(self, now: int | None = None) -> int:
        now_i = int(time.time()) if now is None else int(now)
        cutoff = now_i - int(self._cfg.nonce_ttl_s)
        with closing(self._connect()) as conn, conn:
            # Both clocks must be past the TTL: a future-dated ``ts`` keeps validating
            # until ts + skew, so evicting on created_at alone would reopen the replay.
            cur = conn.execute("DELETE FROM nonces WHERE created_at < ? AND ts < ?", (cutoff, cutoff))
            return int(cur.rowcount or 0)


@dataclass(frozen=True)
class RedisNonceStoreConfig:
    redis_url: str
    nonce_ttl_s: int
    key_prefix: str = "ingest:nonce:"


class RedisNonceStore:
    """
    Redis-backed nonce store for replay protection.

    Uses atomic SET key value NX EX ttl to claim a nonce once.
    """

    def __init__(self, cfg: RedisNonceStoreConfig):
        self._cfg = cfg
        # Bounded timeouts so an unreachable host fails fast (startup ping +
        # per-request claims) instead of hanging the ingest worker.
        self._r = redis.Redis.from_url(
            cfg.redis_url,
            socket_connect_timeout=3,
            socket_timeout=3,
        )

    def ping(self) -> None:
        """Raise if the Redis server is unreachable (used for startup probing)."""
        self._r.ping()

    def _key(self, nonce: str) -> str:
        return f"{self._cfg.key_prefix}{nonce}"

    def claim_or_reject(self, nonce: str, ts: int) -> None:
        # Expire TTL after the later of now and the signed ts (see NonceStoreConfig.nonce_ttl_s).
        expire_at = max(int(time.time()), int(ts)) + int(self._cfg.nonce_ttl_s)
        ok = self._r.set(self._key(nonce), str(ts), exat=expire_at, nx=True)
        if not ok:
            raise ValueError("Nonce already used")

    def cleanup(self, now: int | None = None) -> int:
        # Redis handles TTL expiry automatically.
        return 0

