import os
import random
import sqlite3
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class NonceStoreConfig:
    db_path: str
    # How long to keep nonces around (seconds). Must be >= max timestamp skew window.
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
        with self._connect() as conn:
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

    def claim_or_reject(self, nonce: str, ts: int) -> None:
        """
        Claim a nonce; raise ValueError if the nonce was already used.
        """
        now = int(time.time())
        created_at = now

        try:
            with self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    "INSERT INTO nonces (nonce, ts, created_at) VALUES (?, ?, ?)",
                    (nonce, ts, created_at),
                )
                conn.execute("COMMIT")
        except sqlite3.IntegrityError as e:
            # Duplicate nonce: replay
            raise ValueError("Nonce already used") from e

        # Opportunistic cleanup to bound DB size.
        if self._cfg.cleanup_chance_denom > 0 and random.randint(1, self._cfg.cleanup_chance_denom) == 1:
            self.cleanup(now=now)

    def cleanup(self, now: int | None = None) -> int:
        now_i = int(time.time()) if now is None else int(now)
        cutoff = now_i - int(self._cfg.nonce_ttl_s)
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM nonces WHERE created_at < ?", (cutoff,))
            return int(cur.rowcount or 0)

