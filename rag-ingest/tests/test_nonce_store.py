"""Replay protection: the SQLite and Redis nonce stores."""

import os
import sqlite3
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("INGEST_NONCE_DB", os.path.join(tempfile.mkdtemp(prefix="ingest-test-"), "nonces.db"))

from app import nonce_store as ns
from app.nonce_store import NonceStore, NonceStoreConfig, RedisNonceStore, RedisNonceStoreConfig

T0 = 1_700_000_000


def _sqlite(ttl: int = 100, denom: int = 0) -> NonceStore:
    path = os.path.join(tempfile.mkdtemp(prefix="nonce-test-"), "n.db")
    return NonceStore(NonceStoreConfig(db_path=path, nonce_ttl_s=ttl, cleanup_chance_denom=denom))


class SqliteNonceStoreTests(unittest.TestCase):
    def test_repeat_is_rejected(self) -> None:
        store = _sqlite()
        store.claim_or_reject("n1", T0)
        with self.assertRaises(ValueError):
            store.claim_or_reject("n1", T0)

    def test_cleanup_failure_does_not_fail_a_committed_claim(self) -> None:
        store = _sqlite(denom=1)  # clean up on every claim
        with mock.patch.object(store, "cleanup", side_effect=sqlite3.OperationalError("database is locked")):
            store.claim_or_reject("n1", T0)  # must not raise
        with self.assertRaises(ValueError):
            store.claim_or_reject("n1", T0)  # and the claim stuck

    def test_future_dated_nonce_outlives_its_first_sighting_ttl(self) -> None:
        # Seen at T0 with ts = T0 + 50 (inside a 50 s skew window): it validates until
        # T0 + 100, so a cleanup at T0 + 101 (TTL 100 after created_at) must keep it.
        store = _sqlite(ttl=100)
        with mock.patch.object(ns.time, "time", return_value=T0):
            store.claim_or_reject("n1", T0 + 50)
        store.cleanup(now=T0 + 101)
        with self.assertRaises(ValueError):
            store.claim_or_reject("n1", T0 + 50)

    def test_cleanup_evicts_once_both_clocks_are_past_the_ttl(self) -> None:
        store = _sqlite(ttl=100)
        with mock.patch.object(ns.time, "time", return_value=T0):
            store.claim_or_reject("n1", T0 + 50)
        self.assertEqual(store.cleanup(now=T0 + 151), 1)


class RedisNonceStoreTests(unittest.TestCase):
    def test_expiry_counts_from_the_signed_ts_when_it_is_later(self) -> None:
        store = RedisNonceStore(RedisNonceStoreConfig(redis_url="redis://127.0.0.1:1/0", nonce_ttl_s=100))
        store._r = mock.Mock()
        store._r.set.return_value = True
        with mock.patch.object(ns.time, "time", return_value=T0):
            store.claim_or_reject("n1", T0 + 50)
            store.claim_or_reject("n2", T0 - 50)
        first, second = store._r.set.call_args_list
        self.assertEqual(first.kwargs["exat"], T0 + 150)
        self.assertEqual(second.kwargs["exat"], T0 + 100)
        self.assertTrue(first.kwargs["nx"])


if __name__ == "__main__":
    unittest.main()
