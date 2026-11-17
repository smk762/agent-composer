"""reembed.sync's plan: membership against the to-embed set, not a list scan per point."""

import asyncio
import unittest
from unittest import mock

from app import reembed


class Pid:
    """A point id that counts equality comparisons (a list scan compares against every entry)."""

    eq_calls = 0

    def __init__(self, v):
        self.v = v

    def __hash__(self):
        return hash(self.v)

    def __eq__(self, other):
        Pid.eq_calls += 1
        return isinstance(other, Pid) and self.v == other.v


class ReembedPlanTests(unittest.TestCase):
    def test_plan_does_not_scan_to_embed_per_point(self) -> None:
        n = 400
        src = {Pid(i): {"text": f"t{i}"} for i in range(n)}
        # half changed text (to embed), half unchanged (payload compare); separate but equal ids
        dst = {Pid(i): {"text": f"t{i}" if i % 2 else "old"} for i in range(n)}
        pages = iter([src, dst])

        async def scroll_all(q, collection):
            return next(pages)

        async def exists(q, collection):
            return True

        async def check(*a, **k):
            return None

        Pid.eq_calls = 0
        with mock.patch.object(reembed, "_scroll_all", scroll_all), \
                mock.patch.object(reembed, "_collection_exists", exists), \
                mock.patch.object(reembed, "_check_embedder", check):
            plan = asyncio.run(reembed.sync("a", "b", scheme="none", dry_run=True, client=mock.AsyncMock(),
                                            embedder=object(), log=lambda *_: None))
        self.assertEqual((plan["embed"], plan["payload_only"]), (n // 2, 0))
        self.assertLess(Pid.eq_calls, 10 * n)  # linear, not ~n * n / 2


if __name__ == "__main__":
    unittest.main()
