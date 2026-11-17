"""scripts/eval_retrieval.py: scoring. Stdlib only (httpx/qdrant_client are stubbed):

    python3 -m unittest discover -s scripts/tests
"""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

_stubs = {"httpx": types.ModuleType("httpx"), "qdrant_client": types.ModuleType("qdrant_client")}
_stubs["httpx"].AsyncClient = object
_stubs["qdrant_client"].AsyncQdrantClient = object
_SPEC = importlib.util.spec_from_file_location(
    "eval_retrieval", Path(__file__).resolve().parent.parent / "eval_retrieval.py")
eval_retrieval = importlib.util.module_from_spec(_SPEC)
with mock.patch.dict(sys.modules, _stubs):
    _SPEC.loader.exec_module(eval_retrieval)


class MetricsTests(unittest.TestCase):
    def test_mrr_ignores_ranks_past_10(self) -> None:
        m = eval_retrieval._metrics([1, 11, 20, None])
        self.assertEqual(m["mrr@10"], 0.25)
        self.assertEqual(m["recall@10"], 0.25)


class CompareTests(unittest.TestCase):
    def test_arms_on_one_collection_stay_separate(self) -> None:
        # e.g. --baseline audit_docs:none --candidate audit_docs:nomic: rows are keyed by arm.
        rows = [{"baseline": {"chunk": 3, "doc": 1}, "candidate": {"chunk": 1, "doc": 1}},
                {"baseline": {"chunk": None, "doc": 2}, "candidate": {"chunk": 2, "doc": 2}}]
        out = eval_retrieval._compare(rows, "chunk")
        self.assertEqual(out["baseline"]["recall@1"], 0.0)
        self.assertEqual(out["candidate"]["recall@1"], 0.5)
        self.assertEqual((out["candidate_better"], out["candidate_worse"]), (2, 0))


if __name__ == "__main__":
    unittest.main()
