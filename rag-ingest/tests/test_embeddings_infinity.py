"""Infinity embed/rerank responses are validated, not trusted."""

import asyncio
import unittest
from unittest import mock

from app import embeddings as em


class _Resp:
    def __init__(self, body, status=200):
        self.status_code, self._body, self.text = status, body, ""

    def json(self):
        return self._body


def _client_returning(body):
    class _Client:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, json=None):
            return _Resp(body)

    return _Client


def _embed(body, texts=("a", "b", "c")):
    with mock.patch.object(em.httpx, "AsyncClient", _client_returning(body)):
        return asyncio.run(em.InfinityEmbeddingClient(base_url="http://x").embed(list(texts), profile="code"))


def _rerank(body, docs=("d0", "d1", "d2"), top_k=2):
    with mock.patch.object(em.httpx, "AsyncClient", _client_returning(body)), \
            mock.patch.object(em, "RERANK_ENABLED", True):
        return asyncio.run(em.rerank("q", list(docs), top_k=top_k))


class InfinityEmbedTests(unittest.TestCase):
    def test_items_are_placed_by_index(self):
        data = [{"index": i, "embedding": [float(i)]} for i in (2, 0, 1)]
        self.assertEqual(_embed({"data": data}).vectors, [[0.0], [1.0], [2.0]])

    def test_null_data_is_a_runtime_error_not_a_type_error(self):
        with self.assertRaisesRegex(RuntimeError, "unexpected shape: NoneType"):
            _embed({"data": None})

    def test_numeric_data_is_a_runtime_error_not_a_type_error(self):
        with self.assertRaisesRegex(RuntimeError, "unexpected shape: int"):
            _embed({"data": 3})

    def test_missing_indices_are_refused(self):
        with self.assertRaisesRegex(RuntimeError, "index"):
            _embed({"data": [{"embedding": [1.0]}, {"embedding": [2.0]}, {"embedding": [3.0]}]})

    def test_duplicate_index_is_refused(self):
        data = [{"index": 0, "embedding": [1.0]}, {"index": 0, "embedding": [2.0]}, {"index": 1, "embedding": [3.0]}]
        with self.assertRaisesRegex(RuntimeError, "index"):
            _embed({"data": data})

    def test_out_of_range_index_is_refused(self):
        data = [{"index": 0, "embedding": [1.0]}, {"index": 1, "embedding": [2.0]}, {"index": 3, "embedding": [3.0]}]
        with self.assertRaisesRegex(RuntimeError, "index"):
            _embed({"data": data})

    def test_non_integer_index_is_refused(self):
        data = [{"index": "x", "embedding": [1.0]}, {"index": 1, "embedding": [2.0]}, {"index": 2, "embedding": [3.0]}]
        with self.assertRaisesRegex(RuntimeError, "index"):
            _embed({"data": data})


class RerankTests(unittest.TestCase):
    def test_sorted_filtered_and_truncated_locally(self):
        results = [
            {"index": 0, "relevance_score": 0.1},
            {"index": 7, "relevance_score": 0.99},  # out of range
            {"index": 2, "relevance_score": 0.9},
            {"index": 1, "relevance_score": 0.5},
        ]
        self.assertEqual(_rerank({"results": results}, top_k=2), [2, 1])

    def test_malformed_result_is_a_runtime_error(self):
        with self.assertRaises(RuntimeError):
            _rerank({"results": [{"index": 0}]})

    def test_disabled_keeps_input_order(self):
        self.assertEqual(asyncio.run(em.rerank("q", ["a", "b", "c"], top_k=2)), [0, 1])


if __name__ == "__main__":
    unittest.main()
