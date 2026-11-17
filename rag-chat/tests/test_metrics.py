"""HTTP metrics are labelled by route template, so ids and probes can't grow the series."""

import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from app.metrics import MetricsMiddleware


def _count(method: str, path: str, status: str) -> float:
    return REGISTRY.get_sample_value(
        "http_requests_total", {"method": method, "path": path, "status_code": status}
    ) or 0.0


def _paths() -> set[str]:
    return {
        sample.labels["path"]
        for metric in REGISTRY.collect()
        if metric.name == "http_requests"
        for sample in metric.samples
        if "path" in sample.labels
    }


class RouteLabelTests(unittest.TestCase):
    def setUp(self) -> None:
        app = FastAPI()
        app.add_middleware(MetricsMiddleware)

        @app.get("/test-metrics/clones/{companion_id}/download")
        def download(companion_id: str):
            return {"id": companion_id}

        self.client = TestClient(app)

    def test_parameterised_route_is_one_series(self) -> None:
        template = "/test-metrics/clones/{companion_id}/download"
        before = _count("GET", template, "200")
        for cid in ("alice", "bob", "carol-xyz"):
            self.assertEqual(self.client.get(f"/test-metrics/clones/{cid}/download").status_code, 200)
        self.assertEqual(_count("GET", template, "200") - before, 3)
        self.assertFalse(any("alice" in p or "carol-xyz" in p for p in _paths()))

    def test_unmatched_paths_share_one_bucket(self) -> None:
        before = _count("GET", "<unmatched>", "404")
        for probe in ("/test-metrics/nope-1", "/test-metrics/nope-2/deeper"):
            self.assertEqual(self.client.get(probe).status_code, 404)
        self.assertEqual(_count("GET", "<unmatched>", "404") - before, 2)
        self.assertFalse(any("nope-" in p for p in _paths()))


if __name__ == "__main__":
    unittest.main()
