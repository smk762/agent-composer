"""gpu-tenant.yaml (argus-iris) matches rag-chat's resident gateway.

argus-cortex validates the schema in CI (the gpu-tenant job); this checks what the schema
can't: each argus-http resident is one the gateway serves, at the gateway's paths.
"""

import re
import typing
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from app import gpu_lease
from app.routers import runtime

MANIFEST = Path(__file__).resolve().parents[2] / "gpu-tenant.yaml"
GATEWAY = "/api/runtime/residents/"
# Remote engines are other tenants' residents (Miso is voluptas's, on morpheus).
REMOTE = {"miso"}


@unittest.skipUnless(MANIFEST.exists(), "gpu-tenant.yaml is in the repo checkout, not the image")
class GpuTenantManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
        cls.http = {r["name"]: r for r in cls.manifest["residents"] if r["adapter"] == "argus-http"}

    def test_every_local_sidecar_is_listed(self) -> None:
        # As compose configures dragon: xtts and miso engines, and the Qwen tts sidecar
        # as a gateway-only resident (RUNTIME_EXTRA_RESIDENTS).
        engines = {"xtts": "http://xtts:8031", "miso": "http://morpheus:8034"}
        with patch.object(runtime, "TTS_ENGINES", engines), \
                patch.object(runtime, "RUNTIME_EXTRA_RESIDENTS", {"tts": "http://tts:8031"}):
            local = set(runtime._residents()) - REMOTE
        self.assertEqual(set(self.http), local)

    def test_argus_http_residents_go_through_the_gateway(self) -> None:
        actions = set(typing.get_args(runtime.ResidentAction))
        for name, r in self.http.items():
            with self.subTest(resident=name):
                self.assertTrue(r["base_url"].endswith(GATEWAY + name), r["base_url"])
                self.assertEqual(r.get("health_path", "/health"), "/health")
                # rag-chat adds ARGUS_ADMIN_TOKEN; the caller's Bearer is a rag-chat API key.
                self.assertFalse(r.get("admin_auth", False))
                for field in ("unload_path", "evict_path", "warm_path"):
                    if r.get(field):
                        self.assertIn(r[field].removeprefix("/admin/"), actions, field)
                self.assertEqual(r["unload_path"], "/admin/unload")

    def test_lease_kinds_match_the_code(self) -> None:
        """Every kind= an ollama_lease call (rag-chat and rag-ingest) passes is declared, and vice versa."""
        used = set()
        for app_dir in (MANIFEST.parent / "rag-chat" / "app", MANIFEST.parent / "rag-ingest" / "app"):
            for src in app_dir.rglob("*.py"):
                used |= set(re.findall(r'ollama_lease\([^)]*?kind="([a-z0-9-]+)"', src.read_text(encoding="utf-8"), re.S))
        declared = {k["kind"] for k in self.manifest["kinds"]}
        self.assertEqual(used, declared)

    def test_kind_footprints_match_the_lease_tables(self) -> None:
        """A kind's vram_mb/model_mb are what ollama_lease sends for its target model (#50)."""
        for k in self.manifest["kinds"]:
            model = k["target"].removeprefix("ollama:")
            with self.subTest(kind=k["kind"]):
                self.assertEqual(k["vram_mb"], gpu_lease.vram_mb(model))
                self.assertEqual(k.get("model_mb"), gpu_lease.model_mb(model))
                self.assertIsNotNone(k.get("model_mb"), "declare the weights so iris can reuse a loaded model")

    def test_singguard_is_pinned(self) -> None:
        guard = next(r for r in self.manifest["residents"] if r["name"] == "singguard")
        self.assertEqual(guard["adapter"], "ollama")
        self.assertTrue(guard["pinned"])
        self.assertTrue(guard["target"].startswith("ollama:hf.co/inclusionAI/SingGuard-4b-GGUF"))


if __name__ == "__main__":
    unittest.main()
