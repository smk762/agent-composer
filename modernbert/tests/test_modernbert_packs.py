"""Classify-head sizing, CPU-cached label embeddings, the preload flag and /classify errors.
No torch or GPU needed: torch and transformers are stubbed when absent.

Run from modernbert/: python -m pytest tests/test_modernbert_packs.py -v
"""
import contextlib
import os
import sys
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _stub_torch() -> None:
    torch = sys.modules.get("torch")
    if torch is None:
        torch = types.ModuleType("torch")
        sys.modules["torch"] = torch
    if getattr(torch, "__file__", None):
        return  # real torch
    for name, value in (("float16", "float16"), ("float32", "float32"), ("Tensor", object),
                        ("no_grad", contextlib.nullcontext)):
        if not hasattr(torch, name):
            setattr(torch, name, value)
    if not hasattr(torch, "cuda"):
        torch.cuda = types.SimpleNamespace(is_available=lambda: True, empty_cache=lambda: None)
    if "torch.nn" not in sys.modules:
        nn = types.ModuleType("torch.nn")
        functional = types.ModuleType("torch.nn.functional")
        nn.functional = functional
        torch.nn = nn
        sys.modules["torch.nn"] = nn
        sys.modules["torch.nn.functional"] = functional
    if "transformers" not in sys.modules:
        tf = types.ModuleType("transformers")
        tf.AutoModel = tf.AutoModelForSequenceClassification = tf.AutoTokenizer = object
        sys.modules["transformers"] = tf


_stub_torch()

import app.classifier as classifier_mod  # noqa: E402
from app.classifier import Classifier  # noqa: E402
from app.model import ModelManager, ModelState  # noqa: E402


class FakeModel:
    def to(self, _device):
        return self


class ClassifyHeadSizeTests(unittest.IsolatedAsyncioTestCase):
    def _manager(self) -> ModelManager:
        mgr = ModelManager()
        mgr.task = "classify"
        mgr._target_device = "cpu"
        mgr.keep_alive_gpu = mgr.keep_alive_cpu = -1
        self.loads = []

        def load(n):
            self.loads.append(n)
            return object(), FakeModel()

        mgr._load_sync = load
        return mgr

    async def test_other_label_count_reloads_the_head(self):
        mgr = self._manager()
        async with mgr.use(num_labels=3):
            pass
        async with mgr.use(num_labels=5):
            pass
        self.assertEqual(self.loads, [3, 5])

    async def test_unspecified_or_same_count_keeps_the_loaded_head(self):
        mgr = self._manager()
        async with mgr.use(num_labels=3):
            pass
        async with mgr.use():
            pass
        await mgr.ensure_ready()
        async with mgr.use(num_labels=3):
            pass
        self.assertEqual(self.loads, [3])


class FakeTensor:
    def __init__(self, device):
        self.device = device

    def cpu(self):
        return FakeTensor("cpu")


class LabelCacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_label_embeddings_are_cached_on_cpu(self):
        mgr = types.SimpleNamespace(state=ModelState.GPU, task="zero_shot", model_id="fake")
        clf = Classifier(mgr)
        clf.labels = ["a", "b"]
        clf._encode_sync = lambda texts: FakeTensor("cuda:0")
        functional = types.SimpleNamespace(normalize=lambda x, dim: x)
        with mock.patch.object(classifier_mod, "F", functional):
            await clf._embed_labels()
        self.assertEqual(clf._label_embeddings.device, "cpu")


def _import_main():
    try:
        import httpx  # noqa: F401
        import app.main as main
    except ImportError as exc:  # fastapi/httpx absent
        raise unittest.SkipTest(f"needs fastapi + httpx: {exc}")
    return main


class PreloadFlagTests(unittest.IsolatedAsyncioTestCase):
    async def test_boolean_style_preload_value_does_not_break_startup(self):
        main = _import_main()
        preloaded = []

        async def embed_labels():
            preloaded.append(True)

        @contextlib.asynccontextmanager
        async def use(num_labels=None):
            yield main.manager

        async def unload():
            return None

        with mock.patch.dict(os.environ, {"MODERNBERT_PRELOAD": "true"}), \
                mock.patch.object(main.manager, "use", use), \
                mock.patch.object(main.manager, "unload", unload), \
                mock.patch.object(main.classifier, "_embed_labels", embed_labels):
            async with main.lifespan(main.app):
                pass
        self.assertEqual(preloaded, [True])

        with mock.patch.dict(os.environ, {"MODERNBERT_PRELOAD": "off"}), \
                mock.patch.object(main.manager, "unload", unload), \
                mock.patch.object(main.classifier, "_embed_labels", embed_labels):
            async with main.lifespan(main.app):
                pass
        self.assertEqual(preloaded, [True])


class ClassifyErrorTests(unittest.IsolatedAsyncioTestCase):
    async def _post(self, main, exc):
        import httpx

        async def boom(_req):
            raise exc

        transport = httpx.ASGITransport(app=main.app, raise_app_exceptions=False)
        with mock.patch.object(main.classifier, "classify", boom):
            async with httpx.AsyncClient(transport=transport, base_url="http://mb") as client:
                return await client.post("/classify", json={"text": "hi"})

    async def test_model_unavailable_is_a_generic_503(self):
        main = _import_main()
        from app.model import ModelUnavailableError

        resp = await self._post(main, ModelUnavailableError("/secret/model/path: CUDA error"))
        self.assertEqual(resp.status_code, 503)
        self.assertNotIn("secret", resp.text)

    async def test_programming_error_is_not_a_503(self):
        main = _import_main()
        resp = await self._post(main, TypeError("/secret/internal detail"))
        self.assertEqual(resp.status_code, 500)
        self.assertNotIn("secret", resp.text)

    async def test_http_exception_passes_through(self):
        main = _import_main()
        resp = await self._post(main, main.HTTPException(status_code=413, detail="too long"))
        self.assertEqual(resp.status_code, 413)


if __name__ == "__main__":
    unittest.main()
