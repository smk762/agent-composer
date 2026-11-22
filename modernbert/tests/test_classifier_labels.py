"""Classifier label handling. No torch or GPU needed: torch and transformers are stubbed.

Run from modernbert/: python -m pytest tests/test_classifier_labels.py -v
"""
import asyncio
import contextlib
import os
import sys
import threading
import types
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class _Vec(list):
    """Just enough of a tensor for the supervised path: ``.squeeze(0).cpu().tolist()``."""

    def squeeze(self, _dim):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return list(self)


def _stub_torch() -> None:
    torch = sys.modules.get("torch")
    if torch is None:
        torch = types.ModuleType("torch")
        torch.float16, torch.float32 = "float16", "float32"
        torch.cuda = types.SimpleNamespace(is_available=lambda: True, empty_cache=lambda: None)
        sys.modules["torch"] = torch
    if getattr(torch, "__file__", None):
        return  # real torch
    torch.Tensor = getattr(torch, "Tensor", object)
    torch.no_grad = contextlib.nullcontext
    torch.softmax = lambda logits, dim: _Vec(logits)
    torch.sigmoid = lambda logits: _Vec(logits)
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

from app.classifier import Classifier  # noqa: E402
from app.model import ModelState  # noqa: E402


class FakeManager:
    """A ready manager whose model blocks in its worker thread until released."""

    model_id = "fake"
    task = "classify"
    state = ModelState.GPU

    def __init__(self, logits):
        self.started = threading.Event()
        self.release = threading.Event()
        manager = self

        class Inputs(dict):
            def to(self, _device):
                return self

        self.tokenizer = lambda *a, **k: Inputs()

        def model(**_inputs):
            manager.started.set()
            manager.release.wait(5)
            return types.SimpleNamespace(logits=list(logits))

        self.model = model

    def current_device(self):
        return "cpu"

    @contextlib.asynccontextmanager
    async def use(self, num_labels=2):
        yield self


async def _until(event: threading.Event) -> None:
    for _ in range(200):
        if event.is_set():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("inference never started")


class SupervisedLabelTests(unittest.IsolatedAsyncioTestCase):
    def _classifier(self, labels, logits) -> Classifier:
        clf = Classifier(FakeManager(logits))
        clf.labels = labels
        clf.threshold = 0.0
        return clf

    async def test_label_update_mid_inference_does_not_rename_scores(self):
        clf = self._classifier(["spam", "ham"], [0.9, 0.1])
        task = asyncio.create_task(clf._supervised("hello", 10, 0.0))
        await _until(clf.manager.started)
        clf.labels = ["x", "y", "z"]  # what update_labels() does first
        clf.manager.release.set()
        res = await task
        self.assertEqual(set(res.raw), {"spam", "ham"})
        self.assertEqual(res.labels[0], "spam")

    async def test_head_and_label_count_mismatch_is_an_error_not_a_truncation(self):
        clf = self._classifier(["a", "b", "c"], [0.9, 0.1])  # head loaded for 2 labels
        clf.manager.release.set()
        with self.assertRaises(RuntimeError):
            await clf._supervised("hello", 10, 0.0)


class _TruncatingTokenizer:
    """Word-level stand-in for an HF tokenizer's truncation=True: one sequence loses its tail;
    a pair loses tokens from the longer sequence first (longest_first)."""

    def __init__(self, max_length):
        self.max_length = max_length
        self.seen = []

    def __call__(self, first, second=None, truncation=False, max_length=None, **_kw):
        a = (first[0] if isinstance(first, list) else first).split()
        b = (second[0] if isinstance(second, list) else second).split() if second else []
        while truncation and len(a) + len(b) > max_length:
            if len(a) >= len(b):
                a.pop()
            else:
                b.pop()
        self.seen.append(a + b)

        class Inputs(dict):
            def to(self, _device):
                return self

        return Inputs()


class LongContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_long_context_never_truncates_away_the_text(self):
        from app.schemas import ClassifyRequest

        manager = FakeManager([0.9, 0.1])
        manager.release.set()
        manager.tokenizer = _TruncatingTokenizer(max_length=16)
        clf = Classifier(manager)
        clf.labels, clf.max_length = ["spam", "ham"], 16

        context = ["old turn " * 20] * 5  # far past max_length on its own
        await clf.classify(ClassifyRequest(text="current message", context=context))

        self.assertTrue(manager.tokenizer.seen)
        self.assertEqual(manager.tokenizer.seen[-1][-2:], ["current", "message"])


if __name__ == "__main__":
    unittest.main()
