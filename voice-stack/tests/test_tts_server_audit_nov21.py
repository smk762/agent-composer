"""tts_server regressions: clone-cache LRU refresh, failed GPU<->CPU moves,
non-OOM RuntimeErrors and empty generation results.

torch is stubbed as in test_tts_server.py. Run from voice-stack/:
python -m pytest tests/test_tts_server_audit_nov21.py -v
"""
import asyncio
import os
import sys
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

if "torch" not in sys.modules:
    _torch = types.ModuleType("torch")
    _torch.device = lambda d: d
    _torch.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
    sys.modules["torch"] = _torch

import tts_server  # noqa: E402
from tts_server import TTSModelManager, TTSState  # noqa: E402


class CloneCacheLruTests(unittest.TestCase):
    """9d137fca58: re-cloning a cached key makes it the most recently used."""

    def test_recloned_key_is_not_evicted_next(self):
        cache = tts_server._clone_cache
        saved = dict(cache)
        cache.clear()
        try:
            with mock.patch.object(tts_server, "_CLONE_CACHE_MAX", 2):
                tts_server._cache_clone_prompt("a", 1)
                tts_server._cache_clone_prompt("b", 2)
                tts_server._cache_clone_prompt("a", 3)  # re-clone a
                tts_server._cache_clone_prompt("c", 4)  # evicts the LRU entry
            self.assertEqual(list(cache), ["a", "c"])
            self.assertEqual(cache["a"], 3)
        finally:
            cache.clear()
            cache.update(saved)


class _BrokenInner:
    def __init__(self):
        self.device = "cuda:0"

    def to(self, device):
        raise RuntimeError("CUDA out of memory (partial move)")


class _Wrapper:
    def __init__(self, inner):
        self.model = inner
        self.device = inner.device


def _manager(gpu: float = 0, cpu: float = -1) -> TTSModelManager:
    mgr = TTSModelManager()
    mgr.keep_alive_gpu = gpu
    mgr.keep_alive_cpu = cpu
    mgr._base_model = _Wrapper(_BrokenInner())
    mgr._custom_model = None
    mgr._state = TTSState.GPU
    return mgr


class FailedMoveTests(unittest.IsolatedAsyncioTestCase):
    """2ad6fdab51: a failed move frees the models and lands in ERROR."""

    async def test_failed_demote_does_not_keep_gpu_state(self):
        mgr = _manager()
        with mock.patch.object(tts_server, "DEVICE", "cuda:0"):
            with self.assertRaises(RuntimeError):
                await mgr.demote_to_cpu()
        self.assertEqual(mgr.state, TTSState.ERROR)
        self.assertIsNone(mgr.base_model)

    async def test_eviction_loop_survives_a_failed_demote(self):
        mgr = _manager(gpu=0)
        with mock.patch.object(tts_server, "DEVICE", "cuda:0"):
            mgr._arm_evict()
            task = mgr._evict_task
            await asyncio.wait_for(task, 2)
        self.assertIsNone(task.exception())
        self.assertEqual(mgr.state, TTSState.ERROR)
        self.assertIsNone(mgr.base_model)

    async def test_failed_promotion_does_not_stay_cpu(self):
        mgr = _manager()
        mgr._state = TTSState.CPU
        with mock.patch.object(tts_server, "DEVICE", "cuda:0"):
            with self.assertRaises(RuntimeError):
                await mgr.ensure_gpu()
        self.assertEqual(mgr.state, TTSState.ERROR)
        self.assertIsNone(mgr.base_model)


class _FakeCustom:
    def __init__(self, exc=None, result=None):
        self.exc = exc
        self.result = result

    def get_supported_speakers(self):
        return ["ryan"]

    def generate_custom_voice(self, **kw):
        if self.exc is not None:
            raise self.exc
        return self.result


def _run(custom):
    mgr = TTSModelManager()
    mgr._custom_model = custom
    with mock.patch.object(tts_server, "_mgr", mgr), \
            mock.patch.object(tts_server, "_gpu_memory_info", return_value={}):
        return asyncio.run(tts_server._generate_audio_leased(
            tts_server.SynthesiseRequest(text="hi", speaker="ryan")))


class GenerationErrorTests(unittest.TestCase):
    def test_non_oom_runtime_error_is_a_structured_500(self):
        # 1f2c4e4d28: no bare re-raise; a CUDA assert is not reported as OOM.
        for msg in ("tensor shape mismatch", "CUDA error: device-side assert triggered"):
            r = _run(_FakeCustom(exc=RuntimeError(msg)))
            self.assertEqual(r.status_code, 500, msg)
            self.assertIn(b"error_id", r.body)
            self.assertNotIn(b"OOM", r.body)

    def test_real_oom_is_still_503(self):
        r = _run(_FakeCustom(exc=RuntimeError("CUDA out of memory. Tried to allocate")))
        self.assertEqual(r.status_code, 503)

    def test_empty_generation_is_a_structured_error(self):
        # ffa4a71968: no IndexError on an empty list of audio arrays.
        r = _run(_FakeCustom(result=([], 24000)))
        self.assertEqual(r.status_code, 422)
        self.assertIn(b"no audio", r.body)


if __name__ == "__main__":
    unittest.main()
