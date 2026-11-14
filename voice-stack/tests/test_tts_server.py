"""TTSModelManager lease / keep-alive and /clone-voice validation tests.

No torch, qwen_tts or S3 needed: torch is stubbed and S3 helpers are patched.
Needs fastapi, python-multipart, httpx and structlog.
Run from voice-stack/: python -m pytest tests/test_tts_server.py -v
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


class FakeInner:
    def __init__(self):
        self.device = "cuda:0"
        self.moves = []

    def to(self, device):
        self.moves.append(device)
        self.device = device
        return self


class FakeWrapper:
    def __init__(self):
        self.model = FakeInner()
        self.device = "cuda:0"


def _manager(gpu: float, cpu: float) -> TTSModelManager:
    mgr = TTSModelManager()
    mgr.keep_alive_gpu = gpu
    mgr.keep_alive_cpu = cpu
    mgr._base_model = FakeWrapper()
    mgr._custom_model = FakeWrapper()
    mgr._state = TTSState.GPU
    return mgr


async def _spin(seconds: float = 0.05) -> None:
    await asyncio.sleep(seconds)


@mock.patch.object(tts_server, "DEVICE", "cuda:0")
class LeaseTests(unittest.IsolatedAsyncioTestCase):
    async def test_immediate_keep_alive_does_not_demote_mid_generation(self):
        mgr = _manager(gpu=0, cpu=0)
        base = mgr.base_model
        async with mgr.use():
            await _spin()
            self.assertEqual(mgr.state, TTSState.GPU)
            self.assertIs(mgr.base_model, base)
            self.assertEqual(base.model.moves, [])
        await _spin()
        self.assertEqual(mgr.state, TTSState.UNLOADED)
        self.assertEqual([str(d) for d in base.model.moves], ["cpu"])

    async def test_unload_and_demote_wait_for_generation(self):
        mgr = _manager(gpu=-1, cpu=-1)
        async with mgr.use():
            demote = asyncio.create_task(mgr.demote_to_cpu())
            await _spin()
            self.assertFalse(demote.done())
            self.assertEqual(mgr.state, TTSState.GPU)
        await asyncio.wait_for(demote, 1)
        self.assertEqual(mgr.state, TTSState.CPU)

        mgr._state = TTSState.GPU
        async with mgr.use():
            unload = asyncio.create_task(mgr.unload())
            await _spin()
            self.assertFalse(unload.done())
            self.assertIsNotNone(mgr.custom_model)
        await asyncio.wait_for(unload, 1)
        self.assertEqual(mgr.state, TTSState.UNLOADED)

    async def test_last_used_updated_on_release_and_timer_starts_then(self):
        mgr = _manager(gpu=0.05, cpu=-1)
        async with mgr.use():
            await _spin(0.1)
            started = mgr._last_used
        self.assertGreater(mgr._last_used, started)
        await _spin(0.02)
        self.assertEqual(mgr.state, TTSState.GPU)
        await _spin(0.08)
        self.assertEqual(mgr.state, TTSState.CPU)


@mock.patch.object(tts_server, "DEVICE", "cuda:0")
class KeepAliveTests(unittest.IsolatedAsyncioTestCase):
    async def test_gpu_forever_never_unloads(self):
        for cpu in (0, 0.01):
            mgr = _manager(gpu=-1, cpu=cpu)
            await mgr.ensure_gpu()
            await _spin()
            self.assertEqual(mgr.state, TTSState.GPU, f"cpu={cpu}")
            self.assertIsNotNone(mgr.base_model)

    async def test_cpu_stage_after_demotion(self):
        mgr = _manager(gpu=0.03, cpu=0.05)
        async with mgr.use():
            pass
        await _spin(0.05)
        self.assertEqual(mgr.state, TTSState.CPU)
        await _spin(0.06)
        self.assertEqual(mgr.state, TTSState.UNLOADED)


class CloneVoiceTests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient

        self.client = TestClient(tts_server.app)
        self.files = {"audio": ("a.wav", b"RIFF0000", "audio/wav")}

    def _post(self, **data):
        return self.client.post("/clone-voice", files=self.files, data=data)

    def test_rejects_bad_companion_ids(self):
        with mock.patch.object(tts_server, "_s3_object_exists") as exists:
            for bad in ("../other", "a/b", "", "x" * 65, "a b", "a.b"):
                r = self._post(companion_id=bad)
                self.assertEqual(r.status_code, 422, bad)
            exists.assert_not_called()

    def test_refuses_to_overwrite_existing_clone(self):
        with mock.patch.object(tts_server, "_s3_object_exists", return_value=True) as exists, \
                mock.patch.object(tts_server, "_clone_voice_leased") as leased:
            r = self._post(companion_id="alice_1")
        self.assertEqual(r.status_code, 409)
        exists.assert_called_once_with("voice_clones/alice_1/prompt.pt")
        leased.assert_not_called()

    def test_new_or_replace_proceeds(self):
        async def fake_leased(audio, reference_text, companion_id, s3_key):
            return {"voice_clone_id": s3_key}

        tts_server._mgr._state = TTSState.GPU
        try:
            for exists_val, data in ((False, {}), (True, {"replace": "true"})):
                with mock.patch.object(tts_server, "_s3_object_exists", return_value=exists_val), \
                        mock.patch.object(tts_server, "_clone_voice_leased", side_effect=fake_leased):
                    r = self._post(companion_id="Bob-2", **data)
                self.assertEqual(r.status_code, 200, r.text)
                self.assertEqual(r.json()["voice_clone_id"], "voice_clones/Bob-2/prompt.pt")
        finally:
            tts_server._mgr._state = TTSState.UNLOADED
            tts_server._mgr._idle.set()


if __name__ == "__main__":
    unittest.main()
