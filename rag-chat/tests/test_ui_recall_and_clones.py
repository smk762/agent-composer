"""Guard history recall and the voice-clone list (packs c5e73475b7, 31812f7dd1, c2b7095c57).

No browser harness: these assert on the served page's inline JS."""

import re
import unittest
from unittest import mock

from app.routers import ui


def _page(name: str) -> str:
    with mock.patch.object(ui, "WHISPER_URL", "http://whisper"), mock.patch.object(ui, "TTS_URL", "http://tts"):
        return getattr(ui, name)().body.decode("utf-8")


def _between(src: str, start: str, end: str) -> str:
    i = src.index(start)
    return src[i:src.index(end, i)]


class GuardHistoryRecallTests(unittest.TestCase):
    def setUp(self) -> None:
        self.js = _page("guard_ui")
        self.recall = _between(self.js, 'historyEl.querySelectorAll(".history-item")', 'getElementById("clearHistBtn")')

    def test_model_override_is_saved_and_restored(self) -> None:
        self.assertIn('req_model: req.model || ""', _between(self.js, "function addHistory", "function renderHistory"))
        self.assertIn('modelEl.value = h.req_model || ""', self.recall)

    def test_recall_drops_an_attached_image_and_resets_status(self) -> None:
        self.assertIn('imageEl.value = ""', self.recall)
        self.assertIn('previewEl.style.display = "none"', self.recall)
        self.assertRegex(self.recall, r'statusEl\.textContent = h\.had_image \? .+ : ""')


class VoiceCloneListTests(unittest.TestCase):
    def setUp(self) -> None:
        js = _page("voice_clone_ui")
        self.load = _between(js, "async function loadClones", "// ── Import a voice archive")

    def test_stale_responses_are_dropped(self) -> None:
        self.assertIn("const seq = ++clonesSeq", self.load)
        self.assertGreaterEqual(self.load.count("if (seq !== clonesSeq) return"), 2)  # success and error paths

    def test_rows_act_on_the_engine_their_list_was_loaded_for(self) -> None:
        self.assertIn("const engine = currentEngine()", self.load)
        self.assertEqual(re.findall(r"engineQuery\(\)", self.load), [])  # no click-time engine lookup
        self.assertNotIn("currentEngine()) ", self.load.split("const engine = currentEngine()", 1)[1])


if __name__ == "__main__":
    unittest.main()
