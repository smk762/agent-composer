"""Page-source checks for UI fixes: pipeline CSS braces, guard history recall, deferred blob
revocation, voice-clone score/record races and the chat page's conversation-id check.

There is no browser harness, so these assert on the served page source.
"""

import re
import unittest
from unittest import mock

from app.routers import ui


def _page(name: str) -> str:
    with mock.patch.object(ui, "WHISPER_URL", "http://whisper"), mock.patch.object(ui, "TTS_URL", "http://tts"):
        return getattr(ui, name)().body.decode("utf-8")


class PipelineCssTests(unittest.TestCase):
    def test_no_doubled_braces_in_css(self):
        page = _page("pipeline_ui")
        css = page[page.index("<style>"):page.index("</style>")]
        self.assertNotIn("{{", css)
        self.assertNotIn("}}", css)
        self.assertIn("@keyframes pulse-border {", css)
        self.assertIn("@media (max-width: 900px) { .pl-columns {", css)


class GuardHistoryTests(unittest.TestCase):
    def setUp(self):
        self.page = _page("guard_ui")

    def test_history_keeps_full_input(self):
        self.assertNotIn("input.slice(0, 200)", self.page)
        self.assertIn("input: req.input,", self.page)

    def test_history_restores_request_settings(self):
        for field in ("input_role: req.input_role", "conversation: req.conversation",
                      "policy: req.policy", "req_thinking: req.thinking"):
            self.assertIn(field, self.page)
        self.assertIn("roleEl.value = h.input_role", self.page)
        self.assertIn("policyEl.value = h.policy", self.page)
        self.assertIn("thinkingEl.value = h.req_thinking", self.page)
        self.assertIn("addTurn(t.role, t.content)", self.page)


class DeferredRevokeTests(unittest.TestCase):
    def test_object_url_not_revoked_right_after_click(self):
        for name in ("voice_ui", "pipeline_ui"):
            page = _page(name)
            self.assertIsNone(re.search(r"a\.click\(\);\s*URL\.revokeObjectURL", page), name)
            self.assertIn("setTimeout(() => URL.revokeObjectURL(a.href), 1000)", page, name)


class VoiceCloneRaceTests(unittest.TestCase):
    def setUp(self):
        self.page = _page("voice_clone_ui")

    def test_scores_apply_to_the_clips_sent(self):
        self.assertIn("const sent = clips.slice();", self.page)
        self.assertIn("sent.forEach(c => fd.append(", self.page)
        self.assertNotIn("clips[i].", self.page)

    def test_record_start_is_guarded_before_awaiting_mic(self):
        start = self.page.index("async function startRecording()")
        guard = self.page.index("if (recStarting || recording) return;", start)
        mic = self.page.index("getUserMedia", start)
        self.assertLess(guard, mic)
        self.assertLess(self.page.index("recStarting = true;", start), mic)


class ChatConversationIdTests(unittest.TestCase):
    def test_cid_validated_before_fetch(self):
        page = _page("chat_ui")
        m = re.search(r"const CONV_ID_RE = /(.+?)/i;", page)
        self.assertIsNotNone(m)
        rx = re.compile(m.group(1), re.I)
        self.assertTrue(rx.match("0f8fad5b-d9cb-469f-a165-70867728950e"))
        for bad in ("..", "../admin", "..%2Fclones%2Fx%2Fdownload", "x/../y"):
            self.assertIsNone(rx.match(bad), bad)
        fn = page.index("async function loadConversation(convId)")
        self.assertLess(page.index("CONV_ID_RE.test(convId)", fn), page.index("fetch(", fn))


if __name__ == "__main__":
    unittest.main()
