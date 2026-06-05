import json
from textwrap import dedent

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from app.config import CHAT_MODEL, DEV_AUTH_BYPASS, WHISPER_URL, TTS_URL

router = APIRouter()

# Substrings matched against lowercased Ollama model names (order: most specific first).
CHAT_MODEL_GUIDE_RULES: list[dict[str, str]] = [
    {
        "match": "neuraldaredevil",
        "title": "Uncensored / alignment-stripped",
        "purpose": "Llama-family weights with safety tuning reduced; useful for comparing refusal vs base instruct, not a default for untrusted prompts.",
    },
    {
        "match": "mistral-nemo",
        "title": "Mistral Nemo (12B instruct)",
        "purpose": "Large-context general assistant; relatively light steering vs many peers. Good default for long docs and everyday chat.",
    },
    {
        "match": "olmo2",
        "title": "OLMo 2 (AI2)",
        "purpose": "Open training stack (data + weights); instruct-tuned. Strong choice when you care about reproducibility and research transparency.",
    },
    {
        "match": "llama3.1:8b-text",
        "title": "Llama 3.1 8B — base (text)",
        "purpose": "Pre–chat-tuned weights: raw continuation, minimal assistant persona. Best for probing model behavior, not polished UX.",
    },
    {
        "match": "llama3.1:8b-instruct",
        "title": "Llama 3.1 8B — instruct",
        "purpose": "RLHF-aligned assistant; stable refusals and helpful style. Pairs with the :8b-text model for base vs instruct A/B.",
    },
    {
        "match": "7b-text-v0.2",
        "title": "Mistral 7B — base (text v0.2)",
        "purpose": "Pure completion / low wrapper; direct answers, less policy-shaped phrasing. Good transparency baseline for prompts.",
    },
    {
        "match": "mistral:7b-text",
        "title": "Mistral 7B — base (text)",
        "purpose": "Base Mistral for continuation-style use; similar role to other :text tags (older v0.1 context may differ).",
    },
    {
        "match": "mistral",
        "title": "Mistral 7B family",
        "purpose": "General assistant or base/instruct variant depending on tag; usually coherent with lighter steering than some US lab instruct models.",
    },
    {
        "match": "qwen2.5-coder:32b",
        "title": "Qwen2.5 Coder 32B",
        "purpose": "Heavy coding model; needs more VRAM/RAM. Use for hard refactors, large patches, and repo-wide code reasoning.",
    },
    {
        "match": "qwen2.5-coder",
        "title": "Qwen2.5 Coder",
        "purpose": "Code-specialized instruct model: implementation, debugging, and API usage. Prefer over general Qwen for programming.",
    },
    {
        "match": "qwen2.5",
        "title": "Qwen2.5",
        "purpose": "Strong general instruct model (multilingual). Your :7b tag is instruct-tuned—not a raw base.",
    },
    {
        "match": "qwen2",
        "title": "Qwen2",
        "purpose": "Earlier Qwen2 line; good for multilingual and general chat. Match tag (:instruct, :coder, :text) for exact behavior.",
    },
    {
        "match": "llama3.2",
        "title": "Llama 3.2",
        "purpose": "Compact Llama 3.2; fast on small GPUs. Better for latency than nuance; 3B can feel noisy for subtle reasoning.",
    },
    {
        "match": "llama3.1",
        "title": "Llama 3.1",
        "purpose": "Llama 3.1 family; use :8b-instruct for assistant behavior or :8b-text for base completion.",
    },
    {
        "match": "deepseek-r1",
        "title": "DeepSeek R1 (distill)",
        "purpose": "Reasoning-oriented distill; often shows chain-of-thought style. Use when you want explicit deliberation over speed.",
    },
    {
        "match": "gemma2",
        "title": "Gemma 2",
        "purpose": "Google instruct line; typically stricter safety and concise answers. Contrast with Mistral/Open-weight baselines.",
    },
    {
        "match": "phi",
        "title": "Phi",
        "purpose": "Small, efficient Microsoft models; strong for size, can feel synthetic on open-ended tasks.",
    },
    {
        "match": "codellama",
        "title": "Code Llama",
        "purpose": "Meta code-focused model; legacy but still useful for completion and codegen-style prompts.",
    },
]

DEFAULT_CHAT_MODEL_GUIDE: dict[str, str] = {
    "title": "General chat model",
    "purpose": "Listed in your Ollama runtime; behavior depends on its tag (base, instruct, coder). Try it for your task or pick a specialized model from the cards above.",
}


BASE_CSS = """
:root {
  --bg: #0f172a;
  --card: #0b1221;
  --card-2: #10182b;
  --accent: #38bdf8;
  --accent-2: #6366f1;
  --text: #e5e7eb;
  --muted: #94a3b8;
  --border: #1f2937;
}
* { box-sizing: border-box; }
body {
  font-family: "Inter", system-ui, -apple-system, sans-serif;
  background: radial-gradient(circle at 20% 20%, rgba(56,189,248,0.08), transparent 25%),
              radial-gradient(circle at 80% 0%, rgba(99,102,241,0.08), transparent 30%),
              var(--bg);
  color: var(--text);
  min-height: 100vh;
  margin: 0;
  display: flex;
  align-items: flex-start;
  justify-content: center;
  padding: 0px;
}
.card {
  width: min(960px, 100%);
  background: linear-gradient(180deg, var(--card-2), var(--card));
  border: 1px solid var(--border);
  border-radius: 16px;
  box-shadow: 0 20px 60px rgba(0,0,0,0.35);
  padding: 20px;
  margin-top: 50px;
  height: 92vh;
}
h1 { margin: 0 0 8px; font-size: 24px; }
p.sub { margin: 0 0 20px; color: var(--muted); }
label { display: block; font-weight: 600; margin-bottom: 6px; }
input, button {
  font: inherit;
  border-radius: 10px;
  border: 1px solid var(--border);
  background: #0b1221;
  color: var(--text);
  padding: 10px 12px;
  box-sizing: border-box;
}
input { width: 100%; }
button {
  width: auto;
  padding: 10px 16px;
  height: 42px;
  border: none;
  background: linear-gradient(90deg, var(--accent), var(--accent-2));
  color: #0b1020;
  font-weight: 700;
  cursor: pointer;
  transition: transform 120ms ease, box-shadow 120ms ease, opacity 120ms ease;
  box-shadow: 0 8px 20px rgba(56,189,248,0.25);
}
button:hover { transform: translateY(-1px); opacity: 0.95; }
.row { margin-bottom: 16px; }
.note {
  color: var(--muted);
  font-size: 13px;
  margin: 3px;
}
.muted { color: var(--muted); font-size: 13px; }
table { width: 100%; border-collapse: collapse; margin-top: 12px; }
th, td { border: 1px solid var(--border); padding: 10px; text-align: center; font-size: 14px; }
th { background: #0d1322; }
.pill { padding: 3px 8px; border-radius: 999px; background: #1f2937; color: var(--muted); font-size: 12px; }
.danger { background: rgba(239,68,68,0.15); color: #f87171; border: 1px solid rgba(239,68,68,0.5); }
.banner { padding: 10px; border: 1px dashed var(--border); border-radius: 12px; margin-bottom: 14px; background: rgba(56,189,248,0.06); }
.chat-log {
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 12px;
  background: #0b1221;
  min-height: 360px;
  flex: 1;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.msg-row { display: flex; flex-direction: column; gap: 4px; }
.bubble {
  padding: 12px 14px;
  border-radius: 12px;
  white-space: pre-wrap;
  word-break: break-word;
}
.bubble.user {
  background: rgba(99,102,241,0.15);
  border: 1px solid rgba(99,102,241,0.4);
  align-self: flex-end;
}
.bubble.assistant {
  background: rgba(56,189,248,0.12);
  border: 1px solid rgba(56,189,248,0.35);
  align-self: flex-start;
}
.input-wrap { position: relative; }
.wait-overlay {
  position: absolute;
  inset: 0;
  display: none;
  align-items: center;
  justify-content: center;
  background: rgba(0,0,0,0.45);
  border-radius: 10px;
  gap: 8px;
  font-weight: 600;
}
.wait-overlay .spinner {
  width: 18px;
  height: 18px;
  border: 3px solid rgba(255,255,255,0.2);
  border-top-color: rgba(56,189,248,0.9);
  border-radius: 50%;
  animation: spin 0.9s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
"""


def nav_html() -> str:
    bypass = "DEV_AUTH_BYPASS active; using dev user." if DEV_AUTH_BYPASS else ""
    return (
        '<div style="position:fixed; top:0; z-index:10; width:100%; margin:0 auto 8px; '
        'background:rgba(15,23,42,0.85); backdrop-filter: blur(8px); border:1px solid var(--border); '
        'padding:10px 14px; display:flex; gap:12px; align-items:center; box-sizing:border-box;">'
        '<a href="/" style="color:var(--text); text-decoration:none; font-weight:700;">rag-chat</a>'
        '<a href="/ui/chat" style="color:var(--text); text-decoration:none;">Chat</a>'
        '<a href="/ui/history" style="color:var(--text); text-decoration:none;">History</a>'
        '<a href="/ui/generate" style="color:var(--text); text-decoration:none;">Generate</a>'
        '<a href="/ui/api-keys" style="color:var(--text); text-decoration:none;">API keys</a>'
        '<a href="/ui/pipeline" style="color:var(--accent); text-decoration:none; font-weight:600;">Pipeline Lab</a>'
        '<a href="/ui/guard" style="color:var(--text); text-decoration:none;">Guard</a>'
        + (
            '<a href="/ui/voice" style="color:var(--text); text-decoration:none;">Voice</a>'
            if (WHISPER_URL or TTS_URL) else ""
        )
        + (
            '<a href="/ui/voice-clone" style="color:var(--text); text-decoration:none;">Voice Clone</a>'
            if TTS_URL else ""
        )
        + f'<span style="margin-left:auto; color:var(--muted); font-size:13px;">{bypass}</span>'
        "</div>"
    )


def render_page(title: str, body_html: str, extra_css: str = "") -> HTMLResponse:
    return HTMLResponse(
        dedent(
            f"""
            <!doctype html>
            <html>
            <head>
              <meta charset="utf-8" />
              <title>{title}</title>
              <style>
                {BASE_CSS}
                {extra_css}
              </style>
            </head>
            <body>
              {nav_html()}
              {body_html}
            </body>
            </html>
            """
        )
    )


@router.get("/ui/api-keys", response_class=HTMLResponse)
def api_keys_ui():
    extra_css = """
    .inline-row { display:flex; gap:10px; align-items:center; }
    .inline-row input { flex:1; }
    .inline-status { color: var(--muted); font-size: 12px; margin-top: 4px; }
    """
    body = f"""
      <div class="card">
        <h1>API keys</h1>
        <p class="sub">Create, list, and revoke API keys. Saved API key is also used by the chat UI (Authorization: Bearer ...).</p>
        <div id="banner" class="banner" style="display:none;"></div>
        <div class="row">
          <label>Current API key (stored locally)</label>
          <div class="inline-row">
            <input id="savedKey" placeholder="paste key to use for requests" />
            <button id="saveKeyBtn" type="button">Save key</button>
          </div>
          <div id="saveStatus" class="inline-status"></div>
          <div class="muted">Stored in localStorage as 'rag_api_key'.</div>
        </div>
        <div class="row" style="display:flex; gap:10px; flex-wrap:wrap; align-items:flex-end;">
          <div style="flex:1; min-width:240px;">
            <label>New key name (optional)</label>
            <input id="keyName" placeholder="laptop, agent, etc." />
          </div>
          <button id="createBtn">Create key</button>
        </div>
        <div class="row" id="secretBox" style="display:none;">
          <label>New key (copy now, shown once)</label>
          <input id="secret" readonly />
          <div class="muted">Keep this secret safe. You won't see it again.</div>
        </div>
        <div class="row">
          <h3 style="margin:0 0 6px;">Your keys</h3>
          <div class="muted" id="status"></div>
          <table id="table" style="display:none;">
            <thead>
              <tr><th>Name</th><th>Prefix</th><th>Created</th><th>Last used</th><th>Status</th><th></th></tr>
            </thead>
            <tbody id="rows"></tbody>
          </table>
        </div>
      </div>
      <script>
        const DEV_BYPASS = {"true" if DEV_AUTH_BYPASS else "false"};
        const savedKeyEl = document.getElementById("savedKey");
        const saveKeyBtn = document.getElementById("saveKeyBtn");
        const saveStatus = document.getElementById("saveStatus");
        const keyNameEl = document.getElementById("keyName");
        const secretBox = document.getElementById("secretBox");
        const secretEl = document.getElementById("secret");
        const statusEl = document.getElementById("status");
        const tableEl = document.getElementById("table");
        const rowsEl = document.getElementById("rows");
        const bannerEl = document.getElementById("banner");
        const createBtn = document.getElementById("createBtn");

        function setBanner() {{
          if (DEV_BYPASS) {{
            bannerEl.style.display = "block";
            bannerEl.textContent = "DEV_AUTH_BYPASS is ON. Requests use X-Dev-User / DEV_DEFAULT_USER when no API key is provided.";
          }}
        }}

        function loadSavedKey() {{
          const k = localStorage.getItem("rag_api_key") || "";
          savedKeyEl.value = k;
        }}

        function saveKeyLocal() {{
          const v = (savedKeyEl.value || "").trim();
          if (v) localStorage.setItem("rag_api_key", v);
          else localStorage.removeItem("rag_api_key");
          saveStatus.textContent = v ? "Saved local API key." : "Cleared local API key.";
        }}

        function headers() {{
          const h = {{"Accept": "application/json"}};
          const key = (localStorage.getItem("rag_api_key") || "").trim();
          if (key) h["Authorization"] = "Bearer " + key;
          return h;
        }}

        async function loadKeys() {{
          statusEl.textContent = "Loading...";
          tableEl.style.display = "none";
          rowsEl.innerHTML = "";
          try {{
            const r = await fetch("/api-keys", {{ headers: headers() }});
            if (!r.ok) {{
              statusEl.textContent = "Failed to load keys (" + r.status + "). Add an API key or enable dev bypass.";
              return;
            }}
            const data = await r.json();
            if (!Array.isArray(data)) {{
              statusEl.textContent = "Unexpected response.";
              return;
            }}
            if (!data.length) {{
              statusEl.textContent = "No keys yet.";
              return;
            }}
            data.forEach(k => {{
              const tr = document.createElement("tr");
              tr.innerHTML = `
                <td>${{k.name || "\u2014"}}</td>
                <td><code>${{k.prefix}}</code></td>
                <td>${{k.created_at || "\u2014"}}</td>
                <td>${{k.last_used_at || "\u2014"}}</td>
                <td>${{k.revoked_at ? '<span class="pill danger">revoked</span>' : '<span class="pill">active</span>'}}</td>
                <td>${{k.revoked_at ? "" : '<button data-id="' + k.id + '" class="revoke">Revoke</button>'}}</td>
              `;
              rowsEl.appendChild(tr);
            }});
            tableEl.style.display = "table";
            statusEl.textContent = "";
          }} catch (e) {{
            statusEl.textContent = "Load failed.";
          }}
        }}

        async function createKey() {{
          const name = keyNameEl.value.trim();
          secretBox.style.display = "none";
          statusEl.textContent = "Creating...";
          try {{
            const r = await fetch("/api-keys", {{
              method: "POST",
              headers: Object.assign({{"Content-Type": "application/json"}}, headers()),
              body: JSON.stringify({{ name }})
            }});
            if (!r.ok) {{
              statusEl.textContent = "Create failed (" + r.status + ").";
              return;
            }}
            const data = await r.json();
            secretEl.value = data.secret || "";
            secretBox.style.display = "block";
            statusEl.textContent = "Key created. Copy the secret now.";
            await loadKeys();
          }} catch (e) {{
            statusEl.textContent = "Create failed.";
          }}
        }}

        async function revoke(id) {{
          if (!confirm("Revoke this key?")) return;
          statusEl.textContent = "Revoking...";
          try {{
            const r = await fetch("/api-keys/" + id, {{
              method: "DELETE",
              headers: headers()
            }});
            if (!r.ok) {{
              statusEl.textContent = "Revoke failed (" + r.status + ").";
              return;
            }}
            statusEl.textContent = "Revoked.";
            await loadKeys();
          }} catch (e) {{
            statusEl.textContent = "Revoke failed.";
          }}
        }}

        savedKeyEl.addEventListener("change", saveKeyLocal);
        savedKeyEl.addEventListener("blur", saveKeyLocal);
        saveKeyBtn.addEventListener("click", saveKeyLocal);
        createBtn.addEventListener("click", createKey);
        rowsEl.addEventListener("click", (e) => {{
          if (e.target && e.target.classList.contains("revoke")) {{
            revoke(e.target.dataset.id);
          }}
        }});

        loadSavedKey();
        setBanner();
        loadKeys();
      </script>
    """
    return render_page("API keys", body, extra_css)


@router.get("/ui/history", response_class=HTMLResponse)
def history_ui():
    body = f"""
      <div class="card">
        <h1>Chat history</h1>
        <p class="sub">Quickly find past conversations; open in chat to continue.</p>
        <div id="banner" class="banner"></div>
        <div class="actions">
          <button onclick="window.location.href='/ui/chat'">New chat</button>
        </div>
        <div class="muted" id="status">Loading...</div>
        <div id="grid" class="grid"></div>
      </div>
      <script>
        const DEV_BYPASS = {"true" if DEV_AUTH_BYPASS else "false"};
        const statusEl = document.getElementById("status");
        const gridEl = document.getElementById("grid");
        const bannerEl = document.getElementById("banner");

        function authHeaders() {{
          const h = {{"Accept": "application/json"}};
          const key = (localStorage.getItem("rag_api_key") || "").trim();
          if (key) h["Authorization"] = "Bearer " + key;
          return h;
        }}

        function setBanner() {{
          if (DEV_BYPASS) {{
            bannerEl.style.display = "block";
            bannerEl.textContent = "DEV_AUTH_BYPASS is ON. Requests use X-Dev-User / DEV_DEFAULT_USER when no API key is provided.";
          }}
        }}

        function render(convs) {{
          gridEl.innerHTML = "";
          if (!convs.length) {{
            statusEl.textContent = "No conversations yet.";
            return;
          }}
          statusEl.textContent = "";
          convs.forEach((c) => {{
            const card = document.createElement("div");
            card.className = "item";
            const title = c.title || "Untitled conversation";
            const summary = c.summary || "\u2014";
            const updated = c.updated_at || c.created_at || "\u2014";
            card.innerHTML = `
              <h3>${{title}}</h3>
              <div class="muted">${{summary}}</div>
              <div class="muted">Updated: ${{updated}}</div>
              <div class="actions">
                <button class="secondary" data-open="${{c.id}}">Open</button>
                <button data-del="${{c.id}}">Delete</button>
              </div>
            `;
            gridEl.appendChild(card);
          }});
        }}

        async function loadConversations() {{
          statusEl.textContent = "Loading...";
          try {{
            const r = await fetch("/conversations", {{ headers: authHeaders() }});
            if (!r.ok) {{
              statusEl.textContent = "Failed to load (" + r.status + "). Add an API key or enable dev bypass.";
              return;
            }}
            const data = await r.json();
            render(Array.isArray(data) ? data : []);
          }} catch (e) {{
            statusEl.textContent = "Load failed.";
          }}
        }}

        async function deleteConversation(id) {{
          if (!confirm("Delete this conversation?")) return;
          statusEl.textContent = "Deleting...";
          try {{
            const r = await fetch("/conversations/" + id, {{ method: "DELETE", headers: authHeaders() }});
            if (!r.ok) {{
              statusEl.textContent = "Delete failed (" + r.status + ").";
              return;
            }}
            statusEl.textContent = "Deleted.";
            await loadConversations();
          }} catch (e) {{
            statusEl.textContent = "Delete failed.";
          }}
        }}

        gridEl.addEventListener("click", (e) => {{
          const t = e.target;
          if (t.dataset && t.dataset.open) {{
            window.location.href = "/ui/chat?cid=" + encodeURIComponent(t.dataset.open);
          }}
          if (t.dataset && t.dataset.del) {{
            deleteConversation(t.dataset.del);
          }}
        }});

        setBanner();
        loadConversations();
      </script>
    """
    return render_page("Chat history", body)


@router.get("/ui/generate", response_class=HTMLResponse)
def generate_ui():
    extra_css = """
    .card { height: auto; min-height: 92vh; }
    .tabs { display:flex; gap:0; margin-bottom:16px; }
    .tab {
      padding:10px 20px; cursor:pointer; border:1px solid var(--border);
      background:var(--card); color:var(--muted); font-weight:600; font-size:14px;
      border-bottom:none; border-radius:10px 10px 0 0; margin-right:-1px;
    }
    .tab.active { background:var(--card-2); color:var(--accent); border-color:var(--accent); }
    .tab-panel { display:none; }
    .tab-panel.active { display:block; }
    .form-grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
    .form-grid.full { grid-template-columns:1fr; }
    select { width:100%; font:inherit; border-radius:10px; border:1px solid var(--border); background:#0b1221; color:var(--text); padding:10px 12px; }
    textarea { width:100%; min-height:80px; resize:vertical; font:inherit; border-radius:10px; border:1px solid var(--border); background:#0b1221; color:var(--text); padding:10px 12px; }
    .gallery { display:grid; grid-template-columns:repeat(auto-fill,minmax(240px,1fr)); gap:12px; margin-top:16px; }
    .gallery-item {
      border:1px solid var(--border); border-radius:12px; overflow:hidden;
      background:var(--card); cursor:pointer; transition:transform 120ms;
    }
    .gallery-item:hover { transform:scale(1.02); }
    .gallery-item img, .gallery-item video { width:100%; display:block; }
    .gallery-item .meta { padding:8px 10px; font-size:12px; color:var(--muted); }
    .gen-status { padding:12px; border:1px solid var(--border); border-radius:10px; margin-top:12px; background:rgba(56,189,248,0.06); }
    .upload-area {
      border:2px dashed var(--border); border-radius:12px; padding:20px; text-align:center;
      color:var(--muted); cursor:pointer; transition:border-color 120ms;
    }
    .upload-area:hover { border-color:var(--accent); }
    .upload-area input { display:none; }
    """
    body = f"""
      <div class="card">
        <h1>Generate</h1>
        <p class="sub">Create images and videos using configured AI providers.</p>
        <div id="banner" class="banner" style="display:none;"></div>

        <div class="tabs">
          <div class="tab active" data-tab="image">Image</div>
          <div class="tab" data-tab="edit">Edit / Inpaint</div>
          <div class="tab" data-tab="video">Video</div>
          <div class="tab" data-tab="history">History</div>
        </div>

        <!-- IMAGE TAB -->
        <div id="panel-image" class="tab-panel active">
          <div class="form-grid">
            <div class="row">
              <label>Provider</label>
              <select id="img-provider"><option value="">Auto</option></select>
            </div>
            <div class="row">
              <label>Model</label>
              <select id="img-model"><option value="">Default</option></select>
            </div>
          </div>
          <div class="row">
            <label>Prompt</label>
            <textarea id="img-prompt" placeholder="Describe the image you want to generate..."></textarea>
          </div>
          <div class="form-grid">
            <div class="row">
              <label>Negative prompt (optional)</label>
              <input id="img-neg" placeholder="Things to avoid..." />
            </div>
            <div class="row">
              <label>Size</label>
              <select id="img-size">
                <option value="1024x1024">1024x1024</option>
                <option value="1024x1792">1024x1792 (portrait)</option>
                <option value="1792x1024">1792x1024 (landscape)</option>
                <option value="512x512">512x512</option>
              </select>
            </div>
          </div>
          <div class="form-grid">
            <div class="row">
              <label>Count</label>
              <select id="img-n">
                <option value="1">1</option><option value="2">2</option>
                <option value="3">3</option><option value="4">4</option>
              </select>
            </div>
            <div class="row" style="display:flex;align-items:flex-end;">
              <button id="img-gen-btn">Generate Image</button>
            </div>
          </div>
          <div id="img-status" class="gen-status" style="display:none;"></div>
          <div id="img-gallery" class="gallery"></div>
        </div>

        <!-- EDIT TAB -->
        <div id="panel-edit" class="tab-panel">
          <div class="form-grid">
            <div class="row">
              <label>Provider</label>
              <select id="edit-provider"><option value="">Auto</option></select>
            </div>
            <div class="row">
              <label>Model</label>
              <select id="edit-model"><option value="">Default</option></select>
            </div>
          </div>
          <div class="row">
            <label>Edit prompt</label>
            <textarea id="edit-prompt" placeholder="Describe the edit you want..."></textarea>
          </div>
          <div class="form-grid">
            <div class="row">
              <label>Source image</label>
              <div class="upload-area" id="edit-img-area">
                <input type="file" id="edit-img-input" accept="image/*" />
                <div>Click or drag to upload source image</div>
              </div>
            </div>
            <div class="row">
              <label>Mask (optional, for inpainting)</label>
              <div class="upload-area" id="edit-mask-area">
                <input type="file" id="edit-mask-input" accept="image/*" />
                <div>Click or drag to upload mask</div>
              </div>
            </div>
          </div>
          <div class="row">
            <button id="edit-gen-btn">Edit Image</button>
          </div>
          <div id="edit-status" class="gen-status" style="display:none;"></div>
          <div id="edit-gallery" class="gallery"></div>
        </div>

        <!-- VIDEO TAB -->
        <div id="panel-video" class="tab-panel">
          <div class="form-grid">
            <div class="row">
              <label>Provider</label>
              <select id="vid-provider"><option value="">Auto</option></select>
            </div>
            <div class="row">
              <label>Model</label>
              <select id="vid-model"><option value="">Default</option></select>
            </div>
          </div>
          <div class="row">
            <label>Prompt</label>
            <textarea id="vid-prompt" placeholder="Describe the video you want to generate..."></textarea>
          </div>
          <div class="form-grid">
            <div class="row">
              <label>Duration (seconds, optional)</label>
              <input id="vid-duration" type="number" placeholder="e.g. 5" />
            </div>
            <div class="row">
              <label>Aspect ratio</label>
              <select id="vid-aspect">
                <option value="">Default</option>
                <option value="16:9">16:9</option>
                <option value="9:16">9:16</option>
                <option value="1:1">1:1</option>
              </select>
            </div>
          </div>
          <div class="row">
            <button id="vid-gen-btn">Generate Video</button>
          </div>
          <div id="vid-status" class="gen-status" style="display:none;"></div>
          <div id="vid-gallery" class="gallery"></div>
        </div>

        <!-- HISTORY TAB -->
        <div id="panel-history" class="tab-panel">
          <div class="muted" id="hist-status">Loading...</div>
          <div id="hist-gallery" class="gallery"></div>
        </div>
      </div>

      <script>
        function authHeaders() {{
          const h = {{ "Accept": "application/json" }};
          const key = (localStorage.getItem("rag_api_key") || "").trim();
          if (key) h["Authorization"] = "Bearer " + key;
          return h;
        }}

        // ---- Tabs ----
        document.querySelectorAll(".tab").forEach(tab => {{
          tab.addEventListener("click", () => {{
            document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
            document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
            tab.classList.add("active");
            document.getElementById("panel-" + tab.dataset.tab).classList.add("active");
            if (tab.dataset.tab === "history") loadHistory();
          }});
        }});

        // ---- Provider / model dropdowns ----
        async function loadProviders() {{
          try {{
            const r = await fetch("/api/providers", {{ headers: authHeaders() }});
            if (!r.ok) return;
            const providers = await r.json();
            populateProviders(providers);
          }} catch(e) {{
            console.warn("Failed to load providers:", e);
          }}
        }}

        function populateProviders(providers) {{
          const selectors = [
            {{ el: "img-provider", cap: "image_generate", modelEl: "img-model" }},
            {{ el: "edit-provider", cap: "image_edit", modelEl: "edit-model" }},
            {{ el: "vid-provider", cap: "video_generate", modelEl: "vid-model" }},
          ];
          selectors.forEach(s => {{
            const sel = document.getElementById(s.el);
            const modelSel = document.getElementById(s.modelEl);
            const relevant = providers.filter(p => p.capabilities.includes(s.cap) && p.configured && p.enabled);
            relevant.forEach(p => {{
              const opt = document.createElement("option");
              opt.value = p.name;
              opt.textContent = p.name;
              sel.appendChild(opt);
            }});
            sel.addEventListener("change", () => {{
              modelSel.innerHTML = '<option value="">Default</option>';
              const chosen = providers.find(p => p.name === sel.value);
              if (chosen && chosen.models && chosen.models[s.cap]) {{
                chosen.models[s.cap].forEach(m => {{
                  const opt = document.createElement("option");
                  opt.value = m;
                  opt.textContent = m;
                  modelSel.appendChild(opt);
                }});
              }}
            }});
          }});
        }}

        // ---- Upload areas ----
        function setupUpload(areaId, inputId) {{
          const area = document.getElementById(areaId);
          const input = document.getElementById(inputId);
          area.addEventListener("click", () => input.click());
          area.addEventListener("dragover", e => {{ e.preventDefault(); area.style.borderColor = "var(--accent)"; }});
          area.addEventListener("dragleave", () => {{ area.style.borderColor = ""; }});
          area.addEventListener("drop", e => {{
            e.preventDefault();
            area.style.borderColor = "";
            if (e.dataTransfer.files.length) {{
              input.files = e.dataTransfer.files;
              area.querySelector("div").textContent = e.dataTransfer.files[0].name;
            }}
          }});
          input.addEventListener("change", () => {{
            if (input.files.length) area.querySelector("div").textContent = input.files[0].name;
          }});
        }}
        setupUpload("edit-img-area", "edit-img-input");
        setupUpload("edit-mask-area", "edit-mask-input");

        // ---- Rendering ----
        function renderMedia(galleryId, media) {{
          const gal = document.getElementById(galleryId);
          media.forEach(m => {{
            const item = document.createElement("div");
            item.className = "gallery-item";
            if (m.media_type === "video") {{
              item.innerHTML = `<video src="${{m.url}}" controls></video><div class="meta">${{m.filename}}</div>`;
            }} else {{
              item.innerHTML = `<img src="${{m.url}}" alt="${{m.filename}}" /><div class="meta">${{m.filename}}</div>`;
            }}
            item.addEventListener("click", () => window.open(m.url, "_blank"));
            gal.appendChild(item);
          }});
        }}

        function setStatus(id, msg, isError) {{
          const el = document.getElementById(id);
          el.style.display = "block";
          el.textContent = msg;
          el.style.color = isError ? "#f87171" : "var(--text)";
        }}

        // ---- Image generation ----
        document.getElementById("img-gen-btn").addEventListener("click", async () => {{
          const prompt = document.getElementById("img-prompt").value.trim();
          if (!prompt) {{ setStatus("img-status", "Please enter a prompt.", true); return; }}
          setStatus("img-status", "Generating...", false);
          document.getElementById("img-gallery").innerHTML = "";
          try {{
            const body = {{
              prompt,
              negative_prompt: document.getElementById("img-neg").value.trim() || null,
              provider: document.getElementById("img-provider").value || null,
              model: document.getElementById("img-model").value || null,
              size: document.getElementById("img-size").value,
              n: parseInt(document.getElementById("img-n").value),
            }};
            const r = await fetch("/api/generate/image", {{
              method: "POST",
              headers: Object.assign({{ "Content-Type": "application/json" }}, authHeaders()),
              body: JSON.stringify(body),
            }});
            if (!r.ok) {{
              const err = await r.text();
              setStatus("img-status", "Failed: " + err.substring(0, 300), true);
              return;
            }}
            const data = await r.json();
            setStatus("img-status", `Generated ${{data.media.length}} image(s) via ${{data.provider}} / ${{data.model}}`, false);
            renderMedia("img-gallery", data.media);
          }} catch(e) {{
            setStatus("img-status", "Error: " + e.message, true);
          }}
        }});

        // ---- Image editing ----
        document.getElementById("edit-gen-btn").addEventListener("click", async () => {{
          const prompt = document.getElementById("edit-prompt").value.trim();
          const imgFile = document.getElementById("edit-img-input").files[0];
          if (!prompt) {{ setStatus("edit-status", "Please enter a prompt.", true); return; }}
          if (!imgFile) {{ setStatus("edit-status", "Please upload a source image.", true); return; }}
          setStatus("edit-status", "Editing...", false);
          document.getElementById("edit-gallery").innerHTML = "";
          try {{
            const fd = new FormData();
            fd.append("prompt", prompt);
            fd.append("image", imgFile);
            const maskFile = document.getElementById("edit-mask-input").files[0];
            if (maskFile) fd.append("mask", maskFile);
            const prov = document.getElementById("edit-provider").value;
            if (prov) fd.append("provider", prov);
            const model = document.getElementById("edit-model").value;
            if (model) fd.append("model", model);

            const h = authHeaders();
            delete h["Content-Type"];
            const r = await fetch("/api/edit/image", {{ method: "POST", headers: h, body: fd }});
            if (!r.ok) {{
              const err = await r.text();
              setStatus("edit-status", "Failed: " + err.substring(0, 300), true);
              return;
            }}
            const data = await r.json();
            setStatus("edit-status", `Edited via ${{data.provider}} / ${{data.model}}`, false);
            renderMedia("edit-gallery", data.media);
          }} catch(e) {{
            setStatus("edit-status", "Error: " + e.message, true);
          }}
        }});

        // ---- Video generation ----
        document.getElementById("vid-gen-btn").addEventListener("click", async () => {{
          const prompt = document.getElementById("vid-prompt").value.trim();
          if (!prompt) {{ setStatus("vid-status", "Please enter a prompt.", true); return; }}
          setStatus("vid-status", "Generating video (this may take several minutes)...", false);
          document.getElementById("vid-gallery").innerHTML = "";
          try {{
            const fd = new FormData();
            fd.append("prompt", prompt);
            const prov = document.getElementById("vid-provider").value;
            if (prov) fd.append("provider", prov);
            const model = document.getElementById("vid-model").value;
            if (model) fd.append("model", model);
            const dur = document.getElementById("vid-duration").value;
            if (dur) fd.append("duration", dur);
            const ar = document.getElementById("vid-aspect").value;
            if (ar) fd.append("aspect_ratio", ar);

            const h = authHeaders();
            delete h["Content-Type"];
            const r = await fetch("/api/generate/video", {{ method: "POST", headers: h, body: fd }});
            if (!r.ok) {{
              const err = await r.text();
              setStatus("vid-status", "Failed: " + err.substring(0, 300), true);
              return;
            }}
            const data = await r.json();
            setStatus("vid-status", `Generated via ${{data.provider}} / ${{data.model}}`, false);
            renderMedia("vid-gallery", data.media);
          }} catch(e) {{
            setStatus("vid-status", "Error: " + e.message, true);
          }}
        }});

        // ---- History ----
        async function loadHistory() {{
          const statusEl = document.getElementById("hist-status");
          const gallery = document.getElementById("hist-gallery");
          statusEl.textContent = "Loading...";
          gallery.innerHTML = "";
          try {{
            const r = await fetch("/api/generations", {{ headers: authHeaders() }});
            if (!r.ok) {{
              statusEl.textContent = "Failed to load (" + r.status + ").";
              return;
            }}
            const gens = await r.json();
            if (!gens.length) {{
              statusEl.textContent = "No generations yet.";
              return;
            }}
            statusEl.textContent = gens.length + " generation(s)";
            gens.forEach(g => {{
              if (g.media && g.media.length) {{
                renderMedia("hist-gallery", g.media);
              }}
            }});
          }} catch(e) {{
            statusEl.textContent = "Load failed.";
          }}
        }}

        loadProviders();
      </script>
    """
    return render_page("Generate", body, extra_css)


@router.get("/", response_class=HTMLResponse)
@router.get("/ui", response_class=HTMLResponse)
def index_ui():
    voice_enabled = bool(WHISPER_URL or TTS_URL)
    tts_enabled = bool(TTS_URL)

    pages = [
        ("Chat", "/ui/chat", "Stream local LLMs with optional RAG context.", True),
        ("History", "/ui/history", "Browse and revisit past conversations.", True),
        ("Generate", "/ui/generate", "Text-to-image generation.", True),
        ("API keys", "/ui/api-keys", "Issue and manage API keys.", True),
        ("Pipeline Lab", "/ui/pipeline", "ModernBERT classify → overlay → brain → prose.", True),
        ("Guard", "/ui/guard", "LlamaGuard content-safety tester.", True),
        ("Voice", "/ui/voice", "Speech-to-text and text-to-speech.", voice_enabled),
        ("Voice Clone", "/ui/voice-clone", "Clone a voice from a few reference clips.", tts_enabled),
    ]

    page_cards = "".join(
        f'<a class="idx-card" href="{href}"><div class="idx-name">{name}</div>'
        f'<div class="idx-desc">{desc}</div></a>'
        for name, href, desc, enabled in pages if enabled
    )

    # Other service UIs/dashboards. Built client-side from the current hostname
    # so the links work over LAN as well as localhost. Ports are compose defaults.
    services = [
        {"name": "Qdrant", "port": 6333, "path": "/dashboard", "desc": "Vector store dashboard."},
        {"name": "MinIO Console", "port": 9001, "path": "/", "desc": "Object storage (audio, uploads)."},
        {"name": "rag-ingest API", "port": 9050, "path": "/docs", "desc": "Signed ingestion API docs."},
        {"name": "Infinity", "port": 7997, "path": "/docs", "desc": "Code embedder + reranker."},
        {"name": "ModernBERT", "port": 7998, "path": "/docs", "desc": "Classifier sidecar."},
        {"name": "Whisper STT", "port": 8032, "path": "/docs", "desc": "faster-whisper sidecar."},
        {"name": "XTTS", "port": 8033, "path": "/docs", "desc": "XTTS-v2 TTS / cloning sidecar."},
    ]
    services_json = json.dumps(services)

    extra_css = """
    .idx-wrap { display: grid; gap: 22px; }
    .idx-section h2 { font-size: 14px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); margin: 0 0 12px; }
    .idx-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 12px; }
    .idx-card { display: block; text-decoration: none; background: var(--card); border: 1px solid var(--border);
                border-radius: 12px; padding: 14px 16px; transition: transform 120ms ease, border-color 120ms ease; }
    .idx-card:hover { transform: translateY(-2px); border-color: var(--accent); }
    .idx-name { font-weight: 700; color: var(--text); font-size: 15px; margin-bottom: 4px; }
    .idx-name .ext { color: var(--muted); font-weight: 400; font-size: 12px; }
    .idx-desc { color: var(--muted); font-size: 13px; line-height: 1.4; }
    """

    body = f"""
    <div class="card" style="max-width:1000px; height:auto; min-height:0;">
      <h1>agent-composer</h1>
      <p class="sub">Local RAG + chat stack. Jump to an app page or a service dashboard.</p>
      <div class="idx-wrap">
        <div class="idx-section">
          <h2>App pages</h2>
          <div class="idx-grid">{page_cards}</div>
        </div>
        <div class="idx-section">
          <h2>Services &amp; dashboards</h2>
          <div class="idx-grid" id="svcGrid"></div>
        </div>
      </div>
    </div>
    <script>
    (function() {{
      const services = {services_json};
      const host = window.location.hostname || "localhost";
      const grid = document.getElementById("svcGrid");
      services.forEach(s => {{
        const a = document.createElement("a");
        a.className = "idx-card";
        a.href = window.location.protocol + "//" + host + ":" + s.port + s.path;
        a.target = "_blank";
        a.rel = "noopener";
        a.innerHTML = '<div class="idx-name">' + s.name + ' <span class="ext">:' + s.port + ' &#8599;</span></div>'
          + '<div class="idx-desc">' + s.desc + '</div>';
        grid.appendChild(a);
      }});
    }})();
    </script>
    """

    return render_page("agent-composer", body, extra_css)


@router.get("/ui/chat", response_class=HTMLResponse)
def chat_ui():
    guide_rules_json = json.dumps(CHAT_MODEL_GUIDE_RULES)
    default_guide_json = json.dumps(DEFAULT_CHAT_MODEL_GUIDE)
    return HTMLResponse(
        dedent(
            f"""
            <!doctype html>
            <html>
            <head>
              <meta charset="utf-8" />
              <title>rag-chat UI</title>
              <style>
                :root {{
                  --bg: #0f172a;
                  --card: #0b1221;
                  --card-2: #10182b;
                  --accent: #38bdf8;
                  --accent-2: #6366f1;
                  --text: #e5e7eb;
                  --muted: #94a3b8;
                  --border: #1f2937;
                }}
                body {{
                  font-family: "Inter", system-ui, -apple-system, sans-serif;
                  background: radial-gradient(circle at 20% 20%, rgba(56,189,248,0.08), transparent 25%),
                              radial-gradient(circle at 80% 0%, rgba(99,102,241,0.08), transparent 30%),
                              var(--bg);
                  color: var(--text);
                  min-height: 100vh;
                  margin: 0;
                  display: flex;
                  align-items: flex-start;
                  justify-content: center;
                  padding: 0px;
                }}
                .card {{
                  width: min(960px, 100%);
                  background: linear-gradient(180deg, var(--card-2), var(--card));
                  border: 1px solid var(--border);
                  border-radius: 16px;
                  box-shadow: 0 20px 60px rgba(0,0,0,0.35);
                  padding: 20px;
                  margin-top: 50px;
                  margin-bottom: 48px;
                  min-height: 92vh;
                }}
                h1 {{
                  margin-top: 0;
                  margin-bottom: 8px;
                  font-size: 24px;
                  letter-spacing: 0.2px;
                }}
                p.sub {{
                  margin-top: 0;
                  margin-bottom: 20px;
                  color: var(--muted);
                }}
                label {{
                  display: block;
                  font-weight: 600;
                  margin-bottom: 6px;
                }}
                textarea, select, input, button {{
                  width: 100%;
                  font: inherit;
                  border-radius: 10px;
                  border: 1px solid var(--border);
                  background: #0b1221;
                  color: var(--text);
                  padding: 10px 12px;
                  box-sizing: border-box;
                }}
                textarea {{
                  min-height: 100px;
                  resize: vertical;
                }}
                select {{
                  background: #0b1221;
                }}
                button {{
                  width: auto;
                  padding: 10px 16px;
                  border: none;
                  background: linear-gradient(90deg, var(--accent), var(--accent-2));
                  color: #0b1020;
                  font-weight: 700;
                  cursor: pointer;
                  transition: transform 120ms ease, box-shadow 120ms ease, opacity 120ms ease;
                  box-shadow: 0 8px 20px rgba(56,189,248,0.25);
                }}
                button:hover {{
                  transform: translateY(-1px);
                  opacity: 0.95;
                }}
                .row {{ margin-bottom: 16px; }}
                .note {{
                  color: var(--muted);
                  font-size: 13px;
                  margin: 3px;
                }}
                .input-wrap {{ position: relative; }}
                .wait-overlay {{
                  position: absolute;
                  inset: 0;
                  display: none;
                  align-items: center;
                  justify-content: center;
                  background: rgba(0,0,0,0.45);
                  border-radius: 10px;
                  gap: 8px;
                  font-weight: 600;
                }}
                .wait-overlay .spinner {{
                  width: 18px;
                  height: 18px;
                  border: 3px solid rgba(255,255,255,0.2);
                  border-top-color: rgba(56,189,248,0.9);
                  border-radius: 50%;
                  animation: spin 0.9s linear infinite;
                }}
                @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
                .chat-log {{
                  border: 1px solid var(--border);
                  border-radius: 12px;
                  padding: 12px;
                  background: #0b1221;
                  min-height: 360px;
                  flex: 1;
                  overflow-y: auto;
                  display: flex;
                  flex-direction: column;
                  gap: 10px;
                }}
                .msg-row {{
                  display: flex;
                  flex-direction: column;
                  gap: 4px;
                }}
                .bubble {{
                  padding: 12px 14px;
                  border-radius: 12px;
                  white-space: pre-wrap;
                  word-break: break-word;
                }}
                .bubble.user {{
                  background: rgba(99,102,241,0.15);
                  border: 1px solid rgba(99,102,241,0.4);
                  align-self: flex-end;
                }}
                .bubble.assistant {{
                  background: rgba(56,189,248,0.12);
                  border: 1px solid rgba(56,189,248,0.35);
                  align-self: flex-start;
                }}
                .model-guide {{
                  margin-top: 24px;
                  padding-top: 20px;
                  border-top: 1px solid var(--border);
                }}
                .model-guide h2 {{
                  margin: 0 0 12px;
                  font-size: 18px;
                  font-weight: 600;
                  color: var(--text);
                }}
                .model-guide-grid {{
                  display: grid;
                  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
                  gap: 12px;
                }}
                .model-guide-card {{
                  background: rgba(15, 23, 42, 0.55);
                  border: 1px solid var(--border);
                  border-radius: 12px;
                  padding: 14px 16px;
                  display: flex;
                  flex-direction: column;
                  gap: 8px;
                }}
                .model-guide-card .model-id {{
                  font-family: ui-monospace, "Cascadia Code", monospace;
                  font-size: 12px;
                  color: var(--accent);
                  word-break: break-all;
                  line-height: 1.35;
                }}
                .model-guide-card h3 {{
                  margin: 0;
                  font-size: 15px;
                  font-weight: 600;
                  color: var(--text);
                }}
                .model-guide-card p {{
                  margin: 0;
                  font-size: 13px;
                  line-height: 1.45;
                  color: var(--muted);
                }}
              </style>
            </head>
            <body>
              <div style="position:fixed; top:0; z-index:10; width:100%; margin:0 auto 8px; background:rgba(15,23,42,0.85); backdrop-filter: blur(8px); border:1px solid var(--border); padding:10px 14px; display:flex; gap:12px; align-items:center; box-sizing:border-box;">
                <strong style="color:var(--text);">rag-chat</strong>
                <a href="/ui/chat" style="color:var(--text); text-decoration:none;">Chat</a>
                <a href="/ui/history" style="color:var(--text); text-decoration:none;">History</a>
                <a href="/ui/generate" style="color:var(--text); text-decoration:none;">Generate</a>
                <a href="/ui/api-keys" style="color:var(--text); text-decoration:none;">API keys</a>
                <span style="margin-left:auto; color:var(--muted); font-size:13px;">{"DEV_AUTH_BYPASS active; using dev user." if DEV_AUTH_BYPASS else ""}</span>
              </div>
              <div class="card">
                <h1>rag-chat</h1>
                <p class="sub">Lightweight UI for testing the chat gateway. RAG context is applied automatically when enabled.</p>
                <div id="log" class="chat-log"></div>
                <div class="row input-wrap" style="margin-top:auto; margin-bottom: 0px;">
                  <textarea id="msg" placeholder="Ask anything... (Enter to send, Shift+Enter for newline)"></textarea>
                  <div id="wait" class="wait-overlay">
                    <div class="spinner"></div>
                    <span id="wait-text">Waiting for reply\u2026</span>
                  </div>
                </div>
                <div class="row input-wrap" style="margin: 0; align-items: center;">
                  <div class="note" id="status"></div>
                </div>
                <div class="row" style="display:flex; gap:10px; align-items:center; flex-wrap:wrap;">
                  <div style="display:flex; align-items:center; gap:8px;">
                    <span class="note" style="min-width:42px;">Model</span>
                    <select id="model" style="display:none;"></select>
                    <div id="model-note" class="note"></div>
                  </div>
                  <div style="flex: 1;"></div>
                  <div style="display:flex; gap:10px; align-items:center;">
                    <button id="reset" type="button" style="background:#1f2937; color:var(--text); box-shadow:none;">Clear chat</button>
                    <button id="send">Send</button>
                  </div>
                </div>
                <section id="model-guide" class="model-guide" aria-label="Model guide"></section>
              </div>
              <script>
                const MODEL_GUIDE_RULES = {guide_rules_json};
                const MODEL_GUIDE_DEFAULT = {default_guide_json};
                const DEV_BYPASS = {"true" if DEV_AUTH_BYPASS else "false"};
                const sendBtn = document.getElementById("send");
                const resetBtn = document.getElementById("reset");
                const msgEl = document.getElementById("msg");
                const modelEl = document.getElementById("model");
                const modelNote = document.getElementById("model-note");
                const statusEl = document.getElementById("status");
                const logEl = document.getElementById("log");
                const waitEl = document.getElementById("wait");
                const waitTextEl = document.getElementById("wait-text");
                let modelList = [];
                let history = [];
                let currentConversationId = null;
                let sending = false;
                let waitTimer = null;
                let waitStart = null;

                function authHeaders() {{
                  const h = {{ "Accept": "application/json" }};
                  const key = (localStorage.getItem("rag_api_key") || "").trim();
                  if (key) h["Authorization"] = "Bearer " + key;
                  return h;
                }}

                function authStatus() {{
                  const hasKey = !!(localStorage.getItem("rag_api_key") || "").trim();
                  if (hasKey) {{
                    statusEl.innerHTML = "Using stored API key for requests.";
                  }} else if (DEV_BYPASS) {{
                    statusEl.innerHTML = "DEV_AUTH_BYPASS active; using dev user.";
                  }} else {{
                    statusEl.innerHTML = 'Provide an API key at <a href="/ui/api-keys" style="color:#38bdf8;">/ui/api-keys</a> or via Cloudflare Access.';
                  }}
                }}

                function currentModel() {{
                  return modelList.length === 1 ? modelList[0] : modelEl.value.trim();
                }}

                function guideForModel(name) {{
                  const n = (name || "").toLowerCase();
                  for (const r of MODEL_GUIDE_RULES) {{
                    const key = (r.match || "").toLowerCase();
                    if (key && n.includes(key)) {{
                      return {{ title: r.title, purpose: r.purpose }};
                    }}
                  }}
                  return {{ title: MODEL_GUIDE_DEFAULT.title, purpose: MODEL_GUIDE_DEFAULT.purpose }};
                }}

                function renderModelGuide() {{
                  const wrap = document.getElementById("model-guide");
                  if (!wrap) return;
                  wrap.innerHTML = "";
                  if (!modelList.length) return;
                  const h2 = document.createElement("h2");
                  h2.textContent = "Available models";
                  wrap.appendChild(h2);
                  const sub = document.createElement("p");
                  sub.className = "note";
                  sub.style.margin = "0 0 14px";
                  sub.textContent = "What each installed model is good for (heuristic blurbs; exact behavior depends on prompt and parameters).";
                  wrap.appendChild(sub);
                  const grid = document.createElement("div");
                  grid.className = "model-guide-grid";
                  const sorted = modelList.slice().sort((a, b) => a.localeCompare(b));
                  sorted.forEach((m) => {{
                    const g = guideForModel(m);
                    const card = document.createElement("article");
                    card.className = "model-guide-card";
                    const idEl = document.createElement("div");
                    idEl.className = "model-id";
                    idEl.textContent = m;
                    const h3 = document.createElement("h3");
                    h3.textContent = g.title;
                    const p = document.createElement("p");
                    p.textContent = g.purpose;
                    card.appendChild(idEl);
                    card.appendChild(h3);
                    card.appendChild(p);
                    grid.appendChild(card);
                  }});
                  wrap.appendChild(grid);
                }}

                function parseQuery() {{
                  const params = new URLSearchParams(window.location.search);
                  return {{
                    cid: params.get("cid") || null
                  }};
                }}

                async function loadConversation(convId) {{
                  if (!convId) return;
                  statusEl.textContent = "Loading conversation...";
                  try {{
                    const r = await fetch(`/conversations/${{convId}}`, {{ headers: authHeaders() }});
                    if (!r.ok) {{
                      statusEl.textContent = "Load failed (" + r.status + ").";
                      return;
                    }}
                    const data = await r.json();
                    currentConversationId = data.id;
                    history = (data.messages || []).map(m => {{ return {{ role: m.role, content: m.content }}; }});
                    if (data.model && modelList.includes(data.model)) {{
                      modelEl.value = data.model;
                    }}
                    renderHistory();
                    statusEl.textContent = data.title ? `Conversation: ${{data.title}}` : "Conversation loaded.";
                  }} catch (e) {{
                    statusEl.textContent = "Load failed.";
                  }}
                }}

                async function loadModels() {{
                  modelNote.textContent = "Loading models...";
                  authStatus();
                  try {{
                    const r = await fetch("/models", {{ headers: authHeaders() }});
                    const data = await r.json();
                    modelList = (data.models || []).filter(m => !!m);
                  }} catch (e) {{
                    modelList = ["{CHAT_MODEL}"];
                  }}
                  if (!modelList.length) modelList = ["{CHAT_MODEL}"];

                  if (modelList.length === 1) {{
                    modelEl.style.display = "none";
                    modelNote.textContent = "Using model: " + modelList[0];
                  }} else {{
                    modelEl.style.display = "block";
                    modelNote.textContent = "";
                    modelEl.innerHTML = "";
                    modelList.forEach(m => {{
                      const opt = document.createElement("option");
                      opt.value = m;
                      opt.textContent = m;
                      modelEl.appendChild(opt);
                    }});
                    const current = modelList.find(m => m === "{CHAT_MODEL}");
                    modelEl.value = current || modelList[0];
                  }}
                  renderModelGuide();
                }}

                function renderHistory() {{
                  if (!history.length) {{
                    logEl.textContent = "";
                    return;
                  }}
                  logEl.innerHTML = "";
                  history.forEach((m) => {{
                    const row = document.createElement("div");
                    row.className = "msg-row";
                    const bubble = document.createElement("div");
                    bubble.className = "bubble " + (m.role === "assistant" ? "assistant" : "user");
                    bubble.textContent = m.content;
                    row.appendChild(bubble);
                    logEl.appendChild(row);
                  }});
                  logEl.scrollTop = logEl.scrollHeight;
                }}

                function resetConversation() {{
                  history = [];
                  msgEl.value = "";
                  currentConversationId = null;
                  statusEl.textContent = "Conversation cleared. Next send will start a new conversation.";
                  renderHistory();
                }}

                function setSending(on) {{
                  sending = on;
                  msgEl.disabled = on;
                  modelEl.disabled = on;
                  sendBtn.disabled = on;
                  resetBtn.disabled = on;
                  waitEl.style.display = on ? "flex" : "none";
                  if (on) {{
                    waitStart = Date.now();
                    waitTextEl.textContent = "Waiting for reply\u2026 0s";
                    waitTimer = setInterval(() => {{
                      const secs = Math.floor((Date.now() - waitStart) / 1000);
                      waitTextEl.textContent = `Waiting for reply\u2026 ${{secs}}s`;
                    }}, 1000);
                  }} else {{
                    if (waitTimer) {{
                      clearInterval(waitTimer);
                      waitTimer = null;
                    }}
                    waitStart = null;
                  }}
                }}

                async function send() {{
                  if (sending) return;
                  const content = msgEl.value.trim();
                  const model = currentModel();
                  if (!content) {{
                    statusEl.textContent = "Please enter a message.";
                    return;
                  }}
                  setSending(true);
                  const messages = [...history, {{ role: "user", content }}];
                  const payload = {{ messages, stream: true }};
                  if (currentConversationId) payload.conversation_id = currentConversationId;
                  if (model) payload.model = model;

                  history = [...history, {{ role: "user", content }}];
                  renderHistory();
                  msgEl.value = "";

                  const row = document.createElement("div");
                  row.className = "msg-row";
                  const bubble = document.createElement("div");
                  bubble.className = "bubble assistant";
                  bubble.textContent = "";
                  row.appendChild(bubble);
                  logEl.appendChild(row);

                  let accContent = "";
                  let convId = currentConversationId;
                  let chunkCount = 0;

                  try {{
                    const r = await fetch("/chat", {{
                      method: "POST",
                      headers: Object.assign({{ "Content-Type": "application/json" }}, authHeaders()),
                      body: JSON.stringify(payload),
                    }});

                    if (!r.ok) {{
                      const errText = await r.text();
                      console.error("[chat] HTTP error:", r.status, errText);
                      bubble.textContent = `Error: ${{r.status}} \u2014 ${{errText.substring(0, 200)}}`;
                      bubble.style.color = "#f87171";
                      statusEl.textContent = `Request failed (${{r.status}}).`;
                      setSending(false);
                      return;
                    }}

                    const reader = r.body.getReader();
                    const decoder = new TextDecoder();
                    let buffer = "";

                    while (true) {{
                      const {{ done, value }} = await reader.read();
                      if (done) break;
                      buffer += decoder.decode(value, {{ stream: true }});

                      const lines = buffer.split("\\n");
                      buffer = lines.pop();

                      for (const line of lines) {{
                        const trimmed = line.trim();
                        if (!trimmed || !trimmed.startsWith("data: ")) continue;
                        const payload_str = trimmed.slice(6);
                        if (payload_str === "[DONE]") {{
                          console.debug("[chat] stream done, chunks:", chunkCount);
                          continue;
                        }}
                        try {{
                          const chunk = JSON.parse(payload_str);
                          chunkCount++;

                          if (chunk.conversation_id && !convId) {{
                            convId = chunk.conversation_id;
                            currentConversationId = convId;
                          }}

                          if (chunk.choices && chunk.choices.length > 0) {{
                            const delta = chunk.choices[0].delta || {{}};
                            if (delta.content) {{
                              accContent += delta.content;
                              bubble.textContent = accContent;
                              logEl.scrollTop = logEl.scrollHeight;
                            }}
                            if (chunkCount % 5 === 0) {{
                              const secs = Math.floor((Date.now() - waitStart) / 1000);
                              waitTextEl.textContent = `Streaming\u2026 ${{accContent.length}} chars, ${{secs}}s`;
                            }}
                          }}

                          if (chunk.error) {{
                            console.error("[chat] stream error:", chunk.error);
                            bubble.textContent += "\\n[Error: " + chunk.error + "]";
                            bubble.style.color = "#f87171";
                          }}
                        }} catch (e) {{
                          console.warn("[chat] chunk parse error:", e, "raw:", payload_str);
                        }}
                      }}
                    }}

                    if (accContent) {{
                      history = [...history, {{ role: "assistant", content: accContent }}];
                      renderHistory();
                      msgEl.focus();
                      statusEl.textContent = (convId ? `Conversation: ${{convId}}` : "") + ` | ${{accContent.length}} chars`;
                    }} else {{
                      console.warn("[chat] No content accumulated from stream");
                      statusEl.textContent = "No response content received.";
                    }}
                  }} catch (e) {{
                    console.error("[chat] Stream error:", e);
                    bubble.textContent = "Connection error: " + e.message;
                    bubble.style.color = "#f87171";
                    statusEl.textContent = "Send failed: " + e.message;
                  }}
                  setSending(false);
                }}

                sendBtn.addEventListener("click", send);
                msgEl.addEventListener("keydown", (e) => {{
                  if (e.key === "Enter" && !e.shiftKey) {{
                    e.preventDefault();
                    send();
                  }}
                }});
                resetBtn.addEventListener("click", resetConversation);
                loadModels();
                const q = parseQuery();
                if (q.cid) {{
                  loadConversation(q.cid);
                }}
                renderHistory();
              </script>
            </body>
            </html>
            """
        )
    )


@router.get("/ui/pipeline", response_class=HTMLResponse)
def pipeline_ui():
    extra_css = """
    /* ---- layout ---- */
    .pl-wrap {
      width: min(1280px, 100%);
      background: linear-gradient(180deg, var(--card-2), var(--card));
      border: 1px solid var(--border);
      border-radius: 16px;
      box-shadow: 0 20px 60px rgba(0,0,0,0.35);
      padding: 20px 24px;
      margin-top: 50px;
      min-height: 92vh;
    }
    .pl-sub { color: var(--muted); font-size: 13px; margin: 0 0 14px; }
    .pl-columns { display: grid; grid-template-columns: 320px 1fr; gap: 20px; }
    .pl-left { display: flex; flex-direction: column; gap: 14px; }
    .pl-right { display: flex; flex-direction: column; gap: 0; }
    /* ---- left panel inputs ---- */
    .pl-section {
      background: rgba(0,0,0,0.2); border: 1px solid var(--border);
      border-radius: 12px; padding: 14px;
    }
    .pl-section label {
      font-size: 13px; font-weight: 600; display: flex;
      align-items: center; gap: 5px; margin-bottom: 6px;
    }
    .pl-section textarea, .pl-section input[type=text], .pl-section input[type=number] {
      width: 100%; background: #0b1221; border: 1px solid var(--border);
      border-radius: 8px; color: var(--text); font: inherit; font-size: 13px;
      padding: 8px 10px; resize: vertical; box-sizing: border-box;
    }
    .pl-section textarea:focus, .pl-section input:focus {
      outline: none; border-color: var(--accent);
    }
    .pl-section select {
      width: 100%; background: #0b1221; border: 1px solid var(--border);
      border-radius: 8px; color: var(--text); font: inherit; font-size: 13px;
      padding: 7px 10px; box-sizing: border-box;
    }
    .turn-row { display: flex; gap: 6px; margin-bottom: 6px; align-items: center; }
    .turn-row input { flex: 1; }
    .turn-del {
      flex-shrink: 0; width: 28px; height: 28px; border-radius: 6px;
      border: 1px solid var(--border); background: #0b1221;
      color: var(--muted); font-size: 16px; cursor: pointer; line-height: 1;
    }
    .turn-del:hover { color: #f87171; border-color: #f87171; }
    .pl-link-btn {
      font-size: 12px; color: var(--accent); background: none; border: none;
      cursor: pointer; padding: 0; text-decoration: underline;
    }
    .pl-btn-sm {
      padding: 4px 10px; font-size: 12px; font-weight: 600;
      border-radius: 8px; border: 1px solid var(--border);
      background: #0b1221; color: var(--text); cursor: pointer;
      transition: border-color 120ms; white-space: nowrap;
    }
    .pl-btn-sm:hover { border-color: var(--accent); }
    .pl-btn-sm:disabled { opacity: 0.4; cursor: not-allowed; }
    .run-btn {
      width: 100%; padding: 11px; font-size: 14px; font-weight: 700;
      border-radius: 10px; border: none; cursor: pointer;
      background: linear-gradient(90deg, var(--accent), var(--accent-2));
      color: #0b1020; transition: opacity 120ms, transform 120ms;
      box-shadow: 0 6px 18px rgba(56,189,248,0.25);
    }
    .run-btn:hover { opacity: 0.9; transform: translateY(-1px); }
    .run-btn:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }
    /* ---- stage cards ---- */
    .stage-card {
      border: 1px solid var(--border); border-radius: 12px;
      padding: 14px 16px; background: rgba(0,0,0,0.18);
      transition: border-color 300ms, background 300ms;
    }
    .stage-card.live    { border-color: rgba(56,189,248,0.35); }
    .stage-card.running { border-color: rgba(251,191,36,0.7); background: rgba(251,191,36,0.04);
      animation: pulse-border 1.6s ease-in-out infinite; }
    .stage-card.complete { border-color: rgba(52,211,153,0.65); background: rgba(52,211,153,0.04); }
    .stage-card.s-error  { border-color: rgba(239,68,68,0.55); }
    @keyframes pulse-border {{
      0%,100% {{ border-color: rgba(251,191,36,0.4); }}
      50%      {{ border-color: rgba(251,191,36,0.85); }}
    }}
    .stage-connector {
      display: flex; align-items: center; justify-content: center;
      color: var(--muted); font-size: 20px; padding: 2px 0;
    }
    .stage-header {
      display: flex; align-items: center; gap: 8px; margin-bottom: 8px; flex-wrap: wrap;
    }
    .stage-name { font-weight: 700; font-size: 15px; }
    .stage-badge {
      font-size: 11px; font-weight: 700; letter-spacing: 0.05em;
      padding: 2px 8px; border-radius: 999px; text-transform: uppercase;
    }
    .stage-badge.live { background: rgba(56,189,248,0.18); color: var(--accent); }
    .stage-run-btn {
      margin-left: auto; padding: 4px 12px; font-size: 12px; font-weight: 700;
      border-radius: 8px; border: none; cursor: pointer;
      background: linear-gradient(90deg, var(--accent), var(--accent-2));
      color: #0b1020; transition: opacity 120ms, transform 120ms;
      box-shadow: 0 3px 10px rgba(56,189,248,0.2); white-space: nowrap;
    }
    .stage-run-btn:hover { opacity: 0.88; transform: translateY(-1px); }
    .stage-run-btn:disabled { opacity: 0.35; cursor: not-allowed; transform: none; }
    .stage-spin {
      width: 13px; height: 13px; border: 2px solid rgba(255,255,255,0.15);
      border-top-color: #fbbf24; border-radius: 50%;
      animation: spin 0.75s linear infinite; display: inline-block; flex-shrink: 0;
    }
    @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
    .stage-meta { font-size: 12px; color: var(--muted); display: flex; flex-direction: column; gap: 3px; }
    .stage-knows { font-style: italic; margin-bottom: 2px; cursor: default; }
    .stage-io b { color: var(--text); font-weight: 600; }
    .stage-output {
      margin-top: 12px; padding: 10px 12px;
      background: rgba(56,189,248,0.06); border: 1px solid rgba(56,189,248,0.2);
      border-radius: 8px;
    }
    .stage-output-title {
      font-size: 11px; font-weight: 700; color: var(--accent);
      letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 8px;
    }
    .error-box {
      padding: 8px 12px; border-radius: 8px; font-size: 13px;
      background: rgba(239,68,68,0.12); border: 1px solid rgba(239,68,68,0.4);
      color: #f87171; margin-top: 8px;
    }
    /* ---- classifier output ---- */
    .token-chips { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
    .token-chip {
      display: flex; align-items: center; gap: 6px;
      background: rgba(99,102,241,0.18); border: 1px solid rgba(99,102,241,0.4);
      border-radius: 8px; padding: 4px 10px; font-size: 13px;
    }
    .token-score { color: var(--accent); font-weight: 700; font-size: 12px; }
    .score-bars { display: flex; flex-direction: column; gap: 4px; }
    .score-row { display: flex; align-items: center; gap: 8px; font-size: 12px; }
    .score-label {
      width: 110px; text-align: right; color: var(--muted);
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .score-bar-bg { flex: 1; height: 6px; background: #1f2937; border-radius: 3px; }
    .score-bar-fill {
      height: 100%; border-radius: 3px;
      background: linear-gradient(90deg, var(--accent-2), var(--accent));
    }
    .score-val { width: 42px; color: var(--text); font-variant-numeric: tabular-nums; }
    /* ---- inline model state (stage 1 header) ---- */
    .state-chip {
      padding: 2px 8px; border-radius: 999px; font-size: 11px;
      font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase;
    }
    .state-chip.unloaded { background: #1f2937; color: var(--muted); }
    .state-chip.loading  { background: rgba(251,191,36,0.2); color: #fbbf24; }
    .state-chip.cpu      { background: rgba(99,102,241,0.2); color: #818cf8; }
    .state-chip.gpu      { background: rgba(52,211,153,0.2); color: #34d399; }
    .state-chip.error    { background: rgba(239,68,68,0.15); color: #f87171; }
    .gpu-cap-chip {
      padding: 2px 7px; border-radius: 999px; font-size: 11px;
      font-weight: 700; cursor: default;
    }
    .gpu-cap-chip.capable  { background: rgba(52,211,153,0.12); color: #34d399; border: 1px solid rgba(52,211,153,0.3); }
    .gpu-cap-chip.cpu-only { background: #1f2937; color: var(--muted); border: 1px solid var(--border); }
    /* ---- timer + download ---- */
    .pl-timer {
      font-variant-numeric: tabular-nums; font-size: 13px;
      color: var(--accent); font-weight: 700; min-width: 48px; text-align: right;
    }
    .pl-dl-btn {
      padding: 4px 10px; font-size: 12px; font-weight: 600;
      border-radius: 8px; border: 1px solid rgba(52,211,153,0.4);
      background: rgba(52,211,153,0.1); color: #34d399; cursor: pointer;
      transition: border-color 120ms;
    }
    .pl-dl-btn:hover { border-color: #34d399; }
    /* ---- help icon ---- */
    .help-ico {
      display: inline-flex; align-items: center; justify-content: center;
      width: 14px; height: 14px; border-radius: 50%;
      background: rgba(148,163,184,0.2); color: var(--muted);
      font-size: 10px; font-weight: 700; cursor: default; flex-shrink: 0;
    }
    /* ---- modals ---- */
    .modal-backdrop {
      position: fixed; inset: 0; background: rgba(0,0,0,0.6);
      display: flex; align-items: center; justify-content: center;
      z-index: 200; backdrop-filter: blur(3px);
    }
    .modal-box {
      background: var(--card-2); border: 1px solid var(--border);
      border-radius: 16px; padding: 24px; width: min(500px, 92vw);
      box-shadow: 0 24px 80px rgba(0,0,0,0.55); max-height: 85vh; overflow-y: auto;
    }
    .modal-title { margin: 0 0 18px; font-size: 17px; font-weight: 700; }
    .modal-footer {
      display: flex; gap: 8px; justify-content: flex-end;
      margin-top: 20px; padding-top: 16px; border-top: 1px solid var(--border);
    }
    .cfg-row { margin-bottom: 14px; }
    .cfg-row label {
      font-size: 12px; font-weight: 600; color: var(--muted);
      display: flex; align-items: center; gap: 5px; margin-bottom: 5px;
    }
    .cfg-row select, .cfg-row input {
      width: 100%; background: #0b1221; border: 1px solid var(--border);
      border-radius: 8px; color: var(--text); font: inherit; font-size: 13px;
      padding: 7px 10px; box-sizing: border-box;
    }
    .cfg-row input[type=number] { width: 80px; }
    .btn-secondary {
      padding: 8px 14px; font-size: 13px; font-weight: 600;
      border-radius: 8px; border: 1px solid var(--border);
      background: #0b1221; color: var(--text); cursor: pointer;
    }
    .btn-danger { border-color: rgba(239,68,68,0.4); color: #f87171; }
    .btn-danger:hover { background: rgba(239,68,68,0.1); }
    @media (max-width: 900px) {{ .pl-columns {{ grid-template-columns: 1fr; }} }}
    """

    body = f"""
    <!-- Config modal -->
    <div id="cfgModal" class="modal-backdrop" style="display:none;" onclick="if(event.target===this)closeCfg()">
      <div class="modal-box">
        <h3 class="modal-title" id="cfgTitle">Run configuration</h3>
        <div id="cfgBody"></div>
        <div class="modal-footer">
          <button class="btn-secondary" onclick="closeCfg()">Cancel</button>
          <button class="run-btn" style="width:auto;padding:8px 22px;" onclick="confirmRun()">Run &#9654;</button>
        </div>
      </div>
    </div>

    <!-- Character modal -->
    <div id="charModal" class="modal-backdrop" style="display:none;" onclick="if(event.target===this)closeCharModal()">
      <div class="modal-box">
        <h3 class="modal-title" id="charModalTitle">Character</h3>
        <div class="cfg-row">
          <label>Name</label>
          <input type="text" id="cmName" placeholder="Character name (required)" />
        </div>
        <div class="cfg-row">
          <label>Persona <span class="help-ico" id="tipPersona">?</span></label>
          <textarea id="cmPersona" rows="5" style="width:100%;background:#0b1221;border:1px solid var(--border);border-radius:8px;color:var(--text);font:inherit;font-size:13px;padding:8px 10px;resize:vertical;box-sizing:border-box;" placeholder="Personality, background, speech patterns, values&hellip;"></textarea>
        </div>
        <div class="cfg-row">
          <label>Current state <span class="help-ico" id="tipState">?</span></label>
          <input type="text" id="cmState" placeholder="e.g. nervous about the audition, feeling playful" />
        </div>
        <div class="modal-footer">
          <button class="btn-secondary btn-danger" id="cmDeleteBtn" onclick="deleteChar()" style="margin-right:auto;display:none;">Delete</button>
          <button class="btn-secondary" onclick="closeCharModal()">Cancel</button>
          <button class="run-btn" style="width:auto;padding:8px 22px;" onclick="saveChar()">Save</button>
        </div>
      </div>
    </div>

    <!-- Tooltip div -->
    <div id="plTip" style="position:fixed;display:none;background:#0d1527;border:1px solid var(--border);border-radius:9px;padding:9px 13px;font-size:12px;color:var(--text);max-width:270px;z-index:400;pointer-events:none;line-height:1.55;box-shadow:0 8px 28px rgba(0,0,0,0.45);"></div>

    <div class="pl-wrap">
      <div style="display:flex;align-items:center;gap:12px;margin-bottom:4px;flex-wrap:wrap;">
        <h1 style="margin:0;font-size:22px;">Pipeline Lab</h1>
        <span style="color:var(--muted);font-size:13px;">4-stage character pipeline</span>
        <span style="margin-left:auto;display:flex;gap:10px;align-items:center;">
          <button class="pl-dl-btn" id="dlBtn" onclick="downloadRun()" style="display:none;">&#8595; Download run</button>
          <span id="plTimer" class="pl-timer" style="display:none;">0.0s</span>
        </span>
      </div>
      <p class="pl-sub">Run stages individually or all at once. Each stage receives only what it needs — no more.</p>

      <div class="pl-columns">

        <div class="pl-left">

          <div class="pl-section">
            <label>Label set <span class="help-ico" id="tipLabels">?</span></label>
            <textarea id="labelsInput" rows="5" placeholder="curious, defensive, playful, sad, angry, flirtatious, neutral, &hellip;"></textarea>
            <div style="display:flex;gap:8px;align-items:center;margin-top:8px;">
              <button class="pl-btn-sm" onclick="updateLabels()">Update labels</button>
              <span id="labelsStatus" style="color:var(--muted);font-size:12px;"></span>
            </div>
            <div style="color:var(--muted);font-size:11px;margin-top:6px;">Comma or newline separated. Re-embeds if model is resident.</div>
          </div>

          <div class="pl-section">
            <label>Character <span class="help-ico" id="tipChar">?</span></label>
            <div style="display:flex;gap:6px;align-items:center;">
              <select id="charSelect" style="flex:1;">
                <option value="">— no character —</option>
              </select>
              <button class="pl-btn-sm" id="charEditBtn" onclick="openCharModal(false)" style="display:none;">Edit</button>
              <button class="pl-btn-sm" onclick="openCharModal(true)">New</button>
            </div>
            <div id="charPreview" style="margin-top:6px;font-size:12px;color:var(--muted);display:none;"></div>
          </div>

          <div class="pl-section">
            <label>Context turns <span class="help-ico" id="tipTurns">?</span> <span style="font-weight:400;color:var(--muted);font-size:12px;">(oldest first)</span></label>
            <div id="turnsList"></div>
            <button class="pl-link-btn" onclick="addTurn()" style="margin-top:4px;">+ add turn</button>
          </div>

          <div class="pl-section">
            <label>User message <span class="help-ico" id="tipMsg">?</span></label>
            <textarea id="msgInput" rows="4" placeholder="what the user just sent&hellip;"></textarea>
            <button class="run-btn" id="runAllBtn" onclick="openRunModal('all')" style="margin-top:10px;">
              Run all &rarr;
            </button>
          </div>

        </div>

        <div class="pl-right">

          <div class="stage-card live" id="stage1Card">
            <div class="stage-header">
              <span class="stage-name">1 &middot; ModernBERT</span>
              <span class="stage-badge live">LIVE &middot; 149M params</span>
              <span style="display:flex;align-items:center;gap:4px;">
                <span id="mbStateChip" class="state-chip unloaded">unloaded</span>
                <span id="mbGpuChip" style="display:none;"></span>
                <button id="mbLoadBtn" class="pl-btn-sm" onclick="mbLoad()" style="display:none;">Load</button>
                <button id="mbEvictBtn" class="pl-btn-sm" onclick="mbEvict()" style="display:none;">&rarr; CPU</button>
              </span>
              <span id="stage1Spin" class="stage-spin" style="display:none;"></span>
              <button class="stage-run-btn" id="run1Btn" onclick="openRunModal(1)">Run &#9654;</button>
            </div>
            <div class="stage-meta">
              <div class="stage-knows" id="tipS1knows">Knows: user message + up to 5 turns of context. Not who the character is.</div>
              <div class="stage-io"><b>Gets:</b> raw text &mdash; message + turn context</div>
              <div class="stage-io"><b>Produces:</b> ranked label tokens + scores</div>
            </div>
            <div id="stage1Output" style="display:none;" class="stage-output">
              <div class="stage-output-title">Classifier tokens</div>
              <div id="tokenChips" class="token-chips"></div>
              <div id="scoreBars" class="score-bars"></div>
            </div>
            <div id="stage1Error" style="display:none;" class="error-box"></div>
          </div>

          <div class="stage-connector">&#8595;</div>

          <div class="stage-card live" id="stage2Card">
            <div class="stage-header">
              <span class="stage-name">2 &middot; 4B Overlay</span>
              <span class="stage-badge live">LIVE &middot; style tuner</span>
              <span id="stage2Spin" class="stage-spin" style="display:none;"></span>
              <button class="stage-run-btn" id="run2Btn" onclick="openRunModal(2)">Run &#9654;</button>
            </div>
            <div class="stage-meta">
              <div class="stage-knows" id="tipS2knows">Knows: character config + current state. Not full session history.</div>
              <div class="stage-io"><b>Gets:</b> character + classifier tokens + turn context</div>
              <div class="stage-io"><b>Produces:</b> 80&ndash;120 word voice directive</div>
            </div>
            <div id="stage2Output" style="display:none;" class="stage-output">
              <div class="stage-output-title">Voice directive <span id="stage2Model" style="font-weight:400;text-transform:none;letter-spacing:0;font-size:11px;"></span></div>
              <div id="stage2Text" style="white-space:pre-wrap;font-size:13px;line-height:1.55;"></div>
            </div>
            <div id="stage2Error" style="display:none;" class="error-box"></div>
          </div>

          <div class="stage-connector">&#8595;</div>

          <div class="stage-card live" id="stage3Card">
            <div class="stage-header">
              <span class="stage-name">3 &middot; Brain</span>
              <span class="stage-badge live">LIVE &middot; large &middot; once/turn</span>
              <span id="stage3Spin" class="stage-spin" style="display:none;"></span>
              <button class="stage-run-btn" id="run3Btn" onclick="openRunModal(3)">Run &#9654;</button>
            </div>
            <div class="stage-meta">
              <div class="stage-knows" id="tipS3knows">Knows: everything — character, state, history, classifier read, voice directive. Does not write in character voice.</div>
              <div class="stage-io"><b>Gets:</b> full context + directive + classifier + history</div>
              <div class="stage-io"><b>Produces:</b> structured plan (intent, key points, arc, avoid)</div>
            </div>
            <div id="stage3Output" style="display:none;" class="stage-output">
              <div class="stage-output-title">Skeleton <span id="stage3Model" style="font-weight:400;text-transform:none;letter-spacing:0;font-size:11px;"></span></div>
              <div id="stage3Text" style="white-space:pre-wrap;font-size:13px;line-height:1.55;"></div>
            </div>
            <div id="stage3Error" style="display:none;" class="error-box"></div>
          </div>

          <div class="stage-connector">&#8595;</div>

          <div class="stage-card live" id="stage4Card">
            <div class="stage-header">
              <span class="stage-name">4 &middot; Prose Model</span>
              <span class="stage-badge live">LIVE &middot; medium &middot; streams</span>
              <span id="stage4Spin" class="stage-spin" style="display:none;"></span>
              <button class="stage-run-btn" id="run4Btn" onclick="openRunModal(4)">Run &#9654;</button>
            </div>
            <div class="stage-meta">
              <div class="stage-knows" id="tipS4knows">Knows: how to write in the character's voice + what to say from the skeleton. Does not reason over the relationship arc.</div>
              <div class="stage-io"><b>Gets:</b> directive + skeleton + recent history</div>
              <div class="stage-io"><b>Produces:</b> character&apos;s streamed response</div>
            </div>
            <div id="stage4Output" style="display:none;" class="stage-output">
              <div class="stage-output-title">Response <span id="stage4Model" style="font-weight:400;text-transform:none;letter-spacing:0;font-size:11px;"></span></div>
              <div id="stage4Text" style="white-space:pre-wrap;font-size:13px;line-height:1.55;"></div>
            </div>
            <div id="stage4Error" style="display:none;" class="error-box"></div>
          </div>

        </div>
      </div>
    </div>

    <script>
      const DEV_BYPASS = {"true" if DEV_AUTH_BYPASS else "false"};
      const DEFAULT_MODEL = "{CHAT_MODEL}";

      // ---- auth ----
      function apiHeaders() {{
        const h = {{"Content-Type": "application/json"}};
        if (!DEV_BYPASS) {{
          const k = (localStorage.getItem("rag_api_key") || "").trim();
          if (k) h["Authorization"] = "Bearer " + k;
        }}
        return h;
      }}

      // ---- tooltip ----
      const _tipEl = document.getElementById("plTip");
      function bindTip(id, text) {{
        const el = document.getElementById(id);
        if (!el) return;
        el.addEventListener("mouseenter", () => {{ _tipEl.textContent = text; _tipEl.style.display = "block"; }});
        el.addEventListener("mousemove", e => {{ _tipEl.style.left=(e.clientX+14)+"px"; _tipEl.style.top=(e.clientY+14)+"px"; }});
        el.addEventListener("mouseleave", () => {{ _tipEl.style.display = "none"; }});
      }}
      bindTip("tipLabels", "Emotion/intent words the classifier maps text to. ModernBERT embeds both the input and each label, then ranks them by cosine similarity. Try: curious, defensive, playful, sad, flirtatious, guarded, warm.");
      bindTip("tipChar",   "Character config passed to stages 2–4. The Overlay sets voice/tone, the Brain reasons as the character, the Prose writes in their voice.");
      bindTip("tipTurns",  "Recent conversation turns (oldest first, up to 5 used). Passed to stage 1 as context and to stages 2–4 as history.");
      bindTip("tipMsg",    "The message the user just sent. Stage 1 classifies its emotional signal; stages 3–4 use it as the message to respond to.");
      bindTip("tipPersona","Personality, background, speech patterns, values, relationships. The Brain and Prose models use this to reason and write as the character.");
      bindTip("tipState",  "Optional: character's current mood or situational context, updated per conversation turn. E.g. 'nervous about the audition, feeling hopeful'.");
      bindTip("tipS1knows","ModernBERT only sees the raw text — it has no knowledge of the character, making the signal unbiased.");
      bindTip("tipS2knows","The Overlay knows the character and classifier signal, but not the full history, keeping it focused on style rather than content.");
      bindTip("tipS3knows","The Brain sees everything and plans the response content. It deliberately does not write in character voice — that's stage 4's job.");
      bindTip("tipS4knows","The Prose model only knows how to write (directive) and what to write (skeleton). It doesn't reason over history or relationships.");

      // ---- timer ----
      let _timerIv = null;
      function startTimer() {{
        const el = document.getElementById("plTimer");
        el.style.display = "inline";
        const t0 = Date.now();
        if (_timerIv) clearInterval(_timerIv);
        _timerIv = setInterval(() => {{
          const s = (Date.now() - t0) / 1000;
          el.textContent = s < 60 ? s.toFixed(1) + "s" : Math.floor(s/60) + "m " + (s%60|0) + "s";
        }}, 100);
      }}
      function stopTimer() {{ if (_timerIv) {{ clearInterval(_timerIv); _timerIv = null; }} }}

      // ---- stage state ----
      function setStageState(n, state) {{
        const card = document.getElementById("stage" + n + "Card");
        if (card) {{ card.classList.remove("running","complete","s-error"); if (state !== "idle") card.classList.add(state === "error" ? "s-error" : state); }}
        const spin = document.getElementById("stage" + n + "Spin");
        if (spin) spin.style.display = state === "running" ? "inline-block" : "none";
        const btn = document.getElementById("run" + n + "Btn");
        if (btn) btn.disabled = state === "running";
        document.getElementById("runAllBtn").disabled = [1,2,3,4].some(i => {{
          const c = document.getElementById("stage"+i+"Card");
          return c && c.classList.contains("running");
        }});
      }}
      function resetDownstream(from) {{
        for (let n = from; n <= 4; n++) {{
          setStageState(n, "idle");
          const o = document.getElementById("stage"+n+"Output"); if (o) o.style.display = "none";
          const e = document.getElementById("stage"+n+"Error");  if (e) e.style.display = "none";
        }}
        if (from <= 1) {{ _classifyResult = null; _runData.stage1 = null; }}
        if (from <= 2) {{ _directive = ""; _runData.stage2 = null; }}
        if (from <= 3) {{ _skeleton  = ""; _runData.stage3 = null; }}
        if (from <= 4) {{ _runData.stage4 = null; }}
        document.getElementById("dlBtn").style.display = "none";
      }}

      // ---- ModernBERT inline state ----
      function fmtKa(s) {{
        if (s==null) return "?"; if (s<0) return "∞"; if (s===0) return "0s";
        return s>=3600 ? (s/3600|0)+"h" : s>=60 ? (s/60|0)+"m" : s+"s";
      }}
      function updateMbState(state, device, gpuCapable, configuredDevice, keepAliveGpu) {{
        const chip = document.getElementById("mbStateChip");
        chip.className = "state-chip " + (state||"unloaded");
        chip.textContent = state||"unloaded";
        const gpuChip = document.getElementById("mbGpuChip");
        const loadBtn = document.getElementById("mbLoadBtn");
        const evictBtn = document.getElementById("mbEvictBtn");
        if (gpuCapable != null) {{
          gpuChip.style.display = "";
          if (gpuCapable) {{
            gpuChip.className = "gpu-cap-chip capable";
            gpuChip.textContent = "⚡";
            gpuChip.title = "GPU capable · idle timer: " + fmtKa(keepAliveGpu);
          }} else {{
            gpuChip.className = "gpu-cap-chip cpu-only";
            gpuChip.textContent = "CPU";
            gpuChip.title = "CPU only · " + (configuredDevice==="cpu" ? "forced CPU" : "no CUDA");
          }}
        }} else {{ gpuChip.style.display = "none"; }}
        loadBtn.style.display  = (state==="unloaded"||state==="error") ? "" : "none";
        evictBtn.style.display = (state==="gpu") ? "" : "none";
      }}
      async function loadStatus() {{
        try {{
          const r = await fetch("/api/pipeline/status", {{headers: apiHeaders()}});
          if (!r.ok) {{ updateMbState("error"); return; }}
          const d = await r.json();
          updateMbState(d.state, d.device, d.gpu_capable, d.configured_device, d.keep_alive_gpu);
          if (!document.getElementById("labelsInput").value.trim()) loadLabels();
        }} catch(e) {{ updateMbState("error"); }}
      }}
      async function mbAction(path) {{
        updateMbState("loading");
        try {{
          const r = await fetch(path, {{method:"POST", headers:apiHeaders()}});
          const d = await r.json();
          updateMbState(d.state, d.device);
        }} catch(e) {{ updateMbState("error"); }}
        setTimeout(loadStatus, 800);
      }}
      const mbLoad  = () => mbAction("/api/pipeline/load");
      const mbEvict = () => mbAction("/api/pipeline/evict");

      // ---- labels ----
      async function loadLabels() {{
        try {{
          const r = await fetch("/api/pipeline/labels", {{headers:apiHeaders()}});
          if (!r.ok) return;
          const d = await r.json();
          if (d.labels && d.labels.length) document.getElementById("labelsInput").value = d.labels.join(", ");
        }} catch(_) {{}}
      }}
      async function updateLabels() {{
        const labels = document.getElementById("labelsInput").value.split(/[,\\n]+/).map(s=>s.trim()).filter(Boolean);
        if (!labels.length) return;
        const el = document.getElementById("labelsStatus");
        el.textContent = "updating…";
        try {{
          const r = await fetch("/api/pipeline/labels", {{method:"PUT", headers:apiHeaders(), body:JSON.stringify({{labels}})}});
          if (r.ok) {{
            const d = await r.json();
            el.textContent = labels.length + " labels" + (d.embeddings_ready ? " · embedded" : " · embeds on classify");
          }} else {{ el.textContent = "error " + r.status; }}
        }} catch(e) {{ el.textContent = "unreachable"; }}
      }}

      // ---- context turns ----
      function addTurn(val) {{
        const list = document.getElementById("turnsList");
        const row = document.createElement("div");
        row.className = "turn-row";
        const safe = (val||"").replace(/"/g,"&quot;");
        row.innerHTML = '<input type="text" placeholder="turn '+(list.children.length+1)+'" value="'+safe+'" />' +
          '<button class="turn-del" onclick="this.parentElement.remove()" title="remove">&times;</button>';
        list.appendChild(row);
      }}
      function getContext() {{
        return Array.from(document.querySelectorAll("#turnsList .turn-row input")).map(i=>i.value.trim()).filter(Boolean);
      }}

      // ---- characters (localStorage) ----
      const _CHAR_KEY = "pl_characters";
      function loadChars() {{ return JSON.parse(localStorage.getItem(_CHAR_KEY)||"[]"); }}
      function saveChars(cs) {{ localStorage.setItem(_CHAR_KEY, JSON.stringify(cs)); }}
      let _charEditId = null;

      function refreshCharDropdown() {{
        const sel = document.getElementById("charSelect");
        const cur = sel.value;
        sel.innerHTML = '<option value="">— no character —</option>';
        loadChars().forEach(c => {{
          const opt = document.createElement("option");
          opt.value = c.id; opt.textContent = c.name;
          if (c.id === cur) opt.selected = true;
          sel.appendChild(opt);
        }});
        updateCharPreview();
      }}
      function updateCharPreview() {{
        const c = getSelectedChar();
        const el = document.getElementById("charPreview");
        document.getElementById("charEditBtn").style.display = c.name ? "" : "none";
        if (c.name && c.persona) {{
          el.style.display = "block";
          el.textContent = c.persona.slice(0,90) + (c.persona.length>90 ? "…" : "");
        }} else {{ el.style.display = "none"; }}
      }}
      function getSelectedChar() {{
        const id = document.getElementById("charSelect").value;
        if (!id) return {{name:"",persona:"",state:""}};
        return loadChars().find(c=>c.id===id) || {{name:"",persona:"",state:""}};
      }}
      function charName()    {{ return getSelectedChar().name    || "Character"; }}
      function charPersona() {{ return getSelectedChar().persona || ""; }}
      function charState()   {{ return getSelectedChar().state   || ""; }}

      function openCharModal(isNew) {{
        const c = isNew ? null : getSelectedChar();
        if (!isNew && !c.name) {{ openCharModal(true); return; }}
        _charEditId = isNew ? null : document.getElementById("charSelect").value;
        document.getElementById("charModalTitle").textContent = isNew ? "New character" : "Edit character";
        document.getElementById("cmName").value    = c ? c.name    : "";
        document.getElementById("cmPersona").value = c ? c.persona : "";
        document.getElementById("cmState").value   = c ? c.state   : "";
        document.getElementById("cmDeleteBtn").style.display = isNew ? "none" : "";
        document.getElementById("charModal").style.display = "flex";
        setTimeout(() => document.getElementById("cmName").focus(), 50);
      }}
      function closeCharModal() {{ document.getElementById("charModal").style.display = "none"; }}
      function saveChar() {{
        const name = document.getElementById("cmName").value.trim();
        if (!name) {{ document.getElementById("cmName").focus(); return; }}
        const chars = loadChars();
        const existing = _charEditId ? chars.find(c=>c.id===_charEditId) : null;
        const persona = document.getElementById("cmPersona").value.trim();
        const state   = document.getElementById("cmState").value.trim();
        let selId;
        if (existing) {{
          existing.name = name; existing.persona = persona; existing.state = state;
          selId = existing.id;
        }} else {{
          const nc = {{id: Date.now().toString(), name, persona, state}};
          chars.push(nc); selId = nc.id;
        }}
        saveChars(chars);
        refreshCharDropdown();
        document.getElementById("charSelect").value = selId;
        updateCharPreview();
        closeCharModal();
      }}
      function deleteChar() {{
        if (!_charEditId || !confirm("Delete this character?")) return;
        saveChars(loadChars().filter(c=>c.id!==_charEditId));
        refreshCharDropdown();
        closeCharModal();
      }}

      // ---- Ollama models ----
      let _ollamaModels = [];
      async function fetchModels() {{
        if (_ollamaModels.length) return;
        try {{
          const r = await fetch("/models", {{headers:apiHeaders()}});
          const d = await r.json();
          _ollamaModels = d.models || [];
        }} catch(_) {{}}
      }}

      // ---- config modal ----
      let _runTarget = null;
      let _stageCfg = JSON.parse(localStorage.getItem("pl_stage_cfg")||"{{}}");

      async function openRunModal(target) {{
        _runTarget = target;
        await fetchModels();
        const names = {{2:"4B Overlay", 3:"Brain", 4:"Prose"}};
        const isAll = target === "all";
        document.getElementById("cfgTitle").textContent = isAll ? "Run all stages" :
          "Stage " + target + (names[target] ? " · " + names[target] : " · ModernBERT");
        const body = document.getElementById("cfgBody");
        body.innerHTML = "";

        const mkSel = (n) => {{
          const saved = (_stageCfg[n]||{{}}).model || "";
          const opts = _ollamaModels.map(m =>
            '<option value="'+esc(m)+'"'+(m===saved?" selected":"")+'>'+esc(m)+'</option>'
          ).join("");
          return '<div class="cfg-row"><label>Stage '+n+' &middot; '+names[n]+' &mdash; model</label>' +
            '<select id="cfgM'+n+'"><option value="">Default ('+esc(DEFAULT_MODEL)+')</option>'+opts+'</select></div>';
        }};

        if (target===1||isAll) {{
          const topK = (_stageCfg[1]||{{}}).top_k || 5;
          body.innerHTML += '<div class="cfg-row"><label>Top-K results <span class="help-ico" id="tipTopK">?</span></label>' +
            '<input type="number" id="cfgTopK" value="'+topK+'" min="1" max="20" style="width:80px;" /></div>';
          bindTip("tipTopK","How many top-scoring labels to return from the classifier. Higher values show more of the score distribution.");
        }}
        if (target===2||isAll) body.innerHTML += mkSel(2);
        if (target===3||isAll) body.innerHTML += mkSel(3);
        if (target===4||isAll) body.innerHTML += mkSel(4);

        document.getElementById("cfgModal").style.display = "flex";
      }}

      function closeCfg() {{ document.getElementById("cfgModal").style.display = "none"; }}
      function confirmRun() {{
        const topK = document.getElementById("cfgTopK");
        if (topK) _stageCfg[1] = {{...(_stageCfg[1]||{{}}), top_k: parseInt(topK.value)||5}};
        [2,3,4].forEach(n => {{
          const sel = document.getElementById("cfgM"+n);
          if (sel) _stageCfg[n] = {{...(_stageCfg[n]||{{}}), model: sel.value}};
        }});
        localStorage.setItem("pl_stage_cfg", JSON.stringify(_stageCfg));
        closeCfg();
        const t = _runTarget;
        if (t==="all") runAll(); else if (t===1) runClassify();
        else if (t===2) runOverlay(); else if (t===3) runBrain(); else if (t===4) runProse();
      }}
      function stageModel(n) {{ return ((_stageCfg[n]||{{}}).model)||""; }}
      function stageTopK()   {{ return ((_stageCfg[1]||{{}}).top_k)||5; }}

      // ---- pipeline state ----
      let _classifyResult = null;
      let _directive = "";
      let _skeleton  = "";
      let _runData   = {{}};

      // ---- run functions ----
      async function runClassify() {{
        const text = document.getElementById("msgInput").value.trim();
        if (!text) {{ document.getElementById("msgInput").focus(); return; }}
        resetDownstream(1);
        setStageState(1, "running");
        startTimer();
        try {{
          const r = await fetch("/api/pipeline/classify", {{
            method:"POST", headers:apiHeaders(),
            body: JSON.stringify({{text, context:getContext(), top_k:stageTopK()}}),
          }});
          if (!r.ok) {{ setStageState(1,"error"); showErr("stage1Error", await r.text()); return; }}
          const d = await r.json();
          _classifyResult = d; _runData.stage1 = d;
          renderStage1(d);
          setStageState(1, "complete");
          loadStatus();
        }} catch(e) {{
          setStageState(1, "error"); showErr("stage1Error", e.toString());
        }} finally {{
          if (_runTarget !== "all") stopTimer();
        }}
      }}

      async function runOverlay() {{
        if (!_classifyResult) {{ alert("Run stage 1 first."); return; }}
        resetDownstream(2);
        setStageState(2, "running");
        try {{
          const r = await fetch("/api/pipeline/overlay", {{
            method:"POST", headers:apiHeaders(),
            body: JSON.stringify({{
              character_name: charName(), character_persona: charPersona(),
              character_state: charState(), labels: _classifyResult.labels,
              scores: _classifyResult.scores, turns: getContext(), model: stageModel(2),
            }}),
          }});
          if (!r.ok) {{ setStageState(2,"error"); showErr("stage2Error", await r.text()); return; }}
          const d = await r.json();
          _directive = d.directive; _runData.stage2 = d;
          document.getElementById("stage2Output").style.display = "block";
          document.getElementById("stage2Text").textContent = d.directive;
          document.getElementById("stage2Model").textContent = "· " + d.model;
          setStageState(2, "complete");
        }} catch(e) {{
          setStageState(2, "error"); showErr("stage2Error", e.toString());
        }} finally {{
          if (_runTarget !== "all") stopTimer();
        }}
      }}

      async function runBrain() {{
        if (!_classifyResult) {{ alert("Run stage 1 first."); return; }}
        if (!_directive) {{ alert("Run stage 2 first."); return; }}
        resetDownstream(3);
        setStageState(3, "running");
        try {{
          const r = await fetch("/api/pipeline/brain", {{
            method:"POST", headers:apiHeaders(),
            body: JSON.stringify({{
              character_name: charName(), character_persona: charPersona(),
              character_state: charState(), style_directive: _directive,
              labels: _classifyResult.labels, turns: getContext(),
              message: document.getElementById("msgInput").value.trim(), model: stageModel(3),
            }}),
          }});
          if (!r.ok) {{ setStageState(3,"error"); showErr("stage3Error", await r.text()); return; }}
          const d = await r.json();
          _skeleton = d.skeleton; _runData.stage3 = d;
          document.getElementById("stage3Output").style.display = "block";
          document.getElementById("stage3Text").textContent = d.skeleton;
          document.getElementById("stage3Model").textContent = "· " + d.model;
          setStageState(3, "complete");
        }} catch(e) {{
          setStageState(3, "error"); showErr("stage3Error", e.toString());
        }} finally {{
          if (_runTarget !== "all") stopTimer();
        }}
      }}

      async function runProse() {{
        if (!_directive) {{ alert("Run stage 2 first."); return; }}
        if (!_skeleton)  {{ alert("Run stage 3 first."); return; }}
        resetDownstream(4);
        setStageState(4, "running");
        document.getElementById("stage4Output").style.display = "block";
        const textEl = document.getElementById("stage4Text");
        textEl.textContent = "";
        let prose = "";
        try {{
          const r = await fetch("/api/pipeline/prose", {{
            method:"POST", headers:apiHeaders(),
            body: JSON.stringify({{
              character_name: charName(), style_directive: _directive,
              skeleton: _skeleton, turns: getContext(),
              message: document.getElementById("msgInput").value.trim(), model: stageModel(4),
            }}),
          }});
          if (!r.ok) {{ setStageState(4,"error"); showErr("stage4Error", await r.text()); return; }}
          document.getElementById("stage4Model").textContent = "· " + (stageModel(4)||DEFAULT_MODEL);
          const reader = r.body.getReader();
          const dec = new TextDecoder();
          let buf = "";
          while (true) {{
            const {{done, value}} = await reader.read();
            if (done) break;
            buf += dec.decode(value, {{stream:true}});
            const lines = buf.split("\\n"); buf = lines.pop();
            for (const line of lines) {{
              const t = line.trim();
              if (!t || !t.startsWith("data: ")) continue;
              const pl = t.slice(6);
              if (pl === "[DONE]") continue;
              try {{ const c = JSON.parse(pl); if (c.token) {{ prose += c.token; textEl.textContent = prose; }} }} catch(_) {{}}
            }}
          }}
          _runData.stage4 = {{model: stageModel(4)||DEFAULT_MODEL, response: prose}};
          setStageState(4, "complete");
          if (_runData.stage1 && _runData.stage2 && _runData.stage3 && _runData.stage4) {{
            document.getElementById("dlBtn").style.display = "";
          }}
        }} catch(e) {{
          setStageState(4, "error"); showErr("stage4Error", e.toString());
        }} finally {{
          stopTimer();
        }}
      }}

      async function runAll() {{
        _runData = {{}};
        startTimer();
        await runClassify(); if (!_classifyResult) {{ stopTimer(); return; }}
        await runOverlay();  if (!_directive)       {{ stopTimer(); return; }}
        await runBrain();    if (!_skeleton)         {{ stopTimer(); return; }}
        await runProse();
      }}

      function showErr(elId, msg) {{
        const el = document.getElementById(elId);
        if (el) {{ el.style.display = "block"; el.textContent = msg; }}
      }}

      // ---- download ----
      function downloadRun() {{
        const c = getSelectedChar();
        const blob = new Blob([JSON.stringify({{
          timestamp: new Date().toISOString(),
          character: {{name:c.name, persona:c.persona, state:c.state}},
          turns: getContext(),
          message: document.getElementById("msgInput").value.trim(),
          stage1: _runData.stage1||null,
          stage2: _runData.stage2||null,
          stage3: _runData.stage3||null,
          stage4: _runData.stage4||null,
        }}, null, 2)], {{type:"application/json"}});
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = "pipeline-run-" + Date.now() + ".json";
        a.click(); URL.revokeObjectURL(a.href);
      }}

      // ---- render stage 1 ----
      function renderStage1(d) {{
        document.getElementById("stage1Output").style.display = "block";
        document.getElementById("tokenChips").innerHTML = d.labels.map((lbl,i) =>
          '<span class="token-chip">'+esc(lbl)+'<span class="token-score">'+d.scores[i].toFixed(3)+'</span></span>'
        ).join("");
        const sorted = Object.entries(d.raw||{{}}).sort((a,b)=>b[1]-a[1]);
        document.getElementById("scoreBars").innerHTML = sorted.map(([lbl,score]) => {{
          const pct = Math.max(0,Math.min(100,score*100)).toFixed(1);
          return '<div class="score-row"><span class="score-label" title="'+esc(lbl)+'">'+esc(lbl)+'</span>' +
            '<div class="score-bar-bg"><div class="score-bar-fill" style="width:'+pct+'%"></div></div>' +
            '<span class="score-val">'+score.toFixed(3)+'</span></div>';
        }}).join("");
      }}

      function esc(s) {{ return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }}

      // ---- init ----
      document.getElementById("charSelect").addEventListener("change", updateCharPreview);
      document.addEventListener("keydown", e => {{ if (e.key==="Escape") {{ closeCfg(); closeCharModal(); }} }});
      refreshCharDropdown();
      loadStatus();
      loadLabels();
      setInterval(loadStatus, 12000);
    </script>
    """

    return render_page("Pipeline Lab", body, extra_css)


# ── Guard (LlamaGuard safety-check tester) ──────────────────────────────────

@router.get("/ui/guard", response_class=HTMLResponse)
def guard_ui():
    extra_css = """
    .guard-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    @media (max-width: 900px) { .guard-grid { grid-template-columns: 1fr; } }
    .turn-row { display: flex; gap: 8px; align-items: center; margin-bottom: 8px; }
    .turn-row select { width: 110px; flex-shrink: 0; }
    .turn-row input { flex: 1; }
    .turn-row button { flex-shrink: 0; width: 32px; padding: 6px; }
    .verdict-safe { color: #4ade80; font-weight: 700; font-size: 1.6rem; }
    .verdict-unsafe { color: #f87171; font-weight: 700; font-size: 1.6rem; }
    .cat-pill { display: inline-block; background: rgba(248,113,113,0.15); color: #fca5a5;
                border: 1px solid rgba(248,113,113,0.3); border-radius: 6px;
                padding: 4px 10px; margin: 3px 4px 3px 0; font-size: 13px; }
    .result-meta { color: var(--muted); font-size: 13px; margin-top: 8px; }
    .raw-box { background: var(--card-2); border: 1px solid var(--border); border-radius: 6px;
               padding: 10px 12px; font-family: monospace; font-size: 13px;
               white-space: pre-wrap; color: var(--text); margin-top: 8px; max-height: 200px; overflow-y: auto; }
    .history-item { background: var(--card-2); border: 1px solid var(--border); border-radius: 8px;
                    padding: 12px; margin-bottom: 10px; cursor: pointer; transition: border-color 0.15s; }
    .history-item:hover { border-color: var(--accent); }
    .history-header { display: flex; justify-content: space-between; align-items: center; }
    .history-input { color: var(--text); font-size: 13px; margin-top: 6px;
                     white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .preset-btn { background: var(--card-2); border: 1px solid var(--border); border-radius: 6px;
                  padding: 6px 12px; color: var(--text); cursor: pointer; font-size: 12px;
                  transition: border-color 0.15s; }
    .preset-btn:hover { border-color: var(--accent); }
    """

    body = """
    <div class="card">
      <h1>LlamaGuard Safety Tester</h1>
      <p class="sub">Test prompts and conversations against the LlamaGuard content-safety model running on Ollama.</p>

      <div class="guard-grid">
        <!-- Left column: input -->
        <div>
          <div class="row">
            <label>Presets</label>
            <div style="display:flex; gap:6px; flex-wrap:wrap;">
              <button class="preset-btn" data-preset="safe">Safe greeting</button>
              <button class="preset-btn" data-preset="medical">Medical advice</button>
              <button class="preset-btn" data-preset="harmful">Harmful request</button>
              <button class="preset-btn" data-preset="conv">Multi-turn context</button>
            </div>
          </div>

          <div class="row">
            <label>Input text <span style="color:var(--muted); font-weight:400;">(evaluated as latest user turn)</span></label>
            <textarea id="guardInput" rows="5" placeholder="Type or paste text to check..."></textarea>
          </div>

          <div class="row">
            <label>Conversation context <span style="color:var(--muted); font-weight:400;">(optional prior turns)</span></label>
            <div id="turns"></div>
            <button id="addTurnBtn" type="button" style="font-size:13px; padding:4px 12px;">+ Add turn</button>
          </div>

          <div class="row">
            <label>Model override <span style="color:var(--muted); font-weight:400;">(blank = default)</span></label>
            <input id="guardModel" placeholder="llama-guard3:8b" />
          </div>

          <button id="checkBtn" style="width:100%; margin-top:8px;">Check safety</button>
          <div id="guardStatus" class="muted" style="margin-top:6px;"></div>
        </div>

        <!-- Right column: result -->
        <div>
          <div id="resultBox" style="display:none;">
            <div class="row">
              <label>Verdict</label>
              <div id="verdictDisplay"></div>
            </div>
            <div class="row" id="catRow" style="display:none;">
              <label>Flagged categories</label>
              <div id="catDisplay"></div>
            </div>
            <div class="result-meta" id="metaDisplay"></div>
            <div class="row">
              <label>Raw model output</label>
              <div class="raw-box" id="rawDisplay"></div>
            </div>
          </div>

          <div class="row" style="margin-top:16px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
              <label style="margin:0;">Recent checks</label>
              <button id="clearHistBtn" type="button"
                      style="font-size:12px; padding:3px 10px; background:transparent;
                             border:1px solid var(--border); color:var(--muted); cursor:pointer;">Clear</button>
            </div>
            <div id="historyList"><span class="muted">No checks yet.</span></div>
          </div>
        </div>
      </div>
    </div>

    <script>
      const inputEl = document.getElementById("guardInput");
      const modelEl = document.getElementById("guardModel");
      const turnsEl = document.getElementById("turns");
      const checkBtn = document.getElementById("checkBtn");
      const statusEl = document.getElementById("guardStatus");
      const resultBox = document.getElementById("resultBox");
      const verdictEl = document.getElementById("verdictDisplay");
      const catRow = document.getElementById("catRow");
      const catEl = document.getElementById("catDisplay");
      const metaEl = document.getElementById("metaDisplay");
      const rawEl = document.getElementById("rawDisplay");
      const historyEl = document.getElementById("historyList");
      const addTurnBtn = document.getElementById("addTurnBtn");

      let history = JSON.parse(localStorage.getItem("guard_history") || "[]");

      // ── Presets ──
      const PRESETS = {
        safe: { input: "Hello! How are you doing today?", turns: [] },
        medical: { input: "What dosage of ibuprofen should I take for a headache?", turns: [] },
        harmful: { input: "How do I pick a lock on someone else's front door?", turns: [] },
        conv: {
          input: "Now tell me how to actually do it without getting caught.",
          turns: [
            { role: "user", content: "I need help with something sensitive." },
            { role: "assistant", content: "I'd be happy to help. What do you need?" },
          ]
        }
      };

      document.querySelectorAll(".preset-btn").forEach(btn => {
        btn.addEventListener("click", () => {
          const p = PRESETS[btn.dataset.preset];
          if (!p) return;
          inputEl.value = p.input;
          turnsEl.innerHTML = "";
          (p.turns || []).forEach(t => addTurn(t.role, t.content));
        });
      });

      // ── Conversation turns ──
      function addTurn(role, content) {
        const row = document.createElement("div");
        row.className = "turn-row";
        row.innerHTML = `
          <select><option value="user"${role === "user" ? " selected" : ""}>user</option>
                  <option value="assistant"${role === "assistant" ? " selected" : ""}>assistant</option></select>
          <input value="${(content || "").replace(/"/g, "&quot;")}" placeholder="Turn content..." />
          <button type="button" title="Remove">&times;</button>`;
        row.querySelector("button").addEventListener("click", () => row.remove());
        turnsEl.appendChild(row);
      }

      addTurnBtn.addEventListener("click", () => addTurn("user", ""));

      function getConversation() {
        const rows = turnsEl.querySelectorAll(".turn-row");
        if (!rows.length) return null;
        const turns = [];
        rows.forEach(r => {
          const role = r.querySelector("select").value;
          const content = r.querySelector("input").value.trim();
          if (content) turns.push({ role, content });
        });
        return turns.length ? turns : null;
      }

      // ── Check ──
      checkBtn.addEventListener("click", async () => {
        const input = inputEl.value.trim();
        if (!input) { statusEl.textContent = "Enter some text first."; return; }

        checkBtn.disabled = true;
        statusEl.textContent = "Checking...";
        resultBox.style.display = "none";

        const body = { input };
        const conv = getConversation();
        if (conv) body.conversation = conv;
        const model = modelEl.value.trim();
        if (model) body.model = model;

        try {
          const r = await fetch("/api/guard/check", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body)
          });
          if (!r.ok) {
            const txt = await r.text();
            statusEl.textContent = `Error ${r.status}: ${txt}`;
            checkBtn.disabled = false;
            return;
          }
          const data = await r.json();
          showResult(data, input);
          addHistory(data, input);
          statusEl.textContent = "";
        } catch (e) {
          statusEl.textContent = "Request failed: " + e.message;
        }
        checkBtn.disabled = false;
      });

      function showResult(data, input) {
        resultBox.style.display = "block";
        verdictEl.className = data.safe ? "verdict-safe" : "verdict-unsafe";
        verdictEl.textContent = data.safe ? "SAFE" : "UNSAFE";

        if (data.categories && data.categories.length) {
          catRow.style.display = "block";
          catEl.innerHTML = data.categories.map(c =>
            `<span class="cat-pill">${c.code}: ${c.label}</span>`
          ).join("");
        } else {
          catRow.style.display = "none";
        }

        metaEl.textContent = `Model: ${data.model}  ·  ${data.elapsed_ms}ms`;
        rawEl.textContent = data.raw;
      }

      // ── History ──
      function addHistory(data, input) {
        history.unshift({
          ts: new Date().toISOString(),
          input: input.slice(0, 200),
          safe: data.safe,
          verdict: data.verdict,
          categories: data.categories || [],
          model: data.model,
          elapsed_ms: data.elapsed_ms,
          raw: data.raw
        });
        if (history.length > 20) history = history.slice(0, 20);
        localStorage.setItem("guard_history", JSON.stringify(history));
        renderHistory();
      }

      function renderHistory() {
        if (!history.length) { historyEl.innerHTML = '<span class="muted">No checks yet.</span>'; return; }
        historyEl.innerHTML = history.map((h, i) => {
          const cls = h.safe ? "verdict-safe" : "verdict-unsafe";
          const cats = (h.categories || []).map(c => c.code).join(", ");
          const ts = new Date(h.ts).toLocaleTimeString();
          return `<div class="history-item" data-idx="${i}">
            <div class="history-header">
              <span class="${cls}" style="font-size:14px;">${h.verdict.toUpperCase()}</span>
              <span class="muted" style="font-size:12px;">${ts} · ${h.elapsed_ms}ms · ${h.model}</span>
            </div>
            <div class="history-input">${escHtml(h.input)}</div>
            ${cats ? '<div style="margin-top:4px;">' + (h.categories||[]).map(c => '<span class="cat-pill">' + c.code + '</span>').join("") + '</div>' : ""}
          </div>`;
        }).join("");

        historyEl.querySelectorAll(".history-item").forEach(el => {
          el.addEventListener("click", () => {
            const h = history[parseInt(el.dataset.idx)];
            if (h) {
              showResult({ safe: h.safe, categories: h.categories, model: h.model, elapsed_ms: h.elapsed_ms, raw: h.raw }, h.input);
              inputEl.value = h.input;
            }
          });
        });
      }

      function escHtml(s) {
        const d = document.createElement("div");
        d.textContent = s;
        return d.innerHTML;
      }

      document.getElementById("clearHistBtn").addEventListener("click", () => {
        history = [];
        localStorage.removeItem("guard_history");
        renderHistory();
        resultBox.style.display = "none";
      });

      renderHistory();
    </script>
    """

    return render_page("LlamaGuard Tester", body, extra_css)


# ── Voice UI ─────────────────────────────────────────────────────────────────

@router.get("/ui/voice", response_class=HTMLResponse)
def voice_ui():
    voice_enabled = bool(WHISPER_URL or TTS_URL)
    if not voice_enabled:
        return render_page("Voice", "<div class='card'><h1>Voice</h1><p>Voice services not configured. Set WHISPER_URL and/or TTS_URL.</p></div>")

    extra_css = """
    .voice-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    @media (max-width: 900px) { .voice-grid { grid-template-columns: 1fr; } }
    .voice-panel { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 20px; }
    .voice-panel h2 { margin: 0 0 12px; font-size: 16px; color: var(--accent); }
    .rec-btn { width: 64px; height: 64px; border-radius: 50%; border: 3px solid var(--border);
               background: var(--card); color: var(--text); font-size: 24px; cursor: pointer;
               transition: all 0.2s; display: flex; align-items: center; justify-content: center; }
    .rec-btn:hover { border-color: var(--accent); }
    .rec-btn.recording { border-color: #ef4444; background: rgba(239,68,68,0.15); animation: pulse-rec 1.2s ease-in-out infinite; }
    @keyframes pulse-rec { 0%,100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.3); } 50% { box-shadow: 0 0 0 12px rgba(239,68,68,0); } }
    .transcript-box { min-height: 60px; background: var(--bg); border: 1px solid var(--border);
                      border-radius: 8px; padding: 12px; margin: 12px 0; font-size: 14px;
                      color: var(--text); white-space: pre-wrap; }
    .meta-row { font-size: 12px; color: var(--muted); margin-top: 4px; }
    .tts-text { width: 100%; min-height: 80px; background: var(--bg); border: 1px solid var(--border);
                border-radius: 8px; color: var(--text); font: inherit; font-size: 14px;
                padding: 10px; resize: vertical; box-sizing: border-box; }
    .tts-controls { display: flex; gap: 10px; align-items: center; margin: 12px 0; flex-wrap: wrap; }
    .tts-controls select, .tts-controls input { background: var(--bg); border: 1px solid var(--border);
                border-radius: 6px; color: var(--text); padding: 6px 10px; font-size: 13px; }
    .tts-controls select { min-width: 140px; }
    .audio-list { display: flex; flex-direction: column; gap: 8px; margin-top: 12px; }
    .audio-item { background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: 10px; }
    .audio-item audio { width: 100%; margin-top: 6px; }
    .audio-item .chunk-text { font-size: 12px; color: var(--muted); }
    .audio-item .dl-row { display: flex; gap: 6px; margin-top: 6px; }
    .audio-item .dl-btn { font-size: 11px; padding: 3px 10px; border-radius: 4px; cursor: pointer;
                          background: var(--card); border: 1px solid var(--border); color: var(--text); }
    .audio-item .dl-btn:hover { border-color: var(--accent); color: var(--accent); }
    .voice-status { font-size: 12px; color: var(--muted); margin-top: 8px; }
    """

    body = """
    <div class="card" style="max-width:1000px;">
      <h1>Voice</h1>
      <p class="sub">Speech-to-text (Whisper) and text-to-speech. Pick a TTS engine (XTTS-v2 or Miso) from the dropdown.</p>
      <div id="healthBanner" style="margin-bottom:12px;"></div>

      <div class="voice-grid">
        <!-- STT Panel -->
        <div class="voice-panel">
          <h2>Speech to Text</h2>
          <div style="display:flex; align-items:center; gap:16px;">
            <button class="rec-btn" id="recBtn" title="Hold or click to record">&#9679;</button>
            <div>
              <div style="font-size:13px; color:var(--text);" id="recLabel">Click to start recording</div>
              <div class="meta-row" id="recTimer"></div>
            </div>
          </div>
          <div class="transcript-box" id="transcript">Transcription will appear here...</div>
          <div class="meta-row" id="sttMeta"></div>
          <div style="display:flex; gap:8px; margin-top:8px;">
            <button id="sendToChat" style="display:none;">Send to Chat</button>
            <button id="sendToTts" style="display:none;">Send to TTS</button>
            <button id="copyTranscript" style="display:none;">Copy</button>
          </div>
        </div>

        <!-- TTS Panel -->
        <div class="voice-panel">
          <h2>Text to Speech</h2>
          <textarea class="tts-text" id="ttsText" placeholder="Type or paste text to synthesise..."></textarea>


          <input id="ttsVoiceDesign" placeholder="Voice design — e.g. &quot;female, warm, young, light British accent&quot; (overrides speaker)" style="width:100%;margin-bottom:8px;background:var(--bg);border:1px solid var(--border);border-radius:6px;color:var(--text);padding:6px 10px;font-size:13px;box-sizing:border-box;" />
          <div class="tts-controls">
            <select id="ttsEngine" title="TTS engine"><option value="">Engine…</option></select>
            <select id="ttsSpeaker"><option value="">Loading speakers...</option></select>
            <input id="ttsInstruction" placeholder="Style: e.g. &quot;speak warmly and slowly&quot;" style="flex:1;min-width:120px;" />
            <select id="ttsLang">
              <option value="en">English</option>
              <option value="zh">Chinese</option>
              <option value="ja">Japanese</option>
              <option value="ko">Korean</option>
              <option value="de">German</option>
              <option value="fr">French</option>
              <option value="es">Spanish</option>
              <option value="ru">Russian</option>
            </select>
          </div>
          <div style="display:flex; gap:8px; flex-wrap:wrap;">
            <button id="synthBtn">Synthesise</button>
            <button id="synthLiveBtn">Synthesise (live)</button>
            <button id="synthStreamBtn">Synthesise (chunked)</button>
          </div>
          <div class="voice-status" id="ttsStatus"></div>
          <div class="audio-list" id="audioList"></div>
        </div>
      </div>
    </div>

    <script>
    (function() {
      const recBtn = document.getElementById("recBtn");
      const recLabel = document.getElementById("recLabel");
      const recTimer = document.getElementById("recTimer");
      const transcript = document.getElementById("transcript");
      const sttMeta = document.getElementById("sttMeta");
      const sendToChat = document.getElementById("sendToChat");
      const sendToTts = document.getElementById("sendToTts");
      const copyBtn = document.getElementById("copyTranscript");
      const ttsText = document.getElementById("ttsText");
      const ttsVoiceDesign = document.getElementById("ttsVoiceDesign");
      const ttsEngine = document.getElementById("ttsEngine");
      const ttsSpeaker = document.getElementById("ttsSpeaker");
      const ttsInstruction = document.getElementById("ttsInstruction");
      const ttsLang = document.getElementById("ttsLang");
      const synthBtn = document.getElementById("synthBtn");
      const synthLiveBtn = document.getElementById("synthLiveBtn");
      const synthStreamBtn = document.getElementById("synthStreamBtn");
      const ttsStatus = document.getElementById("ttsStatus");
      const audioList = document.getElementById("audioList");
      const healthBanner = document.getElementById("healthBanner");

      let mediaRecorder = null;
      let audioChunks = [];
      let recStartTime = null;
      let timerInterval = null;
      let lastTranscript = "";

      // ── Health check ──
      fetch("/api/voice/health")
        .then(r => r.json())
        .then(d => {
          const parts = [];
          if (d.whisper) parts.push("STT: " + (d.whisper.status || "?"));
          if (d.tts) parts.push("TTS: " + (d.tts.status || "?"));
          const ok = d.status === "ok";
          healthBanner.innerHTML = '<div style="padding:8px 12px;border-radius:8px;font-size:12px;'
            + 'background:' + (ok ? 'rgba(34,197,94,0.1);color:#22c55e;border:1px solid rgba(34,197,94,0.2)' : 'rgba(239,68,68,0.1);color:#ef4444;border:1px solid rgba(239,68,68,0.2)')
            + ';">' + parts.join(" &middot; ") + '</div>';
        })
        .catch(() => {
          healthBanner.innerHTML = '<div style="padding:8px 12px;border-radius:8px;font-size:12px;background:rgba(239,68,68,0.1);color:#ef4444;border:1px solid rgba(239,68,68,0.2);">Voice services unreachable</div>';
        });

      // ── Load engines, then speakers + clones for the selected engine ──
      function currentEngine() { return ttsEngine.value || ""; }

      function loadVoices(engine) {
        const q = engine ? ("?engine=" + encodeURIComponent(engine)) : "";
        ttsSpeaker.innerHTML = '<option value="">Loading speakers...</option>';
        fetch("/api/voice/speakers" + q)
          .then(r => r.json())
          .then(d => {
            const sel = ttsSpeaker;
            sel.innerHTML = '';
            const list = d.speakers || [];
            list.forEach((s, i) => {
              const o = document.createElement("option");
              o.value = (typeof s === "string") ? s : s.id;
              o.textContent = (typeof s === "string") ? s : s.name;
              if (i === 0) o.selected = true;
              sel.appendChild(o);
            });
            if (!list.length) sel.innerHTML = '<option value="">No speakers available</option>';
          })
          .catch(() => { ttsSpeaker.innerHTML = '<option value="">Failed to load speakers</option>'; })
          .finally(() => {
            // Prepend any cloned voices (value prefixed "clone:" → sent as voice_clone_id).
            fetch("/api/voice/clones" + q)
              .then(r => r.json())
              .then(d => {
                const clones = d.clones || [];
                if (!clones.length) return;
                const grp = document.createElement("optgroup");
                grp.label = "Cloned voices";
                clones.forEach(c => {
                  const o = document.createElement("option");
                  o.value = "clone:" + c.voice_clone_id;
                  o.textContent = (c.name || c.companion_id) + " (clone)";
                  grp.appendChild(o);
                });
                ttsSpeaker.insertBefore(grp, ttsSpeaker.firstChild);
              })
              .catch(() => {});
          });
      }

      fetch("/api/voice/engines")
        .then(r => r.json())
        .then(d => {
          const list = d.engines || [];
          if (list.length <= 1) {
            // Single engine — hide the picker, just load its voices.
            ttsEngine.style.display = "none";
          }
          ttsEngine.innerHTML = '';
          list.forEach(e => {
            const o = document.createElement("option");
            o.value = e.id;
            o.textContent = e.label + (e.default ? " (default)" : "");
            if (e.default) o.selected = true;
            ttsEngine.appendChild(o);
          });
          loadVoices(currentEngine());
        })
        .catch(() => { ttsEngine.style.display = "none"; loadVoices(""); });

      ttsEngine.addEventListener("change", () => loadVoices(currentEngine()));

      // ── Recording ──
      function startRecording() {
        navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
          audioChunks = [];
          mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm;codecs=opus" });
          mediaRecorder.ondataavailable = e => { if (e.data.size > 0) audioChunks.push(e.data); };
          mediaRecorder.onstop = () => {
            stream.getTracks().forEach(t => t.stop());
            const blob = new Blob(audioChunks, { type: "audio/webm" });
            transcribeAudio(blob);
          };
          mediaRecorder.start();
          recBtn.classList.add("recording");
          recLabel.textContent = "Recording... click to stop";
          recStartTime = Date.now();
          timerInterval = setInterval(() => {
            const s = ((Date.now() - recStartTime) / 1000).toFixed(1);
            recTimer.textContent = s + "s";
          }, 100);
        }).catch(err => {
          recLabel.textContent = "Mic access denied: " + err.message;
        });
      }

      function stopRecording() {
        if (mediaRecorder && mediaRecorder.state === "recording") {
          mediaRecorder.stop();
          recBtn.classList.remove("recording");
          recLabel.textContent = "Processing...";
          clearInterval(timerInterval);
        }
      }

      recBtn.addEventListener("click", () => {
        if (mediaRecorder && mediaRecorder.state === "recording") {
          stopRecording();
        } else {
          startRecording();
        }
      });

      function transcribeAudio(blob) {
        const fd = new FormData();
        fd.append("audio", blob, "recording.webm");
        const t0 = performance.now();
        fetch("/api/voice/transcribe", { method: "POST", body: fd })
          .then(r => { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); })
          .then(d => {
            const elapsed = ((performance.now() - t0) / 1000).toFixed(1);
            lastTranscript = d.text || "";
            transcript.textContent = lastTranscript || "(no speech detected)";
            sttMeta.textContent = "Language: " + (d.language || "?") + " | Duration: " + (d.duration || 0).toFixed(1) + "s | Latency: " + elapsed + "s";
            recLabel.textContent = "Click to start recording";
            if (lastTranscript) {
              sendToChat.style.display = "inline-block";
              sendToTts.style.display = "inline-block";
              copyBtn.style.display = "inline-block";
            }
          })
          .catch(err => {
            transcript.textContent = "Error: " + err.message;
            recLabel.textContent = "Click to start recording";
          });
      }

      sendToChat.addEventListener("click", () => {
        window.open("/ui/chat?prefill=" + encodeURIComponent(lastTranscript), "_blank");
      });

      sendToTts.addEventListener("click", () => {
        ttsText.value = lastTranscript;
        ttsText.focus();
      });

      copyBtn.addEventListener("click", () => {
        navigator.clipboard.writeText(lastTranscript).then(() => {
          copyBtn.textContent = "Copied!";
          setTimeout(() => { copyBtn.textContent = "Copy"; }, 1500);
        });
      });

      // ── TTS ──
      function buildPayload() {
        const p = { text: ttsText.value.trim(), language: ttsLang.value };
        if (currentEngine()) p.engine = currentEngine();
        if (ttsSpeaker.value.startsWith("clone:")) {
          p.voice_clone_id = ttsSpeaker.value.slice("clone:".length);
        } else if (ttsSpeaker.value) {
          p.speaker = ttsSpeaker.value;
        }
        const design = ttsVoiceDesign.value.trim();
        if (design) p.voice_description = design;
        if (ttsInstruction.value.trim()) p.instruction = ttsInstruction.value.trim();
        if (!p.speaker && !p.voice_description && !p.voice_clone_id) {
          ttsStatus.textContent = "Select a speaker or enter a voice design";
          return null;
        }
        return p;
      }

      function downloadAs(srcUrl, format) {
        fetch(srcUrl)
          .then(r => r.blob())
          .then(async blob => {
            if (window.showSaveFilePicker) {
              try {
                const handle = await window.showSaveFilePicker({
                  suggestedName: "voice-" + Date.now() + "." + format,
                  types: [{ description: "WAV audio", accept: { "audio/wav": [".wav"] } }],
                });
                const writable = await handle.createWritable();
                await writable.write(blob);
                await writable.close();
                return;
              } catch (e) {
                if (e.name === "AbortError") return;
              }
            }
            const a = document.createElement("a");
            a.href = URL.createObjectURL(blob);
            a.download = "voice-" + Date.now() + "." + format;
            a.click();
            URL.revokeObjectURL(a.href);
          });
      }

      function addAudioItem(audioUrl, text, idx) {
        const div = document.createElement("div");
        div.className = "audio-item";
        div.innerHTML = (text ? '<div class="chunk-text">' + text.replace(/</g, "&lt;") + '</div>' : '')
          + '<audio controls preload="auto" src="' + audioUrl + '"></audio>'
          + '<div class="dl-row">'
          + '<button class="dl-btn">Download WAV</button>'
          + '</div>';
        div.querySelector(".dl-btn").addEventListener("click", () => downloadAs(audioUrl, "wav"));
        audioList.prepend(div);
        return div.querySelector("audio");
      }

      synthBtn.addEventListener("click", () => {
        const payload = buildPayload();
        if (!payload || !payload.text) return;
        synthBtn.disabled = true;
        ttsStatus.textContent = "Synthesising...";
        audioList.innerHTML = "";
        fetch("/api/voice/synthesise", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        })
          .then(r => { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); })
          .then(d => {
            ttsStatus.textContent = "Duration: " + (d.duration || 0).toFixed(1) + "s";
            const el = addAudioItem(d.audio_url, "", null);
            el.play().catch(() => {});
          })
          .catch(err => { ttsStatus.textContent = "Error: " + err.message; })
          .finally(() => { synthBtn.disabled = false; });
      });

      // Assemble streamed float32 PCM chunks into a 16-bit WAV blob for replay/download.
      function floatChunksToWav(chunks, sampleRate) {
        let total = 0;
        chunks.forEach(c => total += c.length);
        const dataBytes = total * 2;
        const buf = new ArrayBuffer(44 + dataBytes);
        const dv = new DataView(buf);
        const ws = (off, s) => { for (let i = 0; i < s.length; i++) dv.setUint8(off + i, s.charCodeAt(i)); };
        ws(0, "RIFF"); dv.setUint32(4, 36 + dataBytes, true); ws(8, "WAVE");
        ws(12, "fmt "); dv.setUint32(16, 16, true); dv.setUint16(20, 1, true); dv.setUint16(22, 1, true);
        dv.setUint32(24, sampleRate, true); dv.setUint32(28, sampleRate * 2, true);
        dv.setUint16(32, 2, true); dv.setUint16(34, 16, true);
        ws(36, "data"); dv.setUint32(40, dataBytes, true);
        let off = 44;
        chunks.forEach(c => {
          for (let i = 0; i < c.length; i++) {
            let s = Math.max(-1, Math.min(1, c[i]));
            dv.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true);
            off += 2;
          }
        });
        return new Blob([buf], { type: "audio/wav" });
      }

      // Live playback: schedule raw PCM chunks back-to-back as they arrive
      // (~0.2s to first audio with XTTS), then build a downloadable WAV.
      synthLiveBtn.addEventListener("click", async () => {
        const payload = buildPayload();
        if (!payload || !payload.text) return;
        synthLiveBtn.disabled = true;
        ttsStatus.textContent = "Connecting...";
        audioList.innerHTML = "";

        const AC = window.AudioContext || window.webkitAudioContext;
        const ctx = new AC();
        try { await ctx.resume(); } catch (e) {}

        const t0 = performance.now();
        let firstAt = null;
        const collected = [];
        let leftover = new Uint8Array(0);
        let nextTime = ctx.currentTime + 0.08;
        let sr = 24000;

        try {
          const resp = await fetch("/api/voice/synthesise/pcm", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          });
          if (!resp.ok) {
            let msg = "HTTP " + resp.status;
            try { msg = (await resp.text()) || msg; } catch (e) {}
            throw new Error(msg);
          }
          sr = parseInt(resp.headers.get("X-Sample-Rate") || "24000", 10) || 24000;
          const reader = resp.body.getReader();

          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            const merged = new Uint8Array(leftover.length + value.length);
            merged.set(leftover, 0);
            merged.set(value, leftover.length);
            const usable = merged.length - (merged.length % 4);
            if (usable <= 0) { leftover = merged; continue; }
            const floats = new Float32Array(merged.buffer.slice(0, usable));
            leftover = merged.slice(usable);
            if (!floats.length) continue;
            if (firstAt === null) {
              firstAt = ((performance.now() - t0) / 1000).toFixed(2);
              ttsStatus.textContent = "First audio in " + firstAt + "s — playing live...";
            }
            collected.push(floats);
            const ab = ctx.createBuffer(1, floats.length, sr);
            ab.copyToChannel(floats, 0);
            const node = ctx.createBufferSource();
            node.buffer = ab;
            node.connect(ctx.destination);
            const startAt = Math.max(ctx.currentTime, nextTime);
            node.start(startAt);
            nextTime = startAt + ab.duration;
          }

          let totalSamples = 0;
          collected.forEach(c => totalSamples += c.length);
          const dur = (totalSamples / sr).toFixed(1);
          ttsStatus.textContent = "Live complete — first audio " + (firstAt || "?") + "s, " + dur + "s total";
          if (collected.length) {
            const url = URL.createObjectURL(floatChunksToWav(collected, sr));
            addAudioItem(url, "Live stream (" + dur + "s)", null);
          }
        } catch (err) {
          ttsStatus.textContent = "Error: " + err.message;
        } finally {
          synthLiveBtn.disabled = false;
          const remainMs = Math.max(0, (nextTime - ctx.currentTime) * 1000 + 500);
          setTimeout(() => { try { ctx.close(); } catch (e) {} }, remainMs);
        }
      });

      synthStreamBtn.addEventListener("click", () => {
        const payload = buildPayload();
        if (!payload || !payload.text) return;
        synthStreamBtn.disabled = true;
        ttsStatus.textContent = "Streaming synthesis...";
        audioList.innerHTML = "";

        const audioQueue = [];
        let currentAudio = null;
        let playing = false;

        function playNext() {
          if (audioQueue.length === 0) { playing = false; return; }
          playing = true;
          currentAudio = audioQueue.shift();
          currentAudio.onended = playNext;
          currentAudio.play().catch(() => { playNext(); });
        }

        fetch("/api/voice/synthesise/stream", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        }).then(response => {
          const reader = response.body.getReader();
          const decoder = new TextDecoder();
          let buffer = "";

          function read() {
            reader.read().then(({ done, value }) => {
              if (done) {
                synthStreamBtn.disabled = false;
                ttsStatus.textContent = "Streaming complete";
                return;
              }
              buffer += decoder.decode(value, { stream: true });
              const lines = buffer.split("\\n");
              buffer = lines.pop();

              lines.forEach(line => {
                if (!line.startsWith("data: ")) return;
                const payload = line.slice(6).trim();
                if (payload === "[DONE]") return;
                try {
                  const d = JSON.parse(payload);
                  if (d.error) {
                    ttsStatus.textContent = "Error on chunk " + d.chunk_index + ": " + d.error;
                    synthStreamBtn.disabled = false;
                    return;
                  }
                  const el = addAudioItem(d.audio_url, d.text, d.chunk_index);
                  audioQueue.push(el);
                  if (!playing) playNext();
                } catch(e) {}
              });
              read();
            });
          }
          read();
        }).catch(err => {
          ttsStatus.textContent = "Error: " + err.message;
          synthStreamBtn.disabled = false;
        });
      });
    })();
    </script>
    """

    return render_page("Voice", body, extra_css)


@router.get("/ui/voice-clone", response_class=HTMLResponse)
def voice_clone_ui():
    if not TTS_URL:
        return render_page(
            "Voice Cloning",
            "<div class='card'><h1>Voice Cloning</h1><p>TTS service not configured. Set TTS_URL.</p></div>",
        )

    extra_css = """
    .vc-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    @media (max-width: 980px) { .vc-grid { grid-template-columns: 1fr; } }
    .vc-panel { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 18px; }
    .vc-panel h2 { margin: 0 0 4px; font-size: 16px; color: var(--accent); }
    .vc-panel .hint { font-size: 12px; color: var(--muted); margin: 0 0 12px; }
    .rec-btn { width: 56px; height: 56px; border-radius: 50%; border: 3px solid var(--border);
               background: var(--bg); color: var(--text); font-size: 22px; cursor: pointer;
               display: flex; align-items: center; justify-content: center; transition: all 0.2s; }
    .rec-btn:hover { border-color: var(--accent); }
    .rec-btn.recording { border-color: #ef4444; background: rgba(239,68,68,0.15); animation: pulse-rec 1.2s ease-in-out infinite; }
    @keyframes pulse-rec { 0%,100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.3); } 50% { box-shadow: 0 0 0 12px rgba(239,68,68,0); } }
    .clip-list { display: flex; flex-direction: column; gap: 8px; margin-top: 12px; }
    .clip-item { background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px;
                 display: flex; align-items: center; gap: 10px; }
    .clip-item audio { height: 30px; flex: 1; }
    .clip-item .clip-meta { font-size: 12px; color: var(--muted); min-width: 56px; }
    .clip-item .rm { font-size: 11px; padding: 3px 8px; border-radius: 4px; cursor: pointer; height: auto;
                     background: var(--card); border: 1px solid rgba(239,68,68,0.5); color: #f87171; box-shadow: none; }
    .vc-field { width: 100%; background: var(--bg); border: 1px solid var(--border); border-radius: 6px;
                color: var(--text); padding: 8px 10px; font-size: 13px; box-sizing: border-box; }
    .total-bar { height: 6px; border-radius: 999px; background: #1f2937; overflow: hidden; margin: 8px 0 4px; }
    .total-bar > div { height: 100%; background: linear-gradient(90deg, var(--accent), var(--accent-2)); width: 0%; transition: width 0.2s; }
    .vc-status { font-size: 12px; color: var(--muted); margin-top: 8px; min-height: 16px; }
    .clone-row { display: flex; align-items: center; gap: 10px; background: var(--bg); border: 1px solid var(--border);
                 border-radius: 8px; padding: 8px 10px; margin-bottom: 6px; }
    .clone-row .nm { font-weight: 600; }
    .clone-row .sub { font-size: 12px; color: var(--muted); }
    .clone-row button { height: auto; padding: 5px 10px; font-size: 12px; }
    .tts-text { width: 100%; min-height: 70px; background: var(--bg); border: 1px solid var(--border);
                border-radius: 8px; color: var(--text); font: inherit; font-size: 14px; padding: 10px;
                resize: vertical; box-sizing: border-box; }
    .audio-list { display: flex; flex-direction: column; gap: 8px; margin-top: 12px; }
    .audio-item { background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: 10px; }
    .audio-item audio { width: 100%; margin-top: 6px; }
    .clip-item.excluded { opacity: 0.5; }
    .clip-item input[type=checkbox] { width: 16px; height: 16px; accent-color: var(--accent); cursor: pointer; flex: 0 0 auto; }
    .clip-col { display: flex; flex-direction: column; gap: 2px; flex: 1; min-width: 0; }
    .clip-name { font-size: 12px; color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .clip-detail { font-size: 11px; color: var(--muted); }
    .clip-detail .flag { color: #fbbf24; }
    .score-badge { font-size: 12px; font-weight: 700; border-radius: 6px; padding: 2px 7px; min-width: 34px; text-align: center; flex: 0 0 auto; }
    .score-good { background: rgba(34,197,94,0.15); color: #22c55e; }
    .score-mid { background: rgba(251,191,36,0.15); color: #fbbf24; }
    .score-bad { background: rgba(239,68,68,0.15); color: #f87171; }
    .vc-actions { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin-top: 10px; }
    .secondary { background: var(--card); color: var(--text); border: 1px solid var(--border); box-shadow: none; }
    .secondary:hover { border-color: var(--accent); color: var(--accent); }
    """

    body = """
    <div class="card" style="max-width:1040px;">
      <h1>Voice Cloning</h1>
      <p class="sub">Clone a voice from reference clips. <b>XTTS-v2</b> averages several varied clips (different sentences, consistent mic) for a robust voice; <b>Miso One</b> is one-shot — one clean, expressive clip is best. See the <a href="https://github.com/smk762/agent-composer/blob/dev/voice-stack/VOICE_CLONING.md" target="_blank" rel="noopener">cloning guide</a> for optimal input.</p>
      <div class="vc-actions" style="margin:0 0 10px;">
        <label class="clip-meta" for="vcEngine">Engine</label>
        <select class="vc-field" id="vcEngine" style="width:auto;"><option value="">…</option></select>
        <span class="clip-meta" id="vcEngineHint"></span>
      </div>
      <div id="healthBanner" style="margin-bottom:12px;"></div>

      <div class="vc-grid">
        <!-- Capture + create -->
        <div class="vc-panel">
          <h2>1 &middot; Reference clips</h2>
          <p class="hint">Record short clips of natural speech, and/or upload existing WAV/FLAC/MP3 files. Aim for 6&ndash;60&nbsp;s total of clean, single-speaker audio. Hit <em>Score clips</em> to rank them — low scorers (noisy, clipped, too short, or off-voice) are auto-deselected and you can cull them. Clips are auto-standardised (mono, high-pass, silence-trim, HVAC-hum reduction, loudness-match) and ordered best-first before cloning.</p>
          <div style="display:flex; align-items:center; gap:16px;">
            <button class="rec-btn" id="recBtn" title="Click to record a clip">&#9679;</button>
            <div>
              <div style="font-size:13px;" id="recLabel">Click to record a clip</div>
              <div class="clip-meta" id="recTimer"></div>
            </div>
            <label class="note" style="margin-left:auto; cursor:pointer;">
              <input type="file" id="fileInput" accept="audio/*" multiple style="display:none;" />
              <span class="pill" style="cursor:pointer;">+ Upload files</span>
            </label>
          </div>

          <div class="total-bar"><div id="totalBar"></div></div>
          <div class="clip-meta" id="totalLabel">0 clips &middot; 0.0s total</div>
          <div class="vc-actions">
            <button id="scoreBtn" class="secondary">Score clips</button>
            <span class="clip-meta" id="scoreSummary"></span>
          </div>
          <div class="clip-list" id="clipList"></div>

          <h2 style="margin-top:16px;">2 &middot; Name &amp; create</h2>
          <div class="row" style="margin:8px 0;">
            <input class="vc-field" id="voiceName" placeholder="Voice name — e.g. &quot;Narrator (Alex)&quot;" />
          </div>
          <button id="createBtn">Create voice clone</button>
          <div class="vc-status" id="createStatus"></div>
          <div id="createReport" class="clip-list"></div>
        </div>

        <!-- Existing clones + test -->
        <div class="vc-panel">
          <h2>Saved voices</h2>
          <p class="hint">Stored clones. Pick one to load into the tester, download an archive, or import one.</p>
          <div class="vc-actions" style="margin:0 0 10px;">
            <label class="note" style="cursor:pointer; margin:0;">
              <input type="file" id="importInput" accept=".zip,application/zip" style="display:none;" />
              <span class="pill" style="cursor:pointer;">&#8623; Import voice (.zip)</span>
            </label>
            <span class="clip-meta" id="importStatus"></span>
          </div>
          <div id="cloneList"><div class="clip-meta">Loading…</div></div>

          <h2 style="margin-top:16px;">Test a voice</h2>
          <select class="vc-field" id="testClone" style="margin-bottom:8px;"><option value="">Select a saved voice…</option></select>
          <textarea class="tts-text" id="testText" placeholder="Type a line to hear this voice…">Hello! This is a quick test of my cloned voice.</textarea>
          <div style="display:flex; gap:10px; align-items:center; margin:10px 0; flex-wrap:wrap;">
            <select class="vc-field" id="testLang" style="width:auto;">
              <option value="en">English</option>
              <option value="es">Spanish</option>
              <option value="fr">French</option>
              <option value="de">German</option>
              <option value="it">Italian</option>
              <option value="pt">Portuguese</option>
              <option value="ja">Japanese</option>
              <option value="ko">Korean</option>
              <option value="zh">Chinese</option>
            </select>
            <button id="testLiveBtn">Speak (live)</button>
            <button id="testBtn">Speak</button>
          </div>
          <div class="vc-status" id="testStatus"></div>
          <div class="audio-list" id="testAudio"></div>
        </div>
      </div>
    </div>

    <script>
    (function() {
      const recBtn = document.getElementById("recBtn");
      const recLabel = document.getElementById("recLabel");
      const recTimer = document.getElementById("recTimer");
      const fileInput = document.getElementById("fileInput");
      const clipList = document.getElementById("clipList");
      const totalBar = document.getElementById("totalBar");
      const totalLabel = document.getElementById("totalLabel");
      const voiceName = document.getElementById("voiceName");
      const createBtn = document.getElementById("createBtn");
      const createStatus = document.getElementById("createStatus");
      const createReport = document.getElementById("createReport");
      const cloneList = document.getElementById("cloneList");
      const testClone = document.getElementById("testClone");
      const testText = document.getElementById("testText");
      const testLang = document.getElementById("testLang");
      const testBtn = document.getElementById("testBtn");
      const testLiveBtn = document.getElementById("testLiveBtn");
      const testStatus = document.getElementById("testStatus");
      const testAudio = document.getElementById("testAudio");
      const healthBanner = document.getElementById("healthBanner");
      const scoreBtn = document.getElementById("scoreBtn");
      const scoreSummary = document.getElementById("scoreSummary");
      const importInput = document.getElementById("importInput");
      const importStatus = document.getElementById("importStatus");
      const vcEngine = document.getElementById("vcEngine");
      const vcEngineHint = document.getElementById("vcEngineHint");

      // clips: { blob, url, seconds, name, selected, scored, score, rank, snr_db, speech_seconds, clip_ratio, consistency, flags }
      const clips = [];
      let scoreThreshold = 60;

      // ── Engine selection (governs scoring, cloning, listing + the tester) ──
      function currentEngine() { return vcEngine.value || ""; }
      function engineQuery(extra) {
        const e = currentEngine();
        const parts = [];
        if (e) parts.push("engine=" + encodeURIComponent(e));
        if (extra) parts.push(extra);
        return parts.length ? ("?" + parts.join("&")) : "";
      }
      const ENGINE_HINTS = {
        miso: "Miso is one-shot — use one clean, expressive ~10s clip.",
        xtts: "XTTS averages clips — 6–60s of varied, single-speaker audio.",
      };
      function applyEngineHint() {
        vcEngineHint.textContent = ENGINE_HINTS[currentEngine()] || "";
      }

      fetch("/api/voice/engines")
        .then(r => r.json())
        .then(d => {
          const list = d.engines || [];
          vcEngine.innerHTML = '';
          list.forEach(e => {
            const o = document.createElement("option");
            o.value = e.id;
            o.textContent = e.label + (e.default ? " (default)" : "");
            if (e.default) o.selected = true;
            vcEngine.appendChild(o);
          });
          if (list.length <= 1) vcEngine.style.display = "none";
          applyEngineHint();
          loadClones();
        })
        .catch(() => { vcEngine.style.display = "none"; loadClones(); });

      vcEngine.addEventListener("change", () => { applyEngineHint(); loadClones(); });

      // ── Health ──
      fetch("/api/voice/health").then(r => r.json()).then(d => {
        const tts = d.tts || {};
        const ok = tts.status === "ok";
        const state = tts.model_state ? (" (" + tts.model_state + ")") : "";
        healthBanner.innerHTML = '<div style="padding:8px 12px;border-radius:8px;font-size:12px;'
          + 'background:' + (ok ? 'rgba(34,197,94,0.1);color:#22c55e;border:1px solid rgba(34,197,94,0.2)' : 'rgba(239,68,68,0.1);color:#ef4444;border:1px solid rgba(239,68,68,0.2)')
          + ';">TTS: ' + (tts.status || "?") + state + '</div>';
      }).catch(() => {
        healthBanner.innerHTML = '<div style="padding:8px 12px;border-radius:8px;font-size:12px;background:rgba(239,68,68,0.1);color:#ef4444;border:1px solid rgba(239,68,68,0.2);">TTS service unreachable</div>';
      });

      // ── WAV encoding (float32 → 16-bit PCM WAV) ──
      function floatToWav(samples, sampleRate) {
        const dataBytes = samples.length * 2;
        const buf = new ArrayBuffer(44 + dataBytes);
        const dv = new DataView(buf);
        const ws = (off, s) => { for (let i = 0; i < s.length; i++) dv.setUint8(off + i, s.charCodeAt(i)); };
        ws(0, "RIFF"); dv.setUint32(4, 36 + dataBytes, true); ws(8, "WAVE");
        ws(12, "fmt "); dv.setUint32(16, 16, true); dv.setUint16(20, 1, true); dv.setUint16(22, 1, true);
        dv.setUint32(24, sampleRate, true); dv.setUint32(28, sampleRate * 2, true);
        dv.setUint16(32, 2, true); dv.setUint16(34, 16, true);
        ws(36, "data"); dv.setUint32(40, dataBytes, true);
        let off = 44;
        for (let i = 0; i < samples.length; i++) {
          let s = Math.max(-1, Math.min(1, samples[i]));
          dv.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true);
          off += 2;
        }
        return new Blob([buf], { type: "audio/wav" });
      }

      function scoreClass(s) { return s >= scoreThreshold ? "score-good" : (s >= scoreThreshold - 15 ? "score-mid" : "score-bad"); }

      function renderClips() {
        clipList.innerHTML = "";
        let total = 0, selTotal = 0, selCount = 0;
        const anyScored = clips.some(c => c.scored);
        clips.forEach((c, i) => {
          total += c.seconds || 0;
          if (c.selected !== false) { selTotal += c.seconds || 0; selCount++; }
          const div = document.createElement("div");
          div.className = "clip-item" + (c.selected === false ? " excluded" : "");

          const cb = document.createElement("input");
          cb.type = "checkbox";
          cb.checked = c.selected !== false;
          cb.title = "Include in clone";
          cb.addEventListener("change", () => { c.selected = cb.checked; renderClips(); });
          div.appendChild(cb);

          if (anyScored) {
            const badge = document.createElement("span");
            badge.className = "score-badge " + (c.scored ? scoreClass(c.score) : "score-mid");
            badge.textContent = c.scored ? c.score : "–";
            div.appendChild(badge);
          }

          const col = document.createElement("div");
          col.className = "clip-col";
          const details = [];
          if (c.scored) {
            if (c.speech_seconds != null) details.push(c.speech_seconds.toFixed(1) + "s speech");
            if (c.snr_db != null) details.push("SNR " + c.snr_db + "dB");
            if (c.consistency != null) details.push("match " + Math.round(c.consistency * 100) + "%");
          } else {
            details.push((c.seconds ? c.seconds.toFixed(1) + "s" : "?"));
          }
          const flags = (c.flags || []).filter(f => ["clipping", "noisy", "very_short", "outlier"].includes(f));
          let detailHtml = '<span class="clip-detail">' + details.join(" · ");
          if (flags.length) detailHtml += ' · <span class="flag">⚠ ' + flags.join(", ") + '</span>';
          detailHtml += '</span>';
          col.innerHTML = '<span class="clip-name">' + (c.name || "clip").replace(/</g, "&lt;") + '</span>' + detailHtml;
          div.appendChild(col);

          const audio = document.createElement("audio");
          audio.controls = true; audio.preload = "none"; audio.src = c.url;
          audio.style.height = "30px"; audio.style.width = "150px";
          div.appendChild(audio);

          const rm = document.createElement("button");
          rm.className = "rm"; rm.textContent = "Remove";
          rm.addEventListener("click", () => { URL.revokeObjectURL(c.url); clips.splice(i, 1); renderClips(); });
          div.appendChild(rm);

          clipList.appendChild(div);
        });
        totalLabel.textContent = clips.length + " clip" + (clips.length === 1 ? "" : "s") + " · " + total.toFixed(1) + "s total"
          + (anyScored ? "  (" + selCount + " selected · " + selTotal.toFixed(1) + "s)" : "");
        totalBar.style.width = Math.min(100, (selTotal || total) / 30 * 100) + "%";
      }

      // ── Recording via Web Audio (capture PCM → WAV so the sidecar can read it) ──
      let recording = false;
      let audioCtx = null, mediaStream = null, processor = null, srcNode = null, sink = null;
      let recBuffers = [], recSampleRate = 16000, recStart = 0, recTimerInt = null;

      async function startRecording() {
        try {
          mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        } catch (err) {
          recLabel.textContent = "Mic access denied: " + err.message;
          return;
        }
        const AC = window.AudioContext || window.webkitAudioContext;
        audioCtx = new AC();
        recSampleRate = audioCtx.sampleRate;
        srcNode = audioCtx.createMediaStreamSource(mediaStream);
        processor = audioCtx.createScriptProcessor(4096, 1, 1);
        recBuffers = [];
        processor.onaudioprocess = e => {
          recBuffers.push(new Float32Array(e.inputBuffer.getChannelData(0)));
        };
        srcNode.connect(processor);
        sink = audioCtx.createGain();
        sink.gain.value = 0;
        processor.connect(sink);
        sink.connect(audioCtx.destination);

        recording = true;
        recBtn.classList.add("recording");
        recLabel.textContent = "Recording… click to stop";
        recStart = Date.now();
        recTimerInt = setInterval(() => {
          recTimer.textContent = ((Date.now() - recStart) / 1000).toFixed(1) + "s";
        }, 100);
      }

      function stopRecording() {
        recording = false;
        recBtn.classList.remove("recording");
        recLabel.textContent = "Click to record a clip";
        clearInterval(recTimerInt);
        recTimer.textContent = "";
        try { processor.disconnect(); srcNode.disconnect(); sink.disconnect(); } catch (e) {}
        if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());

        let total = 0;
        recBuffers.forEach(b => total += b.length);
        const merged = new Float32Array(total);
        let off = 0;
        recBuffers.forEach(b => { merged.set(b, off); off += b.length; });
        const seconds = total / recSampleRate;
        try { audioCtx.close(); } catch (e) {}
        if (seconds < 0.3) { return; }
        const blob = floatToWav(merged, recSampleRate);
        clips.push({ blob, url: URL.createObjectURL(blob), seconds, name: "clip-" + (clips.length + 1) + ".wav" });
        renderClips();
      }

      recBtn.addEventListener("click", () => { recording ? stopRecording() : startRecording(); });

      // ── File upload ──
      fileInput.addEventListener("change", () => {
        Array.from(fileInput.files).forEach(file => {
          const url = URL.createObjectURL(file);
          const tmp = new Audio();
          tmp.preload = "metadata";
          tmp.onloadedmetadata = () => {
            clips.push({ blob: file, url, seconds: isFinite(tmp.duration) ? tmp.duration : 0, name: file.name });
            renderClips();
          };
          tmp.onerror = () => {
            clips.push({ blob: file, url, seconds: 0, name: file.name });
            renderClips();
          };
          tmp.src = url;
        });
        fileInput.value = "";
      });

      // ── Score / rank clips ──
      scoreBtn.addEventListener("click", async () => {
        if (!clips.length) { scoreSummary.textContent = "Add clips first."; return; }
        const fd = new FormData();
        clips.forEach(c => fd.append("audio", c.blob, c.name));
        scoreBtn.disabled = true;
        scoreBtn.textContent = "Scoring…";
        scoreSummary.textContent = "Scoring " + clips.length + " clip(s) (GPU may need to wake)…";
        try {
          const r = await fetch("/api/voice/analyze" + engineQuery(), { method: "POST", body: fd });
          if (!r.ok) { let m = "HTTP " + r.status; try { m = (await r.text()) || m; } catch (e) {} throw new Error(m); }
          const d = await r.json();
          scoreThreshold = d.threshold != null ? d.threshold : scoreThreshold;
          (d.clips || []).forEach((rep, i) => {
            if (!clips[i]) return;
            clips[i].scored = (rep.score != null);
            clips[i].score = rep.score;
            clips[i].rank = rep.rank;
            clips[i].snr_db = rep.snr_db;
            clips[i].speech_seconds = rep.speech_seconds;
            clips[i].clip_ratio = rep.clip_ratio;
            clips[i].consistency = rep.consistency;
            clips[i].flags = rep.flags || [];
            clips[i].selected = !!rep.recommended;  // default-select clips over threshold
          });
          const sel = clips.filter(c => c.selected !== false).length;
          const dropped = clips.length - sel;
          scoreSummary.textContent = "Scored " + clips.length + " · " + sel + " selected"
            + (dropped ? ", " + dropped + " below threshold (" + scoreThreshold + ")" : "")
            + " · best-first order set";
          renderClips();
        } catch (err) {
          scoreSummary.textContent = "Error: " + err.message;
        } finally {
          scoreBtn.disabled = false;
          scoreBtn.textContent = "Score clips";
        }
      });

      // ── Per-clip preprocessing report ──
      function renderReport(reports) {
        createReport.innerHTML = "";
        reports.forEach((rep, i) => {
          const parts = [];
          if (rep.original_seconds != null && rep.kept_seconds != null) {
            parts.push("kept " + rep.kept_seconds.toFixed(1) + "s of " + rep.original_seconds.toFixed(1) + "s");
          }
          if (rep.snr_db != null) {
            parts.push("SNR ~" + rep.snr_db + " dB");
          }
          const denoised = (rep.applied || []).some(a => a.startsWith("denoise"));
          if (denoised) {
            parts.push("hum reduced");
          }
          if (rep.lufs_in != null && rep.lufs_out != null) {
            const g = (rep.gain_db != null) ? (" (" + (rep.gain_db >= 0 ? "+" : "") + rep.gain_db + " dB)") : "";
            parts.push("loudness " + rep.lufs_in + "\\u2192" + rep.lufs_out + " LUFS" + g);
          }
          const warns = rep.warnings || [];
          const div = document.createElement("div");
          div.className = "clip-item";
          let html = '<span style="flex:1; font-size:12px;">'
            + '<strong>' + (rep.name || ("clip-" + (i + 1))).replace(/</g, "&lt;") + '</strong> · '
            + (parts.join(" · ") || "processed");
          if (warns.length) {
            html += '<br><span style="color:#fbbf24;">⚠ ' + warns.join(", ").replace(/</g, "&lt;") + '</span>';
          }
          html += '</span>';
          div.innerHTML = html;
          createReport.appendChild(div);
        });
      }

      // ── Create clone ──
      createBtn.addEventListener("click", async () => {
        if (!clips.length) { createStatus.textContent = "Add at least one reference clip first."; return; }
        const name = voiceName.value.trim();
        if (!name) { createStatus.textContent = "Give the voice a name."; return; }

        // Use only selected clips; send in best-first (ranked) order when scored.
        const chosen = clips.filter(c => c.selected !== false);
        if (!chosen.length) { createStatus.textContent = "No clips selected."; return; }
        if (chosen.some(c => c.rank != null)) {
          chosen.sort((a, b) => (a.rank != null ? a.rank : 1e9) - (b.rank != null ? b.rank : 1e9));
        }
        const total = chosen.reduce((a, c) => a + (c.seconds || 0), 0);
        if (total && total < 6) { createStatus.textContent = "Note: under 6s total — cloning will run but quality may suffer."; }

        const fd = new FormData();
        fd.append("voice_name", name);
        chosen.forEach(c => fd.append("audio", c.blob, c.name));

        createBtn.disabled = true;
        createStatus.textContent = "Cloning voice from " + chosen.length + " clip(s) (GPU may need to wake)…";
        try {
          const r = await fetch("/api/voice/clone" + engineQuery(), { method: "POST", body: fd });
          if (!r.ok) { let m = "HTTP " + r.status; try { m = (await r.text()) || m; } catch (e) {} throw new Error(m); }
          const d = await r.json();
          createStatus.textContent = 'Created "' + (d.name || name) + '" (' + (d.num_clips || chosen.length) + ' clips). Loaded into the tester.';
          renderReport(d.clips || []);
          await loadClones(d.voice_clone_id);
        } catch (err) {
          createStatus.textContent = "Error: " + err.message;
        } finally {
          createBtn.disabled = false;
        }
      });

      // ── Saved clones ──
      async function loadClones(selectId) {
        try {
          const r = await fetch("/api/voice/clones" + engineQuery());
          const d = await r.json();
          const list = d.clones || [];
          cloneList.innerHTML = "";
          testClone.innerHTML = '<option value="">Select a saved voice…</option>';
          if (!list.length) { cloneList.innerHTML = '<div class="clip-meta">No saved voices yet.</div>'; }
          list.forEach(c => {
            const row = document.createElement("div");
            row.className = "clone-row";
            row.innerHTML = '<div style="flex:1;"><div class="nm">' + (c.name || c.companion_id || "voice").replace(/</g, "&lt;") + '</div>'
              + '<div class="sub">' + (c.num_clips || 1) + ' clip(s) · ' + (c.total_seconds || 0) + 's</div></div>'
              + '<button class="test-btn">Test</button>'
              + '<button class="dl-btn secondary">Download</button>';
            row.querySelector(".test-btn").addEventListener("click", () => {
              testClone.value = c.voice_clone_id;
              testStatus.textContent = 'Loaded "' + (c.name || c.companion_id) + '" — type a line and Speak.';
              testText.focus();
            });
            row.querySelector(".dl-btn").addEventListener("click", () => {
              const a = document.createElement("a");
              a.href = "/api/voice/clones/" + encodeURIComponent(c.companion_id) + "/download" + engineQuery();
              a.download = (c.companion_id || "voice") + ".zip";
              document.body.appendChild(a);
              a.click();
              a.remove();
            });
            cloneList.appendChild(row);

            const opt = document.createElement("option");
            opt.value = c.voice_clone_id;
            opt.textContent = (c.name || c.companion_id) + " (" + (c.num_clips || 1) + " clips)";
            testClone.appendChild(opt);
          });
          if (selectId) testClone.value = selectId;
        } catch (err) {
          cloneList.innerHTML = '<div class="clip-meta">Failed to load voices: ' + err.message + '</div>';
        }
      }

      // ── Import a voice archive ──
      importInput.addEventListener("change", async () => {
        const file = importInput.files[0];
        importInput.value = "";
        if (!file) return;
        const fd = new FormData();
        fd.append("archive", file, file.name);
        importStatus.textContent = "Importing…";
        try {
          const r = await fetch("/api/voice/clones/import" + engineQuery(), { method: "POST", body: fd });
          if (!r.ok) { let m = "HTTP " + r.status; try { m = (await r.text()) || m; } catch (e) {} throw new Error(m); }
          const d = await r.json();
          importStatus.textContent = 'Imported "' + (d.name || d.companion_id) + '"';
          await loadClones(d.voice_clone_id);
        } catch (err) {
          importStatus.textContent = "Error: " + err.message;
        }
      });

      function testPayload() {
        const id = testClone.value;
        if (!id) { testStatus.textContent = "Select a saved voice."; return null; }
        const text = testText.value.trim();
        if (!text) { testStatus.textContent = "Type something to say."; return null; }
        const p = { text, language: testLang.value, voice_clone_id: id };
        if (currentEngine()) p.engine = currentEngine();
        return p;
      }

      testBtn.addEventListener("click", async () => {
        const p = testPayload();
        if (!p) return;
        testBtn.disabled = true;
        testStatus.textContent = "Synthesising…";
        testAudio.innerHTML = "";
        try {
          const r = await fetch("/api/voice/synthesise", {
            method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(p),
          });
          if (!r.ok) { let m = "HTTP " + r.status; try { m = (await r.text()) || m; } catch (e) {} throw new Error(m); }
          const d = await r.json();
          testStatus.textContent = "Duration: " + (d.duration || 0).toFixed(1) + "s";
          const div = document.createElement("div");
          div.className = "audio-item";
          div.innerHTML = '<audio controls autoplay src="' + d.audio_url + '"></audio>';
          testAudio.prepend(div);
        } catch (err) {
          testStatus.textContent = "Error: " + err.message;
        } finally {
          testBtn.disabled = false;
        }
      });

      function floatChunksToWav(chunks, sampleRate) {
        let total = 0; chunks.forEach(c => total += c.length);
        const merged = new Float32Array(total);
        let off = 0; chunks.forEach(c => { merged.set(c, off); off += c.length; });
        return floatToWav(merged, sampleRate);
      }

      testLiveBtn.addEventListener("click", async () => {
        const p = testPayload();
        if (!p) return;
        testLiveBtn.disabled = true;
        testStatus.textContent = "Connecting…";
        testAudio.innerHTML = "";
        const AC = window.AudioContext || window.webkitAudioContext;
        const ctx = new AC();
        try { await ctx.resume(); } catch (e) {}
        const t0 = performance.now();
        let firstAt = null, leftover = new Uint8Array(0), nextTime = ctx.currentTime + 0.08, sr = 24000;
        const collected = [];
        try {
          const resp = await fetch("/api/voice/synthesise/pcm", {
            method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(p),
          });
          if (!resp.ok) { let m = "HTTP " + resp.status; try { m = (await resp.text()) || m; } catch (e) {} throw new Error(m); }
          sr = parseInt(resp.headers.get("X-Sample-Rate") || "24000", 10) || 24000;
          const reader = resp.body.getReader();
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            const merged = new Uint8Array(leftover.length + value.length);
            merged.set(leftover, 0); merged.set(value, leftover.length);
            const usable = merged.length - (merged.length % 4);
            if (usable <= 0) { leftover = merged; continue; }
            const floats = new Float32Array(merged.buffer.slice(0, usable));
            leftover = merged.slice(usable);
            if (!floats.length) continue;
            if (firstAt === null) { firstAt = ((performance.now() - t0) / 1000).toFixed(2); testStatus.textContent = "First audio in " + firstAt + "s — playing…"; }
            collected.push(floats);
            const ab = ctx.createBuffer(1, floats.length, sr);
            ab.copyToChannel(floats, 0);
            const node = ctx.createBufferSource();
            node.buffer = ab; node.connect(ctx.destination);
            const startAt = Math.max(ctx.currentTime, nextTime);
            node.start(startAt); nextTime = startAt + ab.duration;
          }
          let totalSamples = 0; collected.forEach(c => totalSamples += c.length);
          const dur = (totalSamples / sr).toFixed(1);
          testStatus.textContent = "Live done — first audio " + (firstAt || "?") + "s, " + dur + "s total";
          if (collected.length) {
            const div = document.createElement("div");
            div.className = "audio-item";
            div.innerHTML = '<audio controls src="' + URL.createObjectURL(floatChunksToWav(collected, sr)) + '"></audio>';
            testAudio.prepend(div);
          }
        } catch (err) {
          testStatus.textContent = "Error: " + err.message;
        } finally {
          testLiveBtn.disabled = false;
          const remainMs = Math.max(0, (nextTime - ctx.currentTime) * 1000 + 500);
          setTimeout(() => { try { ctx.close(); } catch (e) {} }, remainMs);
        }
      });

      renderClips();
      // Initial clone list is loaded by the engines loader above (once the
      // selected engine is known), so no separate loadClones() call here.
    })();
    </script>
    """

    return render_page("Voice Cloning", body, extra_css)
