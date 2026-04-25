import json
from textwrap import dedent

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from app.config import CHAT_MODEL, DEV_AUTH_BYPASS

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
        '<strong style="color:var(--text);">rag-chat</strong>'
        '<a href="/ui/chat" style="color:var(--text); text-decoration:none;">Chat</a>'
        '<a href="/ui/history" style="color:var(--text); text-decoration:none;">History</a>'
        '<a href="/ui/generate" style="color:var(--text); text-decoration:none;">Generate</a>'
        '<a href="/ui/api-keys" style="color:var(--text); text-decoration:none;">API keys</a>'
        '<a href="/ui/pipeline" style="color:var(--accent); text-decoration:none; font-weight:600;">Pipeline Lab</a>'
        f'<span style="margin-left:auto; color:var(--muted); font-size:13px;">{bypass}</span>'
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
    .pl-status-bar {
      display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
      background: rgba(0,0,0,0.25); border: 1px solid var(--border);
      border-radius: 10px; padding: 8px 12px; margin-bottom: 16px;
    }
    .state-chip {
      padding: 3px 10px; border-radius: 999px; font-size: 12px;
      font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase;
    }
    .state-chip.unloaded { background: #1f2937; color: var(--muted); }
    .state-chip.loading  { background: rgba(251,191,36,0.2); color: #fbbf24; }
    .state-chip.cpu      { background: rgba(99,102,241,0.2); color: #818cf8; }
    .state-chip.gpu      { background: rgba(52,211,153,0.2); color: #34d399; }
    .state-chip.error    { background: rgba(239,68,68,0.15); color: #f87171; }
    .pl-status-meta { color: var(--muted); font-size: 12px; }
    .pl-btn-sm {
      padding: 4px 10px; font-size: 12px; font-weight: 600;
      border-radius: 8px; border: 1px solid var(--border);
      background: #0b1221; color: var(--text); cursor: pointer;
      transition: border-color 120ms;
    }
    .pl-btn-sm:hover { border-color: var(--accent); }
    .pl-columns { display: grid; grid-template-columns: 340px 1fr; gap: 20px; }
    .pl-left { display: flex; flex-direction: column; gap: 14px; }
    .pl-section {
      background: rgba(0,0,0,0.2); border: 1px solid var(--border);
      border-radius: 12px; padding: 14px;
    }
    .pl-section label {
      font-size: 13px; font-weight: 600; display: block; margin-bottom: 6px;
    }
    .pl-section textarea, .pl-section input[type=text] {
      width: 100%; background: #0b1221; border: 1px solid var(--border);
      border-radius: 8px; color: var(--text); font: inherit; font-size: 13px;
      padding: 8px 10px; resize: vertical; box-sizing: border-box;
    }
    .pl-section textarea:focus, .pl-section input[type=text]:focus {
      outline: none; border-color: var(--accent);
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
    .run-btn {
      width: 100%; padding: 11px; font-size: 14px; font-weight: 700;
      border-radius: 10px; border: none; cursor: pointer;
      background: linear-gradient(90deg, var(--accent), var(--accent-2));
      color: #0b1020; transition: opacity 120ms, transform 120ms;
      box-shadow: 0 6px 18px rgba(56,189,248,0.25);
    }
    .run-btn:hover { opacity: 0.9; transform: translateY(-1px); }
    .run-btn:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }
    .timing-note { color: var(--muted); font-size: 12px; margin-top: 6px; min-height: 18px; }
    .pl-right { display: flex; flex-direction: column; gap: 0; }
    .stage-card {
      border: 1px solid var(--border); border-radius: 12px;
      padding: 14px 16px; background: rgba(0,0,0,0.18);
    }
    .stage-card.live { border-color: rgba(56,189,248,0.45); }
    .stage-card.stub { opacity: 0.72; }
    .stage-connector {
      display: flex; align-items: center; justify-content: center;
      color: var(--muted); font-size: 20px; padding: 2px 0;
    }
    .stage-header { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
    .stage-name { font-weight: 700; font-size: 15px; }
    .stage-badge {
      font-size: 11px; font-weight: 700; letter-spacing: 0.05em;
      padding: 2px 8px; border-radius: 999px; text-transform: uppercase;
    }
    .stage-badge.live  { background: rgba(56,189,248,0.18); color: var(--accent); }
    .stage-badge.stub  { background: #1f2937; color: var(--muted); }
    .stage-meta { font-size: 12px; color: var(--muted); display: flex; flex-direction: column; gap: 3px; }
    .stage-knows { font-style: italic; margin-bottom: 2px; }
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
      width: 120px; text-align: right; color: var(--muted);
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .score-bar-bg { flex: 1; height: 6px; background: #1f2937; border-radius: 3px; }
    .score-bar-fill {
      height: 100%; border-radius: 3px;
      background: linear-gradient(90deg, var(--accent-2), var(--accent));
    }
    .score-val { width: 42px; color: var(--text); font-variant-numeric: tabular-nums; }
    .stage-stub-preview {
      margin-top: 10px; padding: 8px 10px;
      background: rgba(99,102,241,0.06); border: 1px dashed rgba(99,102,241,0.25);
      border-radius: 8px; font-size: 12px; color: var(--muted);
    }
    .stage-stub-preview b { color: var(--text); }
    .error-box {
      padding: 8px 12px; border-radius: 8px; font-size: 13px;
      background: rgba(239,68,68,0.12); border: 1px solid rgba(239,68,68,0.4);
      color: #f87171; margin-top: 8px;
    }
    @media (max-width: 900px) { .pl-columns { grid-template-columns: 1fr; } }
    """

    body = f"""
    <div class="pl-wrap">
      <div style="display:flex; align-items:baseline; gap:12px; margin-bottom:4px;">
        <h1 style="margin:0; font-size:22px;">Pipeline Lab</h1>
        <span style="color:var(--muted); font-size:13px;">ModernBERT classifier &middot; stage 1 of 4</span>
      </div>
      <p class="pl-sub">
        Scout and refine the classifier step. Stages 2&ndash;4 are stubs showing the intended contract.
        Each model knows exactly what it needs &mdash; no more.
      </p>

      <div class="pl-status-bar">
        <span id="stateChip" class="state-chip unloaded">unloaded</span>
        <span id="statusMeta" class="pl-status-meta">&mdash;</span>
        <span style="margin-left:auto; display:flex; gap:6px;">
          <button class="pl-btn-sm" onclick="mbLoad()">Load</button>
          <button class="pl-btn-sm" onclick="mbEvict()">Evict to CPU</button>
          <button class="pl-btn-sm" onclick="mbUnload()">Unload</button>
        </span>
      </div>

      <div class="pl-columns">

        <div class="pl-left">

          <div class="pl-section">
            <label>Label set</label>
            <textarea id="labelsInput" rows="5"
              placeholder="curious, defensive, playful, sad, angry, flirtatious, neutral, ..."></textarea>
            <div style="display:flex; gap:8px; align-items:center; margin-top:8px;">
              <button class="pl-btn-sm" onclick="updateLabels()">Update labels</button>
              <span id="labelsStatus" style="color:var(--muted); font-size:12px;"></span>
            </div>
            <div style="color:var(--muted); font-size:11px; margin-top:6px;">
              Comma or newline separated. Re-embeds immediately if model is resident.
            </div>
          </div>

          <div class="pl-section">
            <label>Context turns <span style="font-weight:400; color:var(--muted);">(oldest first, up to 5 used)</span></label>
            <div id="turnsList"></div>
            <button class="pl-link-btn" onclick="addTurn()" style="margin-top:4px;">+ add turn</button>
          </div>

          <div class="pl-section">
            <label>User message</label>
            <textarea id="msgInput" rows="4" placeholder="what the user just sent&hellip;"></textarea>
            <button class="run-btn" id="runBtn" onclick="runClassify()" style="margin-top:10px;">
              Run ModernBERT &#8594;
            </button>
            <div class="timing-note" id="timingNote"></div>
          </div>

        </div>

        <div class="pl-right">

          <div class="stage-card live">
            <div class="stage-header">
              <span class="stage-name">1 &middot; ModernBERT</span>
              <span class="stage-badge live">LIVE &middot; 149M params</span>
            </div>
            <div class="stage-meta">
              <div class="stage-knows">Knows: user message + 5 turns of context. Not who the character is.</div>
              <div class="stage-io"><b>Gets:</b> raw text &mdash; message + 5 turns context</div>
              <div class="stage-io"><b>Produces:</b> 5&ndash;10 classifier tokens (ranked labels + scores)</div>
            </div>
            <div id="stage1Output" style="display:none;" class="stage-output">
              <div class="stage-output-title">Output &mdash; classifier tokens</div>
              <div id="tokenChips" class="token-chips"></div>
              <div id="scoreBars" class="score-bars"></div>
            </div>
            <div id="stage1Error" style="display:none;" class="error-box"></div>
          </div>

          <div class="stage-connector">&#8595;</div>

          <div class="stage-card stub">
            <div class="stage-header">
              <span class="stage-name">2 &middot; 4B Overlay</span>
              <span class="stage-badge stub">STUB &middot; external</span>
            </div>
            <div class="stage-meta">
              <div class="stage-knows">Knows: character config + current state. Not full session history.</div>
              <div class="stage-io"><b>Gets:</b> character config + state + classifier tokens + turn context</div>
              <div class="stage-io"><b>Produces:</b> style tuner &mdash; 80&ndash;120 token voice directive</div>
            </div>
            <div id="stage2Preview" class="stage-stub-preview" style="display:none;">
              <b>Would receive classifier tokens:</b>
              <span id="stage2Tokens"></span>
            </div>
          </div>

          <div class="stage-connector">&#8595;</div>

          <div class="stage-card stub">
            <div class="stage-header">
              <span class="stage-name">3 &middot; Brain</span>
              <span class="stage-badge stub">STUB &middot; large &middot; once/turn</span>
            </div>
            <div class="stage-meta">
              <div class="stage-knows">Knows: everything &mdash; character, state, history, memory, promises, events, classifier read, voice directive. Does not write in the character&apos;s voice.</div>
              <div class="stage-io"><b>Gets:</b> full context + style tuner + classifier tokens + history</div>
              <div class="stage-io"><b>Produces:</b> think block + response skeleton</div>
            </div>
          </div>

          <div class="stage-connector">&#8595;</div>

          <div class="stage-card stub">
            <div class="stage-header">
              <span class="stage-name">4 &middot; Prose Model</span>
              <span class="stage-badge stub">STUB &middot; medium &middot; streams</span>
            </div>
            <div class="stage-meta">
              <div class="stage-knows">Knows: how to write in the character&apos;s voice + what to write from the skeleton. Does not reason over the relationship arc.</div>
              <div class="stage-io"><b>Gets:</b> style tuner + skeleton + recent history</div>
              <div class="stage-io"><b>Produces:</b> character&apos;s streamed response</div>
            </div>
          </div>

        </div>
      </div>
    </div>

    <script>
      const DEV_BYPASS = {"true" if DEV_AUTH_BYPASS else "false"};

      function apiHeaders() {{
        const h = {{"Content-Type": "application/json"}};
        if (!DEV_BYPASS) {{
          const k = (localStorage.getItem("rag_api_key") || "").trim();
          if (k) h["Authorization"] = "Bearer " + k;
        }}
        return h;
      }}

      async function loadStatus() {{
        try {{
          const r = await fetch("/api/pipeline/status", {{headers: apiHeaders()}});
          if (!r.ok) {{ setChip("error", "service error"); return; }}
          const d = await r.json();
          setChip(d.state, d.model_id, d.last_used, d.device);
          if (!document.getElementById("labelsInput").value.trim()) loadLabels();
        }} catch(e) {{
          setChip("error", "unreachable");
        }}
      }}

      function setChip(state, modelId, lastUsed, device) {{
        const chip = document.getElementById("stateChip");
        chip.className = "state-chip " + (state || "unloaded");
        chip.textContent = state || "unloaded";
        const meta = document.getElementById("statusMeta");
        const parts = [];
        if (modelId) parts.push(modelId);
        if (device && device !== "none") parts.push(device);
        if (lastUsed) {{
          const secs = Math.round((Date.now() / 1000) - lastUsed);
          parts.push("last used " + (secs < 60 ? secs + "s ago" : Math.round(secs / 60) + "m ago"));
        }}
        meta.textContent = parts.join(" · ") || "—";
      }}

      async function mbAction(path) {{
        setChip("loading");
        try {{
          const r = await fetch(path, {{method: "POST", headers: apiHeaders()}});
          const d = await r.json();
          setChip(d.state, null, null, d.device);
        }} catch(e) {{ setChip("error"); }}
        setTimeout(loadStatus, 1000);
      }}

      const mbLoad   = () => mbAction("/api/pipeline/load");
      const mbEvict  = () => mbAction("/api/pipeline/evict");
      const mbUnload = () => mbAction("/api/pipeline/unload");

      async function loadLabels() {{
        try {{
          const r = await fetch("/api/pipeline/labels", {{headers: apiHeaders()}});
          if (!r.ok) return;
          const d = await r.json();
          if (d.labels && d.labels.length) {{
            document.getElementById("labelsInput").value = d.labels.join(", ");
          }}
        }} catch(_) {{}}
      }}

      async function updateLabels() {{
        const raw = document.getElementById("labelsInput").value;
        const labels = raw.split(/[,\\n]+/).map(s => s.trim()).filter(Boolean);
        if (!labels.length) return;
        const el = document.getElementById("labelsStatus");
        el.textContent = "updating…";
        try {{
          const r = await fetch("/api/pipeline/labels", {{
            method: "PUT",
            headers: apiHeaders(),
            body: JSON.stringify({{labels}}),
          }});
          if (r.ok) {{
            const d = await r.json();
            const emb = d.embeddings_ready ? " · embeddings ready" : " · will embed on next classify";
            el.textContent = labels.length + " labels" + emb;
          }} else {{
            el.textContent = "error " + r.status;
          }}
        }} catch(e) {{ el.textContent = "unreachable"; }}
      }}

      function addTurn(val) {{
        const list = document.getElementById("turnsList");
        const n = list.children.length + 1;
        const row = document.createElement("div");
        row.className = "turn-row";
        const safe = (val || "").replace(/"/g, "&quot;");
        row.innerHTML =
          '<input type="text" placeholder="turn ' + n + '" value="' + safe + '" />' +
          '<button class="turn-del" onclick="this.parentElement.remove()" title="remove">×</button>';
        list.appendChild(row);
      }}

      function getContext() {{
        return Array.from(document.querySelectorAll("#turnsList .turn-row input"))
          .map(i => i.value.trim()).filter(Boolean);
      }}

      async function runClassify() {{
        const text = document.getElementById("msgInput").value.trim();
        if (!text) {{ document.getElementById("msgInput").focus(); return; }}

        const btn = document.getElementById("runBtn");
        btn.disabled = true;
        btn.textContent = "Running…";
        document.getElementById("stage1Output").style.display = "none";
        document.getElementById("stage1Error").style.display = "none";
        document.getElementById("stage2Preview").style.display = "none";
        document.getElementById("timingNote").textContent = "";

        const t0 = Date.now();
        try {{
          const r = await fetch("/api/pipeline/classify", {{
            method: "POST",
            headers: apiHeaders(),
            body: JSON.stringify({{text, context: getContext()}}),
          }});
          const elapsed = Date.now() - t0;

          if (!r.ok) {{
            const msg = await r.text();
            showError(msg);
            document.getElementById("timingNote").textContent = elapsed + "ms (error)";
            return;
          }}

          const d = await r.json();
          renderStage1(d);
          document.getElementById("timingNote").textContent =
            elapsed + "ms · task: " + d.task + " · model state: " + d.model_state;

          if (d.labels && d.labels.length) showStage2Preview(d.labels, d.scores);
          loadStatus();
        }} catch(e) {{
          showError(e.toString());
        }} finally {{
          btn.disabled = false;
          btn.textContent = "Run ModernBERT →";
        }}
      }}

      function showError(msg) {{
        const el = document.getElementById("stage1Error");
        el.style.display = "block";
        el.textContent = msg;
      }}

      function renderStage1(d) {{
        document.getElementById("stage1Output").style.display = "block";

        document.getElementById("tokenChips").innerHTML = d.labels.map((lbl, i) =>
          '<span class="token-chip">' + esc(lbl) +
          '<span class="token-score">' + d.scores[i].toFixed(3) + '</span></span>'
        ).join("");

        const raw = d.raw || {{}};
        const sorted = Object.entries(raw).sort((a, b) => b[1] - a[1]);
        document.getElementById("scoreBars").innerHTML = sorted.map(([lbl, score]) => {{
          const pct = Math.max(0, Math.min(100, score * 100)).toFixed(1);
          return '<div class="score-row">' +
            '<span class="score-label" title="' + esc(lbl) + '">' + esc(lbl) + '</span>' +
            '<div class="score-bar-bg"><div class="score-bar-fill" style="width:' + pct + '%"></div></div>' +
            '<span class="score-val">' + score.toFixed(3) + '</span></div>';
        }}).join("");
      }}

      function showStage2Preview(labels, scores) {{
        document.getElementById("stage2Preview").style.display = "block";
        document.getElementById("stage2Tokens").innerHTML = " " + labels.map((lbl, i) =>
          '<span style="background:rgba(99,102,241,0.15);border:1px solid rgba(99,102,241,0.35);' +
          'border-radius:6px;padding:2px 7px;font-size:12px;margin:0 2px;">' +
          esc(lbl) + ' <span style="color:var(--accent);">' + scores[i].toFixed(2) + '</span></span>'
        ).join("");
      }}

      function esc(s) {{
        return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
      }}

      loadStatus();
      loadLabels();
      setInterval(loadStatus, 12000);
    </script>
    """

    return render_page("Pipeline Lab", body, extra_css)
