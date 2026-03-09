from textwrap import dedent

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from app.config import CHAT_MODEL, DEV_AUTH_BYPASS

router = APIRouter()


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
                  height: 92vh;
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
                <div class="row input-wrap" style="margin; align-items:center; margin: 0px;">
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
              </div>
              <script>
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
