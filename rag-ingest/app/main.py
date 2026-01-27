import base64
import hashlib
import hmac
import json
import os
import time
from textwrap import dedent
from typing import Any, Dict, Optional

from cryptography.fernet import Fernet, InvalidToken
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.ingest_pipeline import ingest_docs
from app.nonce_store import NonceStore, NonceStoreConfig

app = FastAPI(title="RAG Ingestion API")

SHARED_SECRET = os.environ.get("INGEST_SHARED_SECRET", "")
REQUIRE_ENCRYPTION = os.environ.get("INGEST_REQUIRE_ENCRYPTION", "0") == "1"
NONCE_DB_PATH = os.environ.get("INGEST_NONCE_DB", "/data/nonces.db")
MAX_SKEW_S = int(os.environ.get("INGEST_MAX_SKEW_S", "300"))
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")

if not SHARED_SECRET:
    raise RuntimeError("INGEST_SHARED_SECRET is required")

_nonce_store = NonceStore(
    NonceStoreConfig(
        db_path=NONCE_DB_PATH,
        nonce_ttl_s=max(600, MAX_SKEW_S + 120),
    )
)

def _hmac_key() -> bytes:
    return hashlib.sha256(SHARED_SECRET.encode("utf-8")).digest()

def sign_body(ts: str, nonce: str, body_bytes: bytes) -> str:
    body_hash = hashlib.sha256(body_bytes).hexdigest().encode("ascii")
    msg = ts.encode("ascii") + b"." + nonce.encode("ascii") + b"." + body_hash
    return hmac.new(_hmac_key(), msg, hashlib.sha256).hexdigest()

def verify_signature(ts: str, nonce: str, signature: str, body_bytes: bytes, max_skew_s: int = MAX_SKEW_S) -> None:
    try:
        ts_i = int(ts)
    except ValueError:
        raise HTTPException(401, "Invalid X-Timestamp")

    now = int(time.time())
    if abs(now - ts_i) > max_skew_s:
        raise HTTPException(401, "Request timestamp outside allowed window")

    expected = sign_body(ts, nonce, body_bytes)
    if not hmac.compare_digest(expected, signature):
        raise HTTPException(401, "Invalid signature")

    # Replay protection: persist nonces and reject repeats.
    try:
        _nonce_store.claim_or_reject(nonce=nonce, ts=ts_i)
    except ValueError:
        raise HTTPException(401, "Replay detected (nonce already used)")

def _fernet() -> Fernet:
    key = base64.urlsafe_b64encode(hashlib.sha256(SHARED_SECRET.encode("utf-8")).digest())
    return Fernet(key)

class IngestDoc(BaseModel):
    docs: list[Dict[str, Any]] = Field(..., description="List of documents to index")
    source: Optional[str] = Field(None, description="Source identifier (e.g. 'grocy', 'growdb')")
    tags: Optional[list[str]] = Field(default_factory=list)

@app.get("/health")
def health():
    return {"ok": True}


@app.get("/ui/ingest", response_class=HTMLResponse)
def ingest_ui():
    html = dedent(
        """
            <!doctype html>
            <html>
            <head>
              <meta charset="utf-8" />
              <title>rag-ingest UI</title>
              <style>
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
                  padding: 48px 16px 64px;
                }
                .card {
                  width: min(960px, 100%);
                  background: linear-gradient(180deg, var(--card-2), var(--card));
                  border: 1px solid var(--border);
                  border-radius: 16px;
                  box-shadow: 0 20px 60px rgba(0,0,0,0.35);
                  padding: 28px;
                }
                h1 {
                  margin-top: 0;
                  margin-bottom: 8px;
                  font-size: 24px;
                  letter-spacing: 0.2px;
                }
                p.sub {
                  margin-top: 0;
                  margin-bottom: 20px;
                  color: var(--muted);
                }
                label {
                  display: block;
                  font-weight: 600;
                  margin-bottom: 6px;
                }
                textarea, input, button {
                  width: 100%;
                  font: inherit;
                  border-radius: 10px;
                  border: 1px solid var(--border);
                  background: #0b1221;
                  color: var(--text);
                  padding: 10px 12px;
                  box-sizing: border-box;
                }
                textarea {
                  min-height: 170px;
                  resize: vertical;
                }
                button {
                  width: auto;
                  padding: 10px 16px;
                  border: none;
                  background: linear-gradient(90deg, var(--accent), var(--accent-2));
                  color: #0b1020;
                  font-weight: 700;
                  cursor: pointer;
                  transition: transform 120ms ease, box-shadow 120ms ease, opacity 120ms ease;
                  box-shadow: 0 8px 20px rgba(56,189,248,0.25);
                }
                button:hover {
                  transform: translateY(-1px);
                  opacity: 0.95;
                }
                .row { margin-bottom: 16px; }
                .note { color: var(--muted); font-size: 13px; }
                pre {
                  white-space: pre-wrap;
                  word-break: break-word;
                  padding: 14px;
                  background: #0b1221;
                  border: 1px solid var(--border);
                  border-radius: 12px;
                  min-height: 80px;
                }
              </style>
            </head>
            <body>
              <div class="card">
                <h1>rag-ingest</h1>
                <p class="sub">Client-side signed ingestion. Keep this UI behind trusted access. Using embed model: __EMBED_MODEL__. Encryption required: __REQ_ENC__.</p>
                <div class="row">
                  <label>Shared secret</label>
                  <input id="secret" type="password" />
                </div>
                <div class="row">
                  <label>Document ID</label>
                  <input id="docId" placeholder="doc-1" />
                </div>
                <div class="row">
                  <label>Source</label>
                  <input id="source" placeholder="demo" />
                </div>
                <div class="row">
                  <label>Tags (comma separated)</label>
                  <input id="tags" placeholder="example,tag2" />
                </div>
                <div class="row">
                  <label>Text</label>
                  <textarea id="text" placeholder="Document text to ingest..."></textarea>
                </div>
                <button id="send">Send</button>
                <div class="row" style="margin-top:18px;">
                  <h3 style="margin:0 0 8px;">Response</h3>
                  <pre id="out">—</pre>
                </div>
              </div>
              <script>
                const enc = new TextEncoder();
                const outEl = document.getElementById("out");
                const REQUIRE_ENCRYPTION = __REQ_ENC_BOOL__;
                const sendBtn = document.getElementById("send");
                const inputs = Array.from(document.querySelectorAll("input, textarea, button"));

                if (REQUIRE_ENCRYPTION) {
                  outEl.textContent = "UI disabled: this form only works when INGEST_REQUIRE_ENCRYPTION=0. Use a client that sends an encrypted envelope (token) when encryption is required.";
                  sendBtn.disabled = true;
                  inputs.forEach((el) => {
                    if (el !== outEl) el.disabled = true;
                  });
                }

                function hex(buffer) {
                  return Array.from(new Uint8Array(buffer)).map(b => b.toString(16).padStart(2, "0")).join("");
                }

                async function sha256Hex(data) {
                  const hash = await crypto.subtle.digest("SHA-256", data);
                  return hex(hash);
                }

                async function sha256Raw(data) {
                  return new Uint8Array(await crypto.subtle.digest("SHA-256", data));
                }

                async function hmacHex(secret, data) {
                  // Server derives the HMAC key as SHA256(secret); mirror that here.
                  const secretHash = await crypto.subtle.digest("SHA-256", enc.encode(secret));
                  const key = await crypto.subtle.importKey(
                    "raw",
                    secretHash,
                    { name: "HMAC", hash: "SHA-256" },
                    false,
                    ["sign"]
                  );
                  const sig = await crypto.subtle.sign("HMAC", key, data);
                  return hex(sig);
                }

                function base64UrlEncode(bytes) {
                  const b64 = btoa(String.fromCharCode(...bytes));
                  return b64.replace(/\\+/g, "-").replace(/\\//g, "_").replace(/=+$/g, "");
                }

                function pkcs7Pad(buf) {
                  const block = 16;
                  const padLen = block - (buf.length % block || block);
                  const out = new Uint8Array(buf.length + padLen);
                  out.set(buf, 0);
                  out.fill(padLen, buf.length);
                  return out;
                }

                async function fernetEncrypt(secret, plaintextBytes) {
                  // Implements Fernet compatible with Python cryptography.
                  const keyHash = await sha256Raw(enc.encode(secret));
                  const signKey = keyHash.slice(0, 16);
                  const encKey = keyHash.slice(16, 32);

                  const iv = crypto.getRandomValues(new Uint8Array(16));
                  const ts = Math.floor(Date.now() / 1000);
                  const tsBuf = new ArrayBuffer(8);
                  new DataView(tsBuf).setBigUint64(0, BigInt(ts), false); // big-endian

                  const padded = pkcs7Pad(plaintextBytes);
                  const aesKey = await crypto.subtle.importKey("raw", encKey, { name: "AES-CBC" }, false, ["encrypt"]);
                  const ciphertext = new Uint8Array(await crypto.subtle.encrypt({ name: "AES-CBC", iv }, aesKey, padded));

                  const base = new Uint8Array(1 + 8 + 16 + ciphertext.length);
                  base[0] = 0x80;
                  base.set(new Uint8Array(tsBuf), 1);
                  base.set(iv, 9);
                  base.set(ciphertext, 25);

                  const hmacKey = await crypto.subtle.importKey("raw", signKey, { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
                  const sig = new Uint8Array(await crypto.subtle.sign("HMAC", hmacKey, base));

                  const token = new Uint8Array(base.length + sig.length);
                  token.set(base, 0);
                  token.set(sig, base.length);
                  return base64UrlEncode(token);
                }

                function randomHex(bytes) {
                  const arr = new Uint8Array(bytes);
                  crypto.getRandomValues(arr);
                  return hex(arr);
                }

                async function send() {
                  if (REQUIRE_ENCRYPTION) {
                    outEl.textContent = "UI disabled: set INGEST_REQUIRE_ENCRYPTION=0 to use this page, or send an encrypted envelope with your own client.";
                    return;
                  }
                  const secret = document.getElementById("secret").value.trim();
                  const docId = document.getElementById("docId").value.trim() || "doc-1";
                  const source = document.getElementById("source").value.trim() || null;
                  const tagsRaw = document.getElementById("tags").value.trim();
                  const text = document.getElementById("text").value.trim();
                  if (!secret) {
                    outEl.textContent = "Shared secret required.";
                    return;
                  }
                  if (!text) {
                    outEl.textContent = "Text is required.";
                    return;
                  }
                  const tags = tagsRaw ? tagsRaw.split(",").map(t => t.trim()).filter(Boolean) : [];
                  const payload = { docs: [{ id: docId, text, meta: { source } }], source, tags };

                  let bodyStr;
                  if (REQUIRE_ENCRYPTION) {
                    const plaintext = enc.encode(JSON.stringify(payload));
                    const token = await fernetEncrypt(secret, plaintext);
                    bodyStr = JSON.stringify({ token });
                  } else {
                    bodyStr = JSON.stringify(payload);
                  }

                  const bodyBytes = enc.encode(bodyStr);
                  const bodyHash = await sha256Hex(bodyBytes);
                  const ts = Math.floor(Date.now() / 1000).toString();
                  const nonce = randomHex(16);
                  const msg = enc.encode(`${ts}.${nonce}.${bodyHash}`);
                  const sig = await hmacHex(secret, msg);

                  outEl.textContent = "Sending...";
                  try {
                    const r = await fetch("/ingest", {
                      method: "POST",
                      headers: {
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "X-Timestamp": ts,
                        "X-Nonce": nonce,
                        "X-Signature": sig,
                      },
                      body: bodyStr,
                    });
                    const txt = await r.text();
                    try {
                      const data = JSON.parse(txt);
                      outEl.textContent = JSON.stringify(data, null, 2);
                    } catch (e) {
                      outEl.textContent = txt;
                    }
                  } catch (e) {
                    outEl.textContent = "Error: " + e;
                  }
                }

                document.getElementById("send").addEventListener("click", send);
                document.getElementById("text").addEventListener("keydown", (e) => {
                  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) send();
                });
              </script>
            </body>
            </html>
            """
    )
    html = html.replace("__EMBED_MODEL__", EMBED_MODEL)
    html = html.replace("__REQ_ENC__", "enabled" if REQUIRE_ENCRYPTION else "disabled")
    html = html.replace("__REQ_ENC_BOOL__", "true" if REQUIRE_ENCRYPTION else "false")
    return HTMLResponse(html)


@app.post("/ingest")
async def ingest(
    request: Request,
    x_timestamp: str = Header(..., alias="X-Timestamp"),
    x_nonce: str = Header(..., alias="X-Nonce"),
    x_signature: str = Header(..., alias="X-Signature"),
):
    body = await request.body()
    verify_signature(x_timestamp, x_nonce, x_signature, body)

    try:
        payload_json = json.loads(body.decode("utf-8"))
    except Exception:
        raise HTTPException(400, "Body must be JSON")

    # Parse body (either plaintext JSON or encrypted envelope)
    if REQUIRE_ENCRYPTION:
        if "token" not in payload_json:
            raise HTTPException(400, "Encryption required: body must be an envelope with {token}")
        try:
            plaintext = _fernet().decrypt(payload_json["token"].encode("utf-8"))
            data = json.loads(plaintext.decode("utf-8"))
        except (InvalidToken, Exception):
            raise HTTPException(400, "Invalid encrypted envelope")
    else:
        # If token present, decrypt; else accept plaintext
        if "token" in payload_json and isinstance(payload_json["token"], str):
            try:
                plaintext = _fernet().decrypt(payload_json["token"].encode("utf-8"))
                data = json.loads(plaintext.decode("utf-8"))
            except (InvalidToken, Exception):
                raise HTTPException(400, "Invalid encrypted envelope")
        else:
            data = payload_json

    try:
        ingest_doc = IngestDoc.model_validate(data)
    except Exception as e:
        raise HTTPException(422, f"Invalid ingest payload: {e}")

    # Real pipeline:
    # - chunk
    # - embed via Ollama
    # - upsert into Qdrant
    try:
        result = await ingest_docs(ingest_doc.docs, source=ingest_doc.source, tags=ingest_doc.tags or [])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Ingestion pipeline failed: {e}")

    result.update({"received_docs": len(ingest_doc.docs), "source": ingest_doc.source, "tags": ingest_doc.tags or []})
    return result
