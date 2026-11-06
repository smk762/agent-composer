# Integration guide (external services / IDE agents)

This document shows how to call `rag-chat` and `rag-ingest` from another host (laptop, CI, agent) without exposing Ollama or Qdrant.

## Connection options
- **LAN-only:** keep compose defaults (bind `127.0.0.1`) and use host firewall to allow only trusted LAN IPs to ports `9050/9150`, or set `RAG_INGEST_BIND_IP` / `RAG_CHAT_BIND_IP` to a LAN address and restrict at the firewall.
- **Cloudflare Access (recommended for Internet-facing):** run with the `cloud` profile so only `cloudflared` is reachable externally. Protect hostnames with Access (SSO/MFA for humans, service tokens for machines). Do not expose Qdrant or Ollama.

## Endpoints and auth
- **`rag-chat`**
  - URL: `${CHAT_URL:-http://127.0.0.1:9150}`
  - Auth: API key from `/ui/api-keys` → header `Authorization: Bearer <key>`
  - Health: `GET /health`
  - Model names: `/models` returns tag-qualified names (often `:latest`). `/chat` accepts either `foo` or `foo:latest` and will try to resolve/normalize.
- **`rag-ingest`**
  - URL: `${INGEST_URL:-http://127.0.0.1:9050}`
  - Auth: HMAC headers (`X-Timestamp`, `X-Nonce`, `X-Signature`) using `INGEST_SHARED_SECRET`
  - Optional: encrypted envelope if `INGEST_REQUIRE_ENCRYPTION=1`
  - Health: `GET /health`

Recommended client env for agents/CI:
```
CHAT_URL=https://your-chat-hostname
CHAT_API_KEY=...
INGEST_URL=https://your-ingest-hostname
INGEST_SHARED_SECRET=...
```

## Quickstarts
### Chat (curl)
```bash
curl -s "${CHAT_URL}/chat" \
  -H "Authorization: Bearer ${CHAT_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello!"}]}'
```

### Ingest (Python signing helper + curl)
```python
import hashlib, hmac, json, os, secrets, time

SECRET = os.environ["INGEST_SHARED_SECRET"]
INGEST_URL = os.environ.get("INGEST_URL", "http://127.0.0.1:9050/ingest")

def sign(ts: str, nonce: str, body_bytes: bytes) -> str:
    body_hash = hashlib.sha256(body_bytes).hexdigest().encode("ascii")
    msg = ts.encode("ascii") + b"." + nonce.encode("ascii") + b"." + body_hash
    return hmac.new(hashlib.sha256(SECRET.encode()).digest(), msg, hashlib.sha256).hexdigest()

payload = {"docs":[{"id":"doc-1","text":"hello","meta":{"source":"demo"}}],"source":"demo"}
body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()
ts = str(int(time.time()))
nonce = secrets.token_hex(16)
sig = sign(ts, nonce, body)
print(ts, nonce, sig, body.decode())
```

Use the printed values with curl:
```bash
curl -s "${INGEST_URL}" \
  -H "Content-Type: application/json" \
  -H "X-Timestamp: ${TS}" \
  -H "X-Nonce: ${NONCE}" \
  -H "X-Signature: ${SIG}" \
  -d "${BODY}"
```

#### Optional encrypted envelope (when `INGEST_REQUIRE_ENCRYPTION=1`)
```python
import base64, hashlib, json, os
from cryptography.fernet import Fernet

secret = os.environ["INGEST_SHARED_SECRET"]
key = base64.urlsafe_b64encode(hashlib.sha256(secret.encode()).digest())
f = Fernet(key)
inner = {"docs":[{"id":"doc-1","text":"hello"}]}
token = f.encrypt(json.dumps(inner).encode()).decode()
print(json.dumps({"token": token}))
```

## Testing and troubleshooting
- Smoke test against any URL: `CHAT_URL=... INGEST_URL=... python3 scripts/test_endpoints.py`
- Common errors:
  - `401/403` → missing/invalid API key (chat) or signature (ingest) or blocked by Access.
  - `422` → invalid JSON shape.
  - `429`/`5xx` → check Cloudflare/Access policies and app logs.
- Verify health: `curl -s "${CHAT_URL}/health"` and `curl -s "${INGEST_URL}/health"`.
