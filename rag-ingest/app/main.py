import base64
import hashlib
import hmac
import json
import os
import time
from typing import Any, Dict, Optional

from cryptography.fernet import Fernet, InvalidToken
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from app.ingest_pipeline import ingest_docs
from app.nonce_store import NonceStore, NonceStoreConfig

app = FastAPI(title="RAG Ingestion API")

SHARED_SECRET = os.environ.get("INGEST_SHARED_SECRET", "")
REQUIRE_ENCRYPTION = os.environ.get("INGEST_REQUIRE_ENCRYPTION", "0") == "1"
NONCE_DB_PATH = os.environ.get("INGEST_NONCE_DB", "/data/nonces.db")
MAX_SKEW_S = int(os.environ.get("INGEST_MAX_SKEW_S", "300"))

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
