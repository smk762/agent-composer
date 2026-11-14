"""Retrieval for agents and tools: ``POST /search`` over the text collection (``QDRANT_COLLECTION``,
default ``audit_docs``), the same passages /chat injects, returned as JSON with titles and links.

Filter by ``repo`` (tag ``repo:<name>``), ``kind`` (``kind:wiki``, ``kind:agent_pack``, ...) or
any explicit ``tags``; a hit must carry all of them. Auth is the same as /chat (bearer API key,
Cloudflare Access, or the dev bypass).
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app import retrieval
from app.auth import require_user
from app.config import QDRANT_COLLECTION
from app.models.schemas import AuthContext
from app.routers.chat import ollama_embed

router = APIRouter()

MAX_TOP_K = 20


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(5, ge=1, le=MAX_TOP_K)
    repo: Optional[str] = Field(None, max_length=200, description="Shorthand for tag repo:<name>")
    kind: Optional[str] = Field(None, max_length=100, description="Shorthand for tag kind:<kind>, e.g. wiki")
    tags: List[str] = Field(default_factory=list, max_length=10, description="Tags every hit must carry")


class SearchHit(BaseModel):
    score: float
    doc_id: Optional[str] = None
    chunk_index: Optional[int] = None
    title: Optional[str] = None
    url: Optional[str] = None
    path: Optional[str] = None
    source: Optional[str] = None
    tags: List[str] = []
    text: str


class SearchResponse(BaseModel):
    collection: str
    tags: List[str]
    hits: List[SearchHit]


@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest, auth: AuthContext = Depends(require_user)) -> SearchResponse:
    query = req.query.strip()
    if not query:
        raise HTTPException(422, "query is empty")
    tags = retrieval.search_tags(req.repo, req.kind, req.tags)
    try:
        hits = await retrieval.search(query, embed=ollama_embed, top_k=req.top_k, tags=tags)
    except HTTPException:
        raise
    except Exception as e:  # Qdrant down, collection missing, ...
        raise HTTPException(502, f"Search failed: {e}")
    return SearchResponse(collection=QDRANT_COLLECTION, tags=tags, hits=hits)
