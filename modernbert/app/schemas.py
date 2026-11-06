from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ClassifyRequest(BaseModel):
    text: str
    context: Optional[List[str]] = Field(
        default=None,
        description="Recent turns oldest-first; up to 5 are prepended with [SEP]",
    )
    top_k: Optional[int] = None
    threshold: Optional[float] = None


class ClassifyResponse(BaseModel):
    labels: List[str]
    scores: List[float]
    raw: Dict[str, float]
    task: str
    model: str
    model_state: str


class EmbedRequest(BaseModel):
    texts: List[str]


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    dim: int


class LabelUpdate(BaseModel):
    labels: List[str]


class ModelStatus(BaseModel):
    state: str
    model_id: str
    device: str
    last_used: Optional[float]
    keep_alive_gpu: float
    keep_alive_cpu: float
    labels: List[str]
    task: str
    error: Optional[str]
    gpu_capable: bool = False
    configured_device: str = "auto"
