from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ---- Chat / Tool schemas ----

class ToolCallFunction(BaseModel):
    name: str
    arguments: str

class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: ToolCallFunction

class Msg(BaseModel):
    role: Literal["system", "user", "assistant", "tool"] = Field(...)
    content: Optional[str] = Field(None)
    tool_calls: Optional[List[ToolCall]] = Field(None)
    tool_call_id: Optional[str] = Field(None)

class FunctionDef(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]

class ToolDef(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDef

class ChatRequest(BaseModel):
    messages: List[Msg]
    model: Optional[str] = None
    conversation_id: Optional[str] = None
    tools: Optional[List[ToolDef]] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False

class MessageOut(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None

class ChoiceOut(BaseModel):
    message: MessageOut
    finish_reason: Optional[str] = None

class UsageOut(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatResponse(BaseModel):
    id: str
    created: int
    model: str
    choices: List[ChoiceOut]
    usage: Optional[UsageOut] = None
    conversation_id: Optional[str] = None

class AuthContext(BaseModel):
    user_id: str
    via: Literal["access", "apikey", "dev"]


# ---- API Key schemas ----

class ApiKeyCreate(BaseModel):
    name: Optional[str] = Field(None, description="Friendly label for the key")

class ApiKeyOut(BaseModel):
    id: str
    name: Optional[str]
    prefix: str
    created_at: Optional[datetime]
    last_used_at: Optional[datetime]
    revoked_at: Optional[datetime]

class ApiKeySecret(ApiKeyOut):
    secret: str


# ---- Conversation schemas ----

class ConversationOut(BaseModel):
    id: str
    title: Optional[str]
    summary: Optional[str]
    model: Optional[str]
    created_at: Optional[datetime]
    updated_at: Optional[datetime]

class ConversationDetail(ConversationOut):
    messages: List[Msg]

class ConversationCreate(BaseModel):
    title: Optional[str] = None
    model: Optional[str] = None


# ---- Generation schemas ----

class ImageGenerateRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    size: str = "1024x1024"
    n: int = 1
    style: Optional[str] = None

class ImageEditRequest(BaseModel):
    prompt: str
    provider: Optional[str] = None
    model: Optional[str] = None
    size: Optional[str] = None

class VideoGenerateRequest(BaseModel):
    prompt: str
    provider: Optional[str] = None
    model: Optional[str] = None
    duration: Optional[float] = None
    aspect_ratio: Optional[str] = None

class MediaAssetOut(BaseModel):
    id: str
    media_type: str
    mime_type: str
    filename: str
    url: str
    width: Optional[int] = None
    height: Optional[int] = None
    duration_seconds: Optional[float] = None
    created_at: Optional[datetime] = None

class GenerationResponse(BaseModel):
    id: str
    status: str
    media: List[MediaAssetOut]
    provider: str
    model: str
    error: Optional[str] = None

class GenerationStatusResponse(BaseModel):
    id: str
    status: str
    provider: str
    capability: str
    media: List[MediaAssetOut]
    error: Optional[str] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


# ---- Provider schemas ----

class ProviderConfigCreate(BaseModel):
    api_key: Optional[str] = None
    settings: Optional[Dict[str, Any]] = None
    enabled: bool = True

class ProviderConfigOut(BaseModel):
    id: str
    provider_name: str
    has_api_key: bool
    settings: Optional[Dict[str, Any]] = None
    enabled: bool
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class ProviderInfo(BaseModel):
    name: str
    capabilities: List[str]
    configured: bool
    enabled: bool
    models: Optional[Dict[str, List[str]]] = None
