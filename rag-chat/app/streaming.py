"""
Streaming support for rag-chat.
Server-Sent Events (SSE) implementation for token-by-token responses.

stream_ollama_chat() yields StreamChunk objects (not raw SSE strings).
The caller is responsible for serializing to SSE format, which avoids
the overhead of double serialize/deserialize for accumulation.
"""

import json
import secrets
import time
from typing import AsyncIterator, Any, Dict, List, Optional

import httpx
from pydantic import BaseModel


class StreamDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[Any]] = None  # ToolCall-like dicts


class StreamChoice(BaseModel):
    delta: StreamDelta
    finish_reason: Optional[str] = None
    index: int = 0


class StreamChunk(BaseModel):
    """OpenAI-compatible streaming chunk format."""
    id: str
    created: int
    model: str
    choices: List[StreamChoice]
    conversation_id: Optional[str] = None


async def stream_ollama_chat(
    url: str,
    payload: Dict[str, Any],
    model: str,
    conversation_id: Optional[str] = None,
    client: Optional[httpx.AsyncClient] = None,
) -> AsyncIterator[StreamChunk]:
    """
    Stream chat responses from Ollama, yielding StreamChunk objects.

    Callers should serialize chunks to SSE themselves, e.g.:
        for chunk in stream_ollama_chat(...):
            yield f"data: {chunk.model_dump_json(exclude_none=True)}\\n\\n"
        yield "data: [DONE]\\n\\n"

    If *client* is provided it will be used as-is (caller manages its
    lifecycle).  Otherwise a temporary client is created per call.
    """
    payload["stream"] = True

    # All chunks in a single stream share the same id and creation timestamp,
    # consistent with the OpenAI streaming format.
    chunk_id = f"chatcmpl-{secrets.token_hex(12)}"
    created = int(time.time())

    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(timeout=120)

    try:
        async with client.stream("POST", f"{url}/api/chat", json=payload) as response:
            if response.status_code != 200:
                error_body = (await response.aread()).decode("utf-8", errors="replace")
                raise StreamError(
                    f"Ollama returned {response.status_code}: {error_body}"
                )

            # First chunk: send role
            yield StreamChunk(
                id=chunk_id,
                created=created,
                model=model,
                choices=[StreamChoice(
                    delta=StreamDelta(role="assistant"),
                    index=0,
                )],
                conversation_id=conversation_id,
            )

            # Stream tokens
            accumulated_tool_calls: List[Dict[str, Any]] = []
            async for line in response.aiter_lines():
                if not line.strip():
                    continue

                try:
                    chunk_data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                message = chunk_data.get("message", {})
                content = message.get("content")
                tool_calls = message.get("tool_calls")
                done = chunk_data.get("done", False)

                # Handle content token
                if content:
                    yield StreamChunk(
                        id=chunk_id,
                        created=created,
                        model=model,
                        choices=[StreamChoice(
                            delta=StreamDelta(content=content),
                            index=0,
                        )],
                        conversation_id=conversation_id,
                    )

                # Accumulate tool calls from Ollama
                if tool_calls:
                    accumulated_tool_calls = tool_calls

                # Final chunk
                if done:
                    if accumulated_tool_calls:
                        formatted_tool_calls = []
                        for tc in accumulated_tool_calls:
                            func = tc.get("function", {})
                            # Ollama returns arguments as a dict; OpenAI
                            # format requires a JSON string.
                            raw_args = func.get("arguments", {})
                            args_str = (
                                json.dumps(raw_args)
                                if isinstance(raw_args, dict)
                                else str(raw_args)
                            )
                            formatted_tool_calls.append({
                                "id": tc.get("id", f"call_{secrets.token_hex(12)}"),
                                "type": "function",
                                "function": {
                                    "name": func.get("name", ""),
                                    "arguments": args_str,
                                },
                            })

                        yield StreamChunk(
                            id=chunk_id,
                            created=created,
                            model=model,
                            choices=[StreamChoice(
                                delta=StreamDelta(tool_calls=formatted_tool_calls),
                                finish_reason="tool_calls",
                                index=0,
                            )],
                            conversation_id=conversation_id,
                        )
                    else:
                        # Regular completion
                        yield StreamChunk(
                            id=chunk_id,
                            created=created,
                            model=model,
                            choices=[StreamChoice(
                                delta=StreamDelta(),
                                finish_reason="stop",
                                index=0,
                            )],
                            conversation_id=conversation_id,
                        )
                    break

    except StreamError:
        raise
    except Exception as e:
        raise StreamError(f"Stream error: {e}") from e
    finally:
        if owns_client:
            await client.aclose()


class StreamError(Exception):
    """Raised when streaming from Ollama fails."""
