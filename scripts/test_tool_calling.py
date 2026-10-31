#!/usr/bin/env python3
"""
Test script for tool-calling support in rag-chat.
Tests the OpenAI-compatible format with tools and tool_calls.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Tuple


CHAT_BASE = os.environ.get("CHAT_URL", "http://127.0.0.1:9150")
API_KEY = os.environ.get("RAG_API_KEY", "")


def _auth_headers() -> dict[str, str]:
    """Return auth headers if an API key is configured."""
    hdrs: dict[str, str] = {"Accept": "application/json", "Content-Type": "application/json"}
    if API_KEY:
        hdrs["Authorization"] = f"Bearer {API_KEY}"
    return hdrs


def _request(
    url: str, method: str = "GET", body: bytes | None = None, headers: dict[str, str] | None = None
) -> Tuple[int, Any]:
    req = urllib.request.Request(url, data=body, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
            ctype = resp.headers.get("Content-Type", "")
            if "application/json" in ctype:
                return resp.status, json.loads(raw.decode("utf-8") or "{}")
            return resp.status, raw.decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", errors="replace")
        return e.code, msg
    except Exception as e:
        return 0, str(e)


def post_json(url: str, payload: dict[str, Any]) -> Tuple[int, Any]:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return _request(url, "POST", body, _auth_headers())


def test_basic_chat() -> bool:
    """Test basic chat without tools."""
    print("\n=== Test 1: Basic chat (no tools) ===")
    payload = {"messages": [{"role": "user", "content": "Say hello in one word."}]}
    status, data = post_json(f"{CHAT_BASE}/chat", payload)
    
    if status != 200:
        print(f"FAIL: Status {status}")
        print(f"Response: {data}")
        return False
    
    # Check OpenAI-compatible format
    if "choices" not in data:
        print("FAIL: Missing 'choices' field")
        print(f"Response: {data}")
        return False
    
    if not data["choices"]:
        print("FAIL: Empty choices array")
        return False
    
    message = data["choices"][0].get("message", {})
    if "content" not in message:
        print("FAIL: Missing content in message")
        return False
    
    print(f"OK: Got response: {message['content'][:50]}...")
    print(f"Model: {data.get('model', 'unknown')}")
    if data.get("usage"):
        print(f"Usage: {data['usage']}")
    return True


def test_tool_calling() -> bool:
    """Test chat with tools - model should recognize tool availability."""
    print("\n=== Test 2: Chat with tools parameter ===")
    
    payload = {
        "messages": [
            {"role": "user", "content": "Search for bacon products"}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "search_products",
                    "description": "Search the product catalog by name or keyword",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "q": {"type": "string", "description": "Search query"},
                            "limit": {"type": "integer", "description": "Max results"}
                        },
                        "required": ["q"]
                    }
                }
            }
        ],
        "temperature": 0.2,
        "max_tokens": 500
    }
    
    status, data = post_json(f"{CHAT_BASE}/chat", payload)
    
    if status != 200:
        print(f"FAIL: Status {status}")
        print(f"Response: {data}")
        return False
    
    if "choices" not in data:
        print("FAIL: Missing 'choices' field")
        return False
    
    message = data["choices"][0].get("message", {})
    
    # Check if model used tools or returned content
    has_tool_calls = message.get("tool_calls") is not None
    has_content = message.get("content") is not None
    
    if has_tool_calls:
        print(f"OK: Model requested tool calls: {len(message['tool_calls'])} call(s)")
        for tc in message["tool_calls"]:
            func = tc.get("function", {})
            print(f"  - {func.get('name')}: {func.get('arguments')}")
    elif has_content:
        print(f"OK: Model returned direct answer (no tool call)")
        print(f"Content: {message['content'][:100]}...")
    else:
        print("FAIL: No tool_calls and no content in response")
        return False
    
    return True


def test_tool_response() -> bool:
    """Test multi-turn conversation with tool response."""
    print("\n=== Test 3: Multi-turn with tool message ===")
    
    # First request with tools
    payload1 = {
        "messages": [
            {"role": "user", "content": "What products contain bacon?"}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "search_products",
                    "description": "Search the product catalog",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "q": {"type": "string"}
                        },
                        "required": ["q"]
                    }
                }
            }
        ]
    }
    
    status, data = post_json(f"{CHAT_BASE}/chat", payload1)
    
    if status != 200:
        print(f"FAIL: First request failed with status {status}")
        return False
    
    # Simulate tool execution and send back result
    payload2 = {
        "messages": [
            {"role": "user", "content": "What products contain bacon?"},
            {"role": "assistant", "content": "Let me search for bacon products.", 
             "tool_calls": [
                 {
                     "id": "call_123",
                     "type": "function",
                     "function": {"name": "search_products", "arguments": '{"q": "bacon"}'}
                 }
             ]},
            {"role": "tool", "tool_call_id": "call_123", 
             "content": '{"results": [{"name": "Bacon Strips", "price": 5.99}]}'}
        ]
    }
    
    status, data = post_json(f"{CHAT_BASE}/chat", payload2)
    
    if status != 200:
        print(f"FAIL: Second request failed with status {status}")
        print(f"Response: {data}")
        return False
    
    if "choices" not in data:
        print("FAIL: Missing 'choices' in response")
        return False
    
    message = data["choices"][0].get("message", {})
    if message.get("content"):
        print(f"OK: Final answer received")
        print(f"Content: {message['content'][:150]}...")
        return True
    else:
        print("FAIL: No content in final response")
        return False


def main() -> None:
    print("Testing rag-chat tool-calling support...")
    print(f"Chat endpoint: {CHAT_BASE}")
    
    ok = True
    ok &= test_basic_chat()
    ok &= test_tool_calling()
    ok &= test_tool_response()
    
    if ok:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
