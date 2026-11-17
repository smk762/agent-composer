#!/usr/bin/env python3
"""
Smoke-test streaming support in a running rag-chat (manual script, not a pytest module:
the checks are named ``check_*`` and return a bool so ``main`` can run all of them).

    CHAT_URL=http://127.0.0.1:9150 RAG_API_KEY=... python3 scripts/test_streaming.py
"""

import json
import os
import sys
import time
import urllib.request
from typing import Any


CHAT_URL = os.environ.get("CHAT_URL", "http://127.0.0.1:9150") + "/chat"
API_KEY = os.environ.get("RAG_API_KEY") or os.environ.get("CHAT_API_KEY", "")


def _auth_headers() -> dict[str, str]:
    """Return auth headers if an API key is configured."""
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    return headers


def _delta(chunk: dict[str, Any]) -> dict[str, Any]:
    """The first choice's delta; ``{}`` for chunks without choices (e.g. usage-only)."""
    choices = chunk.get('choices') or [{}]
    return choices[0].get('delta') or {}


def _merge_tool_calls(acc: dict[int, dict[str, Any]], fragments: list[dict[str, Any]]) -> None:
    """Fold streamed tool-call fragments into ``acc`` by index. Only the first fragment of a
    call carries its id/name; later ones carry more ``arguments`` to append."""
    for pos, tc in enumerate(fragments):
        idx = tc.get('index', pos)
        call = acc.setdefault(idx, {'id': None, 'name': '', 'arguments': ''})
        fn = tc.get('function') or {}
        if tc.get('id'):
            call['id'] = tc['id']
        if fn.get('name'):
            call['name'] += fn['name']
        call['arguments'] += fn.get('arguments') or ''


def check_basic_streaming() -> bool:
    """Test basic streaming without tools."""
    print("\n=== Test 1: Basic Streaming ===")
    
    payload = {
        "messages": [{"role": "user", "content": "Count from 1 to 5"}],
        "stream": True
    }
    
    body = json.dumps(payload).encode('utf-8')
    headers = _auth_headers()
    headers['Accept'] = 'text/event-stream'
    req = urllib.request.Request(
        CHAT_URL,
        data=body,
        headers=headers,
        method='POST'
    )
    
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            # Check headers
            content_type = response.headers.get('Content-Type', '')
            if 'text/event-stream' not in content_type:
                print(f"FAIL: Wrong content type: {content_type}")
                return False
            
            print("Streaming response:")
            accumulated = ""
            chunks_received = 0
            
            for line in response:
                line_str = line.decode('utf-8').strip()
                if not line_str:
                    continue
                
                if line_str.startswith('data: '):
                    data_str = line_str[6:]
                    
                    if data_str == '[DONE]':
                        print("\n[Stream complete]")
                        break
                    
                    try:
                        chunk = json.loads(data_str)
                        if 'error' in chunk:
                            print(f"FAIL: Error in stream: {chunk['error']}")
                            return False
                        
                        content = _delta(chunk).get('content')
                        if content:
                            print(content, end='', flush=True)
                            accumulated += content
                            chunks_received += 1
                    except json.JSONDecodeError as e:
                        print(f"\nWARN: JSON decode error: {e}")
            
            print(f"\n✓ Received {chunks_received} content chunks")
            print(f"✓ Accumulated content: {len(accumulated)} chars")
            
            if chunks_received > 0:
                print("OK: Streaming works!")
                return True
            else:
                print("FAIL: No content received")
                return False
    
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {e}")
        return False


def check_streaming_with_tools() -> bool:
    """Test streaming with tools."""
    print("\n=== Test 2: Streaming with Tools ===")
    
    payload = {
        "messages": [{"role": "user", "content": "Search for test"}],
        "stream": True,
        "tools": [{
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search for something",
                "parameters": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"]
                }
            }
        }]
    }
    
    body = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        CHAT_URL,
        data=body,
        headers=_auth_headers(),
        method='POST'
    )
    
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            print("Streaming with tools:")
            accumulated_content = ""
            tool_calls: dict[int, dict[str, Any]] = {}
            
            for line in response:
                line_str = line.decode('utf-8').strip()
                if not line_str or not line_str.startswith('data: '):
                    continue
                
                data_str = line_str[6:]
                if data_str == '[DONE]':
                    break
                
                try:
                    chunk = json.loads(data_str)
                    delta = _delta(chunk)
                    
                    content = delta.get('content')
                    if content:
                        print(content, end='', flush=True)
                        accumulated_content += content
                    
                    if delta.get('tool_calls'):
                        _merge_tool_calls(tool_calls, delta['tool_calls'])
                
                except json.JSONDecodeError:
                    pass
            
            print("\n")
            if tool_calls:
                print(f"✓ Tool calls received: {len(tool_calls)}")
                for _, call in sorted(tool_calls.items()):
                    print(f"  - {call['name']}: {call['arguments']}")
                if not all(call['name'] for call in tool_calls.values()):
                    print("FAIL: Tool call without a function name")
                    return False
            if accumulated_content or tool_calls:
                print("OK: Streaming with tools works!")
                return True
            else:
                print("FAIL: No content or tool calls received")
                return False
    
    except Exception as e:
        print(f"FAIL: {e}")
        return False


def check_non_streaming_still_works() -> bool:
    """Verify non-streaming still works."""
    print("\n=== Test 3: Non-Streaming (Backward Compat) ===")
    
    payload = {
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False
    }
    
    body = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        CHAT_URL,
        data=body,
        headers=_auth_headers(),
        method='POST'
    )
    
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode('utf-8'))
            
            if data.get('choices'):
                message = data['choices'][0].get('message') or {}
                content = message.get('content') or ''
                if not content and not message.get('tool_calls'):
                    print("FAIL: Empty response")
                    return False
                print(f"Response: {content[:100]}")
                print("OK: Non-streaming works!")
                return True
            else:
                print("FAIL: Invalid response format")
                return False
    
    except Exception as e:
        print(f"FAIL: {e}")
        return False


def main():
    print("="*70)
    print("Testing RAG-Chat Streaming Support")
    print("="*70)
    print(f"Endpoint: {CHAT_URL}")
    
    ok = True
    ok &= check_non_streaming_still_works()  # Test backward compat first
    ok &= check_basic_streaming()
    ok &= check_streaming_with_tools()
    
    print("\n" + "="*70)
    if ok:
        print("✅ All streaming tests passed!")
    else:
        print("❌ Some tests failed")
        sys.exit(1)
    print("="*70)


if __name__ == "__main__":
    main()
