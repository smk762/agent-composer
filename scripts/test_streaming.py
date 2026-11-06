#!/usr/bin/env python3
"""
Test streaming support in rag-chat.
"""

import json
import os
import sys
import time
import urllib.request
from typing import Any


CHAT_URL = os.environ.get("CHAT_URL", "http://127.0.0.1:9150") + "/chat"
API_KEY = os.environ.get("RAG_API_KEY", "")


def _auth_headers() -> dict[str, str]:
    """Return auth headers if an API key is configured."""
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    return headers


def test_basic_streaming():
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
                        
                        delta = chunk.get('choices', [{}])[0].get('delta', {})
                        if 'content' in delta:
                            print(delta['content'], end='', flush=True)
                            accumulated += delta['content']
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


def test_streaming_with_tools():
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
            tool_calls_received = None
            
            for line in response:
                line_str = line.decode('utf-8').strip()
                if not line_str or not line_str.startswith('data: '):
                    continue
                
                data_str = line_str[6:]
                if data_str == '[DONE]':
                    break
                
                try:
                    chunk = json.loads(data_str)
                    delta = chunk.get('choices', [{}])[0].get('delta', {})
                    
                    if 'content' in delta:
                        print(delta['content'], end='', flush=True)
                        accumulated_content += delta['content']
                    
                    if 'tool_calls' in delta:
                        tool_calls_received = delta['tool_calls']
                        print(f"\n✓ Tool calls received: {len(tool_calls_received)}")
                        for tc in tool_calls_received:
                            print(f"  - {tc['function']['name']}: {tc['function']['arguments']}")
                
                except json.JSONDecodeError:
                    pass
            
            print("\n")
            if accumulated_content or tool_calls_received:
                print("OK: Streaming with tools works!")
                return True
            else:
                print("FAIL: No content or tool calls received")
                return False
    
    except Exception as e:
        print(f"FAIL: {e}")
        return False


def test_non_streaming_still_works():
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
            
            if 'choices' in data and data['choices']:
                message = data['choices'][0].get('message', {})
                content = message.get('content', '')
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
    ok &= test_non_streaming_still_works()  # Test backward compat first
    ok &= test_basic_streaming()
    ok &= test_streaming_with_tools()
    
    print("\n" + "="*70)
    if ok:
        print("✅ All streaming tests passed!")
    else:
        print("❌ Some tests failed")
        sys.exit(1)
    print("="*70)


if __name__ == "__main__":
    main()
