#!/usr/bin/env python3
"""
Example client demonstrating tool-calling workflow with rag-chat.

This script shows how an external agent would integrate with rag-chat
to perform tool calling in a loop until the model returns a final answer.

Dependency-free (stdlib only).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List


CHAT_URL = os.getenv("CHAT_URL", "http://127.0.0.1:9150") + "/chat"
API_KEY = os.getenv("RAG_API_KEY", "")


# Define available tools
TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Search the product catalog by name or keyword",
            "parameters": {
                "type": "object",
                "properties": {
                    "q": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Maximum number of results (default: 5)"},
                },
                "required": ["q"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_product_details",
            "description": "Get detailed information about a specific product",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "string", "description": "Product ID"},
                },
                "required": ["product_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_inventory",
            "description": "Check inventory level for a product",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "string", "description": "Product ID"},
                },
                "required": ["product_id"],
            },
        },
    },
]


def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> str:
    """
    Simulate tool execution. In a real agent, this would call actual functions.
    """
    print(f"  Executing tool: {tool_name}")
    print(f"  Arguments: {json.dumps(arguments, indent=2)}")

    if tool_name == "search_products":
        query = arguments.get("q", "")
        limit = arguments.get("limit", 5)
        results = [
            {"id": "prod-001", "name": f"Bacon Strips ({query})", "price": 5.99, "in_stock": True},
            {"id": "prod-002", "name": f"Bacon Bits ({query})", "price": 3.49, "in_stock": True},
            {"id": "prod-003", "name": f"Turkey Bacon ({query})", "price": 6.99, "in_stock": False},
        ][:limit]
        return json.dumps({"results": results, "total": len(results)})

    if tool_name == "get_product_details":
        product_id = arguments.get("product_id")
        return json.dumps({
            "id": product_id,
            "name": "Premium Bacon Strips",
            "price": 5.99,
            "weight": "16 oz",
            "brand": "Farm Fresh",
            "description": "Thick-cut hickory smoked bacon",
        })

    if tool_name == "check_inventory":
        product_id = arguments.get("product_id")
        return json.dumps({
            "product_id": product_id,
            "quantity": 42,
            "location": "Aisle 7",
            "last_restocked": "2024-02-10",
        })

    return json.dumps({"error": f"Unknown tool: {tool_name}"})


def call_chat(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    """Send a chat request to rag-chat (stdlib only)."""
    headers: dict[str, str] = {"Content-Type": "application/json", "Accept": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    payload: Dict[str, Any] = {
        "messages": messages,
        "tools": tools or TOOLS,
        "temperature": 0.2,
        "max_tokens": 1000,
    }

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(CHAT_URL, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_agent_loop(user_query: str, max_iterations: int = 5) -> None:
    """
    Run the agent tool-calling loop until the model returns a final answer.
    """
    print(f"\n{'='*70}")
    print(f"User Query: {user_query}")
    print(f"{'='*70}\n")

    messages: List[Dict[str, Any]] = [{"role": "user", "content": user_query}]

    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1} ---")

        response = call_chat(messages)

        if not response.get("choices"):
            print("ERROR: No choices in response")
            break

        assistant_msg = response["choices"][0]["message"]
        content = assistant_msg.get("content")
        tool_calls = assistant_msg.get("tool_calls")

        # If no tool calls, we have the final answer
        if not tool_calls:
            print("\nFinal Answer:")
            print(f"  {content}")
            if response.get("usage"):
                print(f"\nToken Usage: {response['usage']}")
            break

        # Model requested tool calls
        print(f"\nAssistant requested {len(tool_calls)} tool call(s):")

        # Add assistant message with tool calls to history
        messages.append({
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls,
        })

        # Execute each tool and add results to messages
        for tool_call in tool_calls:
            tool_id = tool_call["id"]
            function = tool_call["function"]
            tool_name = function["name"]

            try:
                arguments = json.loads(function["arguments"])
            except json.JSONDecodeError:
                arguments = {}

            result = execute_tool(tool_name, arguments)
            print(f"  Result: {result[:100]}...")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_id,
                "content": result,
            })
    else:
        print(f"\nWARNING: Reached max iterations ({max_iterations}) without final answer")


def main() -> None:
    """Run example queries demonstrating tool calling."""
    print("=" * 70)
    print("RAG-Chat Tool-Calling Example Client")
    print("=" * 70)
    print(f"Chat endpoint: {CHAT_URL}")
    print(f"Available tools: {len(TOOLS)}")

    # Example 1: Simple search
    run_agent_loop("What bacon products do we have?")

    # Example 2: Multi-step query
    run_agent_loop("Search for bacon products and check the inventory for the first result")

    # Example 3: Query without tools needed
    run_agent_loop("What is bacon made from?")

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
