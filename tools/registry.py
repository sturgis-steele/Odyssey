#!/usr/bin/env python3
"""
Tool registry for Odyssey – registers functions with JSON schemas for llama.cpp tool calling.
"""

from typing import Dict, List, Callable, Any

tools_registry: Dict[str, Dict] = {}
tool_functions: Dict[str, Callable] = {}

def register_tool(name: str, description: str, parameters: Dict[str, Any] = None):
    """
    Decorator that registers a function as an LLM-callable tool.
    """
    def decorator(func: Callable) -> Callable:
        tools_registry[name] = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters or {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }
        tool_functions[name] = func
        return func
    return decorator

def get_tool_schemas() -> List[Dict]:
    """Return list of tool schemas to pass to llm.create_chat_completion(tools=...)."""
    return list(tools_registry.values())

def get_tool_function(name: str) -> Callable | None:
    """Retrieve the actual Python function by tool name."""
    return tool_functions.get(name)