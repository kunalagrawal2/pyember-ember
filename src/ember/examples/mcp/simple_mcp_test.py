"""
Simple MCP Test

A minimalist test of the McpProvider to verify basic functionality.
"""

import asyncio
import json
from typing import Dict, Any
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Import the run_stdio_client function directly
from ember.core.registry.model.providers.mcp.mcp_client import run_stdio_client
from ember.core.registry.model.providers.mcp.mcp_provider import create_mcp_provider

# Define the callback function at MODULE level (not inside another function)
async def do_everything(session):
    """Handle session operations."""
    print("Connected to MCP server!")
    
    # Initialize the session
    await session.initialize()
    
    # List resources
    print("\n=== Resources ===")
    try:
        resources = await asyncio.wait_for(session.list_resources(), timeout=5.0)
        for resource in resources:
            print(f"- {resource.name}: {resource.description}")
    except asyncio.TimeoutError:
        print("Resource listing timed out!")
    
    # List tools
    print("\n=== Tools ===")
    try:
        tools = await asyncio.wait_for(session.list_tools(), timeout=5.0)
        for tool in tools:
            print(f"- {tool.name}: {tool.description}")
    except asyncio.TimeoutError:
        print("Tool listing timed out!")
    
    return "Done"

# Simple test using direct connection
async def run_simple_test():
    """Run a simple direct test to the MCP server."""
    print("Starting simple direct test...")
    print(type(do_everything))
    try:
        result = await asyncio.wait_for(
            run_stdio_client(
                command="python",
                args=["-m", "ember.examples.mcp.simple_mcp_server", "--no-buffer"],
                callback=do_everything
            ),
            timeout=10.0
        )
        print(f"Result: {result}")
        return result
    except asyncio.TimeoutError:
        print("Test timed out!")
        return "Timeout"
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        return f"Error: {e}"

# Main entry point
if __name__ == "__main__":
    print("Starting MCP test...")
    asyncio.run(run_simple_test()) 