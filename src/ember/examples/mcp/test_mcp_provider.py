"""
Test MCP Provider

This script demonstrates how to use the McpProvider to connect to the example MCP server.
"""

import asyncio
import json
from typing import Dict, Any

from ember.core.registry.model.providers.mcp.mcp_provider import create_mcp_provider


async def test_mcp_provider():
    """Test the MCP provider with the example server."""
    # Create the MCP provider
    provider = create_mcp_provider(
        model_id="mcp:example",
        server_type="stdio",
        command="python",
        args=["-m", "ember.examples.mcp.example_mcp_server"],
        # Make sure we're using the module path as Python would see it
    )
    
    # Use the provider as an async context manager
    async with provider:
        print("Connected to MCP server")
        
        # List available resources
        print("\n=== Resources ===")
        resources = await provider.list_resources()
        for resource in resources:
            print(f"- {resource.name}: {resource.description}")
        
        # Read a resource
        print("\n=== Reading Resource ===")
        users_data, mime_type = await provider.read_resource("data://users")
        print(f"Users data ({mime_type}):")
        print(users_data)
        
        # List available tools
        print("\n=== Tools ===")
        tools = await provider.list_tools()
        for tool in tools:
            print(f"- {tool.name}: {tool.description}")
            print(f"  Parameters: {tool.parameters}")
            print(f"  Returns: {tool.returns}")
        
        # Call a tool
        print("\n=== Calling Tool ===")
        result = await provider.call_tool("add", {"a": 5, "b": 3})
        print(f"5 + 3 = {result}")
        
        # Search for users
        search_result = await provider.call_tool("search_users", {"query": "alice"})
        print(f"Search results for 'alice': {json.dumps(search_result, indent=2)}")
        
        # Get current time
        time_result = await provider.call_tool("get_current_time", {})
        print(f"Current server time: {time_result}")
        
        # List available prompts
        print("\n=== Prompts ===")
        prompts = await provider.list_prompts()
        for prompt in prompts:
            print(f"- {prompt.name}: {prompt.description}")
        
        # Get a prompt
        print("\n=== Getting Prompt ===")
        greeting_prompt = await provider.get_prompt("greeting", {"name": "Ember User"})
        print("Greeting prompt:")
        print(greeting_prompt.text)
        
        # Test sampling
        print("\n=== Testing Sampling ===")
        response = await provider.generate_with_model(
            prompt="Hello, I'm testing the MCP provider!",
            system_prompt="You are a helpful assistant."
        )
        print("AI response:")
        print(response)
        
        # Read a file resource
        print("\n=== Reading File Resource ===")
        file_path = "mcp_test_files/sample.txt"
        file_content, file_mime = await provider.read_resource(f"file://{file_path}")
        print(f"File content ({file_mime}):")
        print(file_content)


if __name__ == "__main__":
    asyncio.run(test_mcp_provider()) 