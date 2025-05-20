#!/usr/bin/env python3
"""
Basic example demonstrating the use of the McpClient utility within Ember.

This script showcases various capabilities of the MCP (Model Control Protocol) integration.
The MCP client requires an underlying LLM model to be provided during initialization.

To run:
    export OPENAI_API_KEY="your-key" in .env file # Required as MCP client needs a model
    uv run python src/ember/examples/integration/basic_mcp_example.py

Requires:
    - A running Python environment with Ember and MCP dependencies installed.
    - The example_mcp_server.py script available at the specified path.
    - An OpenAI API key (or other supported model API key)
"""

import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

# Add project root to sys.path for local development
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Ember imports
from ember.core.registry.model.base.registry.model_registry import ModelRegistry
from ember.core.registry.model.base.schemas.chat_schemas import (
    ChatRequest,
    ChatResponse,
)
from ember.core.registry.model.base.schemas.model_info import ModelInfo
from ember.core.registry.model.base.schemas.provider_info import ProviderInfo

# Import the McpClient class from utils
from ember.core.utils.mcp.client import McpClient

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global registry and client for examples to use
registry = None
mcp_client = None
wrapped_model = None


def setup_registry_and_client() -> ModelRegistry:
    """Initializes the ModelRegistry and sets up the MCP client."""
    global wrapped_model, mcp_client

    print("\n=== Registry Initialization and MCP Client Setup ===")
    registry = ModelRegistry()

    # --- Register OpenAI Model (Required for MCP) ---
    openai_model_id = "openai:gpt-4o"
    openai_key = os.environ.get("OPENAI_API_KEY")

    if not openai_key:
        raise ValueError("OPENAI_API_KEY environment variable is required but not set")

    if not registry.is_registered(openai_model_id):
        registry.register_model(
            ModelInfo(
                id=openai_model_id, name="gpt-4o", provider=ProviderInfo(name="OpenAI")
            )
        )
        print(f"Registered OpenAI model: {openai_model_id}")

    # Get the OpenAI model to use as the wrapped model
    wrapped_model = registry.get_model(openai_model_id)
    print(f"Retrieved OpenAI model: {openai_model_id}")

    # --- Setup MCP Client ---
    server_script_path = os.path.join(
        os.path.dirname(__file__), "example_mcp_server.py"
    )
    python_executable = sys.executable

    if not os.path.exists(server_script_path):
        logger.error(f"MCP server script not found at: {server_script_path}")
        raise FileNotFoundError(f"MCP server script not found: {server_script_path}")

    # Create MCP client with the wrapped model
    mcp_client = McpClient(
        model=wrapped_model,
        server_command=python_executable,
        server_args=[server_script_path],
        server_env={"PYTHONUNBUFFERED": "1"},
        logger=logger,
    )
    print("Created MCP client with underlying model")

    print("Setup complete.")
    return registry


async def request_with_tool():
    """Example 1: Making a request that will intelligently use tools."""
    print("\n=== Example 1: Request with Tool Usage ===")

    try:
        # Ensure client is connected
        if not mcp_client._session:
            await mcp_client.connect()

        # Send a request that should trigger tool usage
        request = ChatRequest(prompt="Echo this text: 'Hello MCP!'")

        print(f"Sending request that may use tools: '{request.prompt}'")
        response = await mcp_client.forward(request)

        print("-" * 30)
        print(f"Response:")
        print(
            f"  Data: {response.data[:200]}..."
            if len(response.data) > 200
            else f"  Data: {response.data}"
        )
        if hasattr(response, "usage") and response.usage:
            print(f"  Usage: {response.usage}")
        print("-" * 30)

    except Exception as e:
        logger.error(f"Error in tool request example: {e}", exc_info=True)


async def request_without_tool():
    """Example 2: Making a request that won't use tools."""
    print("\n=== Example 2: Request without Tool Usage ===")

    try:
        # Ensure client is connected
        if not mcp_client._session:
            await mcp_client.connect()

        # Send a request that shouldn't need tools
        request = ChatRequest(prompt="Tell me about UC Berkeley")

        print(f"Sending request that shouldn't use tools: '{request.prompt}'")
        response = await mcp_client.forward(request)

        print("-" * 30)
        print(f"Response:")
        print(
            f"  Data: {response.data[:200]}..."
            if len(response.data) > 200
            else f"  Data: {response.data}"
        )
        if hasattr(response, "usage") and response.usage:
            print(f"  Usage: {response.usage}")
        print("-" * 30)

    except Exception as e:
        logger.error(f"Error in non-tool request example: {e}", exc_info=True)


async def use_tool_directly():
    """Example 3: Using tools directly through the MCP client."""
    print("\n=== Example 3: Direct Tool Usage ===")

    try:
        # Ensure client is connected
        if not mcp_client._session:
            await mcp_client.connect()

        # List available tools
        print("Listing available tools...")
        tools = await mcp_client.list_tools()
        print(f"Found {len(tools)} tools: {[tool.name for tool in tools]}")

        # Call the echo tool directly
        tool_name = "echo_tool"
        arguments = {"message": "This is a direct tool test"}

        print(f"Calling tool '{tool_name}' with arguments: {arguments}")
        result = await mcp_client.call_tool(tool_name, arguments)

        print("-" * 30)
        print(f"Tool result: {result}")
        print("-" * 30)

    except Exception as e:
        logger.error(f"Error in tool usage example: {e}", exc_info=True)


async def access_resource():
    """Example 4: Demonstrating resource API unsupport.

    This example shows that the server doesn't support the resources API.
    The example_mcp_server.py has a resource template but doesn't implement
    the full resources API capability.
    """
    print("\n=== Example 4: Resource API Support Check ===")

    try:
        # Ensure client is connected
        if not mcp_client._session:
            await mcp_client.connect()

        # List available resources - this should fail gracefully
        print("Attempting to list resources...")
        try:
            resources = await mcp_client.list_resources()
            print(
                f"Found {len(resources)} resources: {[resource.uri for resource in resources]}"
            )
        except Exception as e:
            print(f"Expected error listing resources: {str(e)}")
            print(
                "This is expected because the server doesn't support the resources API"
            )
            return

        # If we somehow got here (we shouldn't), try to access a resource
        resource_uri = "echo://resource-test"
        print(f"\nAttempting to access resource: {resource_uri}")
        try:
            contents = await mcp_client.read_resource(resource_uri)
            print(f"Resource content: {contents}")
        except Exception as e:
            print(f"Expected error reading resource: {str(e)}")
            print(
                "This is expected because the server doesn't support the resources API"
            )

    except Exception as e:
        logger.error(f"Unexpected error in resource example: {e}", exc_info=True)
        print("This was an unexpected error, not related to API support")


async def main():
    """Main function to run all examples."""
    global registry, mcp_client, wrapped_model

    print("Running MCP Integration Examples...")

    try:
        # Setup registry and client
        registry = setup_registry_and_client()

        # Run examples
        logger.info("Integration Examples")
        logger.info("Running Example 1: MCP Request with intelligent tool call")
        await request_with_tool()
        logger.info("Running Example 2: MCP Request (should infer no tool call)")
        await request_without_tool()
        logger.info("Direct Examples")
        logger.info("Running Example 3: Tool Usage")
        await use_tool_directly()
        logger.info("Running Example 4: Resource Access")
        await access_resource()

    except Exception as e:
        logger.exception(f"An error occurred: {e}")
    finally:
        # Cleanup - terminate MCP client
        if mcp_client:
            try:
                logger.info("Terminating MCP client")
                await mcp_client.terminate()
                logger.info("MCP client terminated successfully")
            except Exception as term_error:
                logger.error(f"Error during client termination: {term_error}")

    logger.info("All examples completed")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Examples interrupted by user")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
