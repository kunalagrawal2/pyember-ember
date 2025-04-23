#!/usr/bin/env python3
"""
Basic example demonstrating the use of the McpClient provider within Ember.

This script showcases various capabilities of the MCP (Model Control Protocol) integration.
The MCP provider always requires an underlying LLM model inside it.

To run:
    export OPENAI_API_KEY="your-key"  # Required as MCP provider needs a model
    uv run python src/ember/examples/integration/basic_mcp_example.py

Requires:
    - A running Python environment with Ember and MCP dependencies installed.
    - The example_mcp_server.py script available at the specified path.
    - An OpenAI API key (or other supported model API key)
"""

import asyncio
import logging
import os
import sys
import json
from typing import Optional, Dict, Any, List

# Add project root to sys.path for local development
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Ember imports
from ember.core.registry.model.base.registry.model_registry import ModelRegistry
from ember.core.registry.model.base.schemas.model_info import ModelInfo
from ember.core.registry.model.base.schemas.provider_info import ProviderInfo
from ember.core.registry.model.base.schemas.chat_schemas import ChatRequest, ChatResponse
# Import the McpClient class to ensure its @provider decorator runs
from ember.core.registry.model.providers.mcp.mcp_provider import McpClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global registry for examples to use
registry = None
mcp_model = None
wrapped_model = None

def setup_registry_and_models() -> ModelRegistry:
    """Initializes the ModelRegistry and registers models."""
    global wrapped_model, mcp_model
    
    print("\n=== Registry Initialization and MCP Model Setup ===")
    registry = ModelRegistry()

    # --- Register OpenAI Model (Required for MCP) ---
    openai_model_id = "openai:gpt-4o"
    openai_key = os.environ.get("OPENAI_API_KEY")

    if not openai_key:
        raise ValueError("OPENAI_API_KEY environment variable is required but not set")

    if not registry.is_registered(openai_model_id):
        registry.register_model(
            ModelInfo(
                id=openai_model_id,
                name="gpt-4o", # Suggestion: automatically get name from id?
                provider=ProviderInfo(name="OpenAI")
            )
        )
        print(f"Registered OpenAI model: {openai_model_id}")
    
    # Get the OpenAI model first, set it as the wrapped model
    wrapped_model = registry.get_model(openai_model_id)
    print(f"Retrieved OpenAI model: {openai_model_id}")

    # --- Configuration for McpClient ---
    mcp_model_id = "mcp:stdio-echo-server"
    server_script_path = os.path.join(os.path.dirname(__file__), "example_mcp_server.py")
    python_executable = sys.executable

    if not os.path.exists(server_script_path):
        logger.error(f"MCP server script not found at: {server_script_path}")
        raise FileNotFoundError(f"MCP server script not found: {server_script_path}")

    mcp_command = python_executable
    mcp_args = server_script_path

    # --- Define MCP ModelInfo ---
    mcp_model_info = ModelInfo(
        id=mcp_model_id,
        provider=ProviderInfo(
            name="MCP",
            custom_args={
                "command": mcp_command,
                "args": mcp_args,
                "env": json.dumps({"PYTHONUNBUFFERED": "1"}),
            }
        ),
    )

    if not registry.is_registered(mcp_model_id):
        registry.register_model(mcp_model_info)
        print(f"Registered MCP model: {mcp_model_id}")
    
    # Get the MCP model and inject the wrapped model
    #TODO Need a better way rather than injecting
    mcp_model = registry.get_model(mcp_model_id)
    mcp_model.set_wrapped_model(wrapped_model)  # Inject wrapped model during setup

    print(f"Retrieved MCP model and injected underlying model")

    print("Registry setup complete.")
    return registry

async def use_prompt_with_tool():
    """Example 1: Forwarding a request directly to the underlying model."""
    print("\n=== Example 1: Forwarding Request to Underlying Model ===")
    
    try:
        # Initialize session if needed
        if not hasattr(mcp_model, '_session') or mcp_model._session is None:
            await mcp_model.initialize_session()
        
        # Send a request that will use the underlying model
        request = ChatRequest(prompt="Echo this text: 'Hello MCP!'")
        
        print(f"Sending request to underlying model: '{request.prompt}'")
        response = await mcp_model.forward(request)
        
        print("-" * 30)
        print(f"Response via underlying model:")
        print(f"  Data: {response.data[:200]}..." if len(response.data) > 200 else f"  Data: {response.data}")
        if hasattr(response, 'usage') and response.usage:
            print(f"  Usage: {response.usage}")
        print("-" * 30)
        
    except Exception as e:
        logger.error(f"Error in forward request example: {e}", exc_info=True)

async def use_prompt_without_tool():
    """Example 1: Forwarding a request directly to the underlying model."""
    print("\n=== Example 1: Forwarding Request to Underlying Model ===")
    
    try:
        # Initialize session if needed
        if not hasattr(mcp_model, '_session') or mcp_model._session is None:
            await mcp_model.initialize_session()
        
        # Send a request that will use the underlying model, which shouldn't need a tool call
        request = ChatRequest(prompt="Tell me about UC Berkeley")
        
        print(f"Sending request to underlying model: '{request.prompt}'")
        response = await mcp_model.forward(request)
        
        print("-" * 30)
        print(f"Response via underlying model:")
        print(f"  Data: {response.data[:200]}..." if len(response.data) > 200 else f"  Data: {response.data}")
        if hasattr(response, 'usage') and response.usage:
            print(f"  Usage: {response.usage}")
        print("-" * 30)
        
    except Exception as e:
        logger.error(f"Error in forward request example: {e}", exc_info=True)

async def use_tool():
    """Example 2: Using tools provided by the MCP server."""
    print("\n=== Example 2: Using MCP Server Tools ===")
    
    try:
        # Initialize the session if not already done
        if not hasattr(mcp_model, '_session') or mcp_model._session is None:
            await mcp_model.initialize_session()
        
        # Check if tools capability is available
        if not hasattr(mcp_model._server_capabilities, 'tools') or not mcp_model._server_capabilities.tools:
            print("Server doesn't support tools API. Skipping tools example.")
            return
            
        # List available tools
        print("Listing available tools...")
        tools = await mcp_model.list_tools()
        print(f"Found {len(tools)} tools: {[tool.name for tool in tools]}")
        
        # Call the echo tool
        tool_name = "echo_tool"
        arguments = {"message": "This is a tool test"}
        
        print(f"Calling tool '{tool_name}' with arguments: {arguments}")
        result = await mcp_model.call_tool(tool_name, arguments)
        
        print("-" * 30)
        print(f"Tool result: {result}")
        print("-" * 30)
        
    except Exception as e:
        logger.error(f"Error in tool usage example: {e}", exc_info=True)


async def access_resource():
    """Example 3: Accessing resources from the MCP server."""
    print("\n=== Example 3: Accessing MCP Server Resources ===")
    
    try:
        # Initialize the session if not already done
        if not hasattr(mcp_model, '_session') or mcp_model._session is None:
            await mcp_model.initialize_session()
        
        # Check if resources capability is available
        if not hasattr(mcp_model._server_capabilities, 'resources') or not mcp_model._server_capabilities.resources:
            print("Server doesn't support resources API. Skipping resources example.")
            return

        # List available resources
        # There should be zero available resources in this example because we have a resource template
        print("Listing available resources...")
        resources = await mcp_model.list_resources()
        print(f"Found {len(resources)} resources: {[resource.name for resource in resources]}")
            
        # Access a resource
        resource_uri = "echo://resource-test"
        
        print(f"Requesting resource: {resource_uri}")
        contents = await mcp_model.read_resource(resource_uri)
        
        print("-" * 30)
        print(f"Resource content: {contents}")
        print("-" * 30)
        
    except Exception as e:
        logger.error(f"Error in resource access example: {e}", exc_info=True)



async def main():
    """Main function to run all examples."""
    global registry, mcp_model, wrapped_model
    
    print("Running MCP Integration Examples...")
    
    try:
        # Setup registry and get models - this already handles wrapping
        registry = setup_registry_and_models()
        
        # Run examples
        logger.info("Integration Examples")
        logger.info("Running Example 1: Prompt Usage with tool call")
        await use_prompt_with_tool()
        logger.info("Running Example 2: Prompt Usage (no tool call)")
        await use_prompt_without_tool()
        logger.info("Direct Examples")
        logger.info("Running Example 2: Tool Usage")
        await use_tool()
        logger.info("Running Example 3: Resource Access")
        await access_resource()
        
            
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
    finally:
        # Cleanup - terminate MCP model to close subprocess
        if mcp_model:
            try:
                logger.info(f"Terminating MCP model")
                await mcp_model.terminate()
                logger.info(f"MCP model terminated successfully")
            except Exception as term_error:
                logger.error(f"Error during model termination: {term_error}")

    logger.info("All examples completed")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Examples interrupted by user")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
