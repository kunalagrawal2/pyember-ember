#!/usr/bin/env python3
"""
Basic example demonstrating the use of the McpClient provider within Ember,
structured similarly to model_api_example.py.

This script connects to an MCP server (like example_mcp_server.py) via stdio
and sends a simple chat message through it.

To run:
    # Ensure OPENAI_API_KEY is set if you intend to use OpenAI models elsewhere,
    # though this specific example doesn't directly call OpenAI.
    export OPENAI_API_KEY="your-key" 
    uv run python src/ember/examples/integration/basic_mcp_example.py

Requires:
    - A running Python environment with Ember and MCP dependencies installed.
    - The example_mcp_server.py script available at the specified path.
"""

import asyncio
import logging
import os
import sys
import json # For potential future use with structured prompts/tool calls
from typing import Tuple

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
# Import OpenAI provider if needed for registration type hints (optional)
# from ember.core.registry.model.providers.openai.openai_provider import OpenAIProvider

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_registry_and_models() -> ModelRegistry:
    """Initializes the ModelRegistry and registers the MCP and OpenAI models."""
    print("\n=== Registry Initialization and MCP Model Setup ===")
    registry = ModelRegistry()

    # --- Register OpenAI Model (Prerequisite for Manual Injection) ---
    openai_model_id = "openai:gpt-4o" # Or any other valid OpenAI model
    openai_key = os.environ.get("OPENAI_API_KEY")

    if not openai_key:
        print(f"Warning: OPENAI_API_KEY not set. Cannot register '{openai_model_id}'.")
    elif not registry.is_registered(openai_model_id):
        registry.register_model(
            ModelInfo(
                id=openai_model_id,
                provider=ProviderInfo(name="OpenAI") # API key will be picked up from env by default
            )
        )
        print(f"Registered OpenAI model: {openai_model_id}")
    else:
         print(f"Model {openai_model_id} already registered.")


    # --- Configuration for McpClient ---
    mcp_model_id = "mcp:stdio-echo-server"
    server_script_path = os.path.join(os.path.dirname(__file__), "example_mcp_server.py")
    python_executable = sys.executable

    logger.debug(f"server script path: {server_script_path}")
    if not os.path.exists(server_script_path):
        logger.error(f"MCP server script not found at: {server_script_path}")
        raise FileNotFoundError(f"MCP server script not found: {server_script_path}")

    mcp_command = python_executable
    mcp_args = server_script_path

    # --- Define MCP ModelInfo (WITHOUT wrapping info) ---
    # We will inject the OpenAI model manually later
    mcp_model_info = ModelInfo(
        id=mcp_model_id,
        provider=ProviderInfo(
            name="MCP", # Matches the McpClient's @provider decorator
            # NO wrapped_model_id here - we modify the instance later
            custom_args={
                "command": mcp_command,
                "args": mcp_args, # Pass as string or list
                # Add "env": {'VAR': 'value'} or json string if needed
            }
        ),
    )

    if not registry.is_registered(mcp_model_id):
        registry.register_model(mcp_model_info)
        print(f"Registered MCP model: {mcp_model_id} (underlying model to be injected manually)")
    else:
        print(f"Model {mcp_model_id} already registered.")


    print("Registry setup complete.")
    return registry


async def run_mcp_chat_example(mcp_model: McpClient):
    """Demonstrates sending a chat message via the provided McpClient instance."""
    print("\n=== MCP Chat Example ===")
    model_id = mcp_model.model_info.id # Get ID from the instance if needed for logging

    # No need to call registry.get_model here

    # Add a check to ensure the model was injected (still useful)
    if not hasattr(mcp_model, '_model') or mcp_model._model is None:
         logger.error(f"MCP model '{model_id}' received, but its internal '_model' was not set. Injection likely failed.")
         return

    logger.info(f"Using provided MCP model: {model_id}. Internal model: {mcp_model._model.model_info.id}")


    # --- Run a Simple Chat Request ---
    request = ChatRequest(prompt="Hello MCP server, this is a test via manual injection.")

    logger.info(f"Sending standard chat request to {model_id}: '{request.prompt}'")
    try:
        # Forward call now goes through McpClient -> MCP Server -> McpClient.handle_sampling_message -> Injected OpenAI Model
        response = await mcp_model.forward(request) # Await the async call

        print("-" * 30)
        print(f"Response from {model_id} (via {mcp_model._model.model_info.id}):")
        print(f"  Data: {response.data}")
        print(f"  Usage: {response.usage}")
        print(f"  Raw Output: {response.raw_output}")
        print("-" * 30)

    except Exception as e:
        logger.error(f"Error during chat request via {model_id}: {e}", exc_info=True)


async def main():
    """Sets up registry, initializes Ember, manually injects model, and runs the MCP example."""
    openai_key_present = bool(os.environ.get("OPENAI_API_KEY"))
    if not openai_key_present:
        print("Warning: OPENAI_API_KEY not set. Needed for registering OpenAI and running the MCP example.")
        # return # Optional: exit if key is missing

    print("Running Basic MCP Integration Example (with manual model injection)...")
    openai_model_id = "openai:gpt-4o"
    mcp_model_id = "mcp:stdio-echo-server"
    registry = None
    mcp_model = None # Initialize to None for finally block
    openai_model = None # Initialize to None

    try:
        # 1. Setup Registry and Models
        registry = setup_registry_and_models()

        # Check if prerequisite models were registered
        if not openai_key_present or not registry.is_registered(openai_model_id):
             logger.warning(f"OpenAI Model '{openai_model_id}' was not registered (likely missing API key). Cannot proceed.")
             return
        if not registry.is_registered(mcp_model_id):
             logger.warning(f"MCP Model '{mcp_model_id}' was not registered. Cannot proceed.")
             return # Exit if MCP model isn't available

        # 2. Get both model instances
        openai_model = registry.get_model(openai_model_id)
        mcp_model = registry.get_model(mcp_model_id)

        # 3. Manually Inject OpenAI model
        logger.info(f"Manually injecting OpenAI model ({openai_model_id}) into MCP model ({mcp_model_id}) instance.")
        mcp_model._model = openai_model
        logger.info(f"Injection complete. McpClient instance now references: {mcp_model._model}")


        # 4. Run the MCP chat example
        # CHANGE: Pass the prepared mcp_model instance directly
        await run_mcp_chat_example(mcp_model)

    except FileNotFoundError as e:
         # Catch specific setup errors if needed
         logger.error(f"Setup failed: {e}")
    except Exception as e:
         logger.exception(f"An error occurred during the main execution: {e}")
    finally:
        # Ensure termination happens even if errors occur during setup or run
        # Terminate the MCP model, which should handle its subprocess
        if mcp_model:
            logger.info(f"Ensuring termination of model: {mcp_model_id}")
            try:
                await mcp_model.terminate() # Await async call
                logger.info(f"Model {mcp_model_id} terminated successfully.")
            except Exception as term_error:
                logger.error(f"Error during model termination for {mcp_model_id}: {term_error}", exc_info=True)
        # Note: We don't typically need to explicitly terminate the OpenAI model unless it holds persistent resources.

    logger.info("Example finished.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FileNotFoundError as e:
        # This might be caught in main now, but keep for safety
        logger.error(f"Missing file: {e}")
    except KeyboardInterrupt:
        logger.info("Example interrupted by user.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
