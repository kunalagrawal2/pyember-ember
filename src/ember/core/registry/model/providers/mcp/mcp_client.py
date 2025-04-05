"""
Simplified MCP Client

A minimalist implementation following the MCP SDK examples.
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Awaitable, cast

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from mcp.types import (
    Resource, Tool, Prompt, 
    CreateMessageRequest, CreateMessageResult,
    TextContent
)


async def run_stdio_client(
    command: str,
    args: List[str],
    callback: Optional[Callable[[ClientSession], Awaitable[Any]]] = None,
    sampling_callback: Optional[Callable[[CreateMessageRequest], Awaitable[CreateMessageResult]]] = None,
) -> Any:
    """
    Run an MCP client connected to a stdio server.
    
    Args:
        command: Command to run the server
        args: Arguments for the command
        callback: Optional callback to run with the session
        sampling_callback: Optional callback for handling sampling requests
        
    Returns:
        The result of the callback, or None if no callback was provided
    """
    # # Create server parameters
    # server_params = StdioServerParameters(
    #     command=command,
    #     args=args
    # )
    
    # # Connect to the server
    # async with stdio_client(server_params) as (read, write):
    #     async with ClientSession(read, write, sampling_callback=sampling_callback) as session:
    #         # If a callback was provided, run it
    #         if callback:
    #             return await callback(session)
    #         return session  # Not normally returned, but useful for testing

    result = await asyncio.wait_for(
    run_stdio_client(
        command="python",
        args=["-m", "ember.examples.mcp.example_mcp_server"],
        callback=do_everything
    ), 
    timeout=10.0  # Add an overall timeout
)


async def run_sse_client(
    url: str,
    callback: Optional[Callable[[ClientSession], Awaitable[Any]]] = None,
    sampling_callback: Optional[Callable[[CreateMessageRequest], Awaitable[CreateMessageResult]]] = None,
) -> Any:
    """
    Run an MCP client connected to an SSE server.
    
    Args:
        url: URL of the server
        callback: Optional callback to run with the session
        sampling_callback: Optional callback for handling sampling requests
        
    Returns:
        The result of the callback, or None if no callback was provided
    """
    # Connect to the server
    async with sse_client(url) as (read, write):
        async with ClientSession(read, write, sampling_callback=sampling_callback) as session:
            # If a callback was provided, run it
            if callback:
                return await callback(session)
            return session  # Not normally returned, but useful for testing


# Simple convenience function to create a sample response
async def default_sampling_callback(request: CreateMessageRequest) -> CreateMessageResult:
    """Default callback for handling sampling requests."""
    # Extract the user message
    messages = request.messages
    user_message = "No message provided"
    
    for msg in messages:
        if msg.role == "user":
            content = msg.content
            if isinstance(content, TextContent):
                user_message = content.text
            break
    
    # Create a simple response
    return CreateMessageResult(
        role="assistant",
        content=TextContent(
            type="text",
            text=f"This is a simulated response to: '{user_message}'"
        ),
        model="mcp-model",
        stopReason="endTurn"
    )


if __name__ == "__main__":
    """Example usage of the MCP client."""
    import sys
    
    async def example_callback(session: ClientSession) -> None:
        """Example callback to run with an MCP session."""
        print("Connected to MCP server!")
        
        # List resources
        resources = await session.list_resources()
        print(f"Available resources: {len(resources)}")
        for resource in resources:
            print(f"  - {resource.name}: {resource.description}")
    
    if len(sys.argv) < 2:
        print("Usage: python mcp_client.py <command> [args...]")
        sys.exit(1)
    
    command = sys.argv[1]
    args = sys.argv[2:] if len(sys.argv) > 2 else []
    
    print(f"Connecting to MCP server: {command} {args}")
    asyncio.run(run_stdio_client(command, args, callback=example_callback)) 