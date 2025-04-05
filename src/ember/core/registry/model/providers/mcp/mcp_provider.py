"""
MCP Provider Implementation - Simplified

A minimal wrapper around the MCP client for Ember integration.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Awaitable

from mcp import ClientSession
from mcp.types import Resource, Tool, Prompt, CreateMessageRequest, CreateMessageResult

from ember.core.registry.model.providers.base_provider import BaseProviderModel
from ember.core.registry.model.base.schemas.model_info import ModelInfo
from ember.core.registry.model.providers.mcp.mcp_client import run_stdio_client, run_sse_client, default_sampling_callback
from ember.core.registry.model.base.schemas.chat_schemas import ChatRequest, ChatResponse


class McpProvider(BaseProviderModel):
    """
    Provider for Model Context Protocol (MCP) servers.
    
    This provider implements the BaseProviderModel interface to allow
    access to MCP servers within the Ember framework. It provides a simple wrapper
    around the MCP client for integration with Ember's provider architecture.
    """

    # Provider name for registration
    PROVIDER_NAME: str = "mcp"

    def __init__(
        self,
        model_info: ModelInfo,
        server_type: str = "stdio",
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
        url: Optional[str] = None,
        sampling_callback: Optional[Callable[[CreateMessageRequest], Awaitable[CreateMessageResult]]] = None,
    ):
        """
        Initialize the MCP provider.
        
        Args:
            model_info: Information about the MCP server model
            server_type: Type of MCP server connection ("stdio" or "sse")
            command: Command to run for stdio servers
            args: Arguments for the command
            url: URL for SSE servers
            sampling_callback: Callback for handling sampling requests
        """
        # Initialize BaseProviderModel
        self.model_info = model_info
        
        # Store configuration
        self.server_type = server_type
        self.command = command
        self.args = args or []
        self.url = url
        self.sampling_callback = sampling_callback or default_sampling_callback
        
        # We'll create the session on demand
        self.session = None
        self._active_context = None
        
    def create_client(self) -> Any:
        """
        Create a client for connecting to the MCP server.
        
        This method is required by BaseProviderModel but doesn't actually create
        a persistent client, as MCP clients are designed to be used as context managers.
        
        Returns:
            A placeholder value, as the real client is created during session setup.
        """
        # We don't create a persistent client here, as MCP clients are used as context managers
        return None
    
    async def forward(self, request: ChatRequest) -> ChatResponse:
        """
        Process a chat request through the MCP server.
        
        Args:
            request: The chat request to process
            
        Returns:
            The chat response from the MCP server
        """
        # For MCP, we convert the Ember chat request to an MCP CreateMessageRequest
        # and process it through the sampling callback
        
        # Create a simple response acknowledging the request
        return ChatResponse(
            response="MCP providers are primarily for accessing resources, tools, and prompts. "
                    "Direct chat functionality is limited. Try using specific MCP methods instead.",
            model=self.model_info.name,
            usage={"prompt_tokens": len(request.prompt), "completion_tokens": 0, "total_tokens": len(request.prompt)}
        )
    
    async def _ensure_session(self) -> ClientSession:
        """
        Ensure a session is established to the MCP server.
        
        Returns:
            An established ClientSession
        """
        if self.session is not None:
            return self.session
        
        # Create a new session
        if self.server_type == "stdio":
            if not self.command:
                raise ValueError("Command must be provided for stdio connection")
            
            # This will be awaited by __aenter__
            self._run_stdio = run_stdio_client(
                self.command, 
                self.args, 
                sampling_callback=self.sampling_callback
            )
            
        elif self.server_type == "sse":
            if not self.url:
                raise ValueError("URL must be provided for SSE connection")
            
            # This will be awaited by __aenter__
            self._run_sse = run_sse_client(
                self.url, 
                sampling_callback=self.sampling_callback
            )
            
        else:
            raise ValueError(f"Unsupported server type: {self.server_type}")
        
        return self.session
    
    async def close(self) -> None:
        """Close the connection to the MCP server."""
        # The session will be closed by the context manager
        self.session = None
    
    async def list_resources(self) -> List[Resource]:
        """List available resources from the MCP server."""
        # We'll use the context manager pattern for this
        async def _list_resources(session: ClientSession) -> List[Resource]:
            return await session.list_resources()
        
        if self.server_type == "stdio":
            return await run_stdio_client(self.command, self.args, _list_resources, self.sampling_callback)
        else:
            return await run_sse_client(self.url, _list_resources, self.sampling_callback)
    
    async def read_resource(self, name: str) -> Tuple[str, str]:
        """
        Read a resource from the MCP server.
        
        Args:
            name: Name of the resource
            
        Returns:
            Content and MIME type of the resource
        """
        async def _read_resource(session: ClientSession) -> Tuple[str, str]:
            return await session.read_resource(name)
        
        if self.server_type == "stdio":
            return await run_stdio_client(self.command, self.args, _read_resource, self.sampling_callback)
        else:
            return await run_sse_client(self.url, _read_resource, self.sampling_callback)
    
    async def list_tools(self) -> List[Tool]:
        """List available tools from the MCP server."""
        async def _list_tools(session: ClientSession) -> List[Tool]:
            return await session.list_tools()
        
        if self.server_type == "stdio":
            return await run_stdio_client(self.command, self.args, _list_tools, self.sampling_callback)
        else:
            return await run_sse_client(self.url, _list_tools, self.sampling_callback)
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call a tool on the MCP server.
        
        Args:
            name: Name of the tool
            arguments: Arguments for the tool
            
        Returns:
            Result of the tool call
        """
        async def _call_tool(session: ClientSession) -> Any:
            result = await session.call_tool(name, arguments)
            return result
        
        if self.server_type == "stdio":
            return await run_stdio_client(self.command, self.args, _call_tool, self.sampling_callback)
        else:
            return await run_sse_client(self.url, _call_tool, self.sampling_callback)
    
    async def list_prompts(self) -> List[Prompt]:
        """List available prompts from the MCP server."""
        async def _list_prompts(session: ClientSession) -> List[Prompt]:
            return await session.list_prompts()
        
        if self.server_type == "stdio":
            return await run_stdio_client(self.command, self.args, _list_prompts, self.sampling_callback)
        else:
            return await run_sse_client(self.url, _list_prompts, self.sampling_callback)
    
    async def get_prompt(self, name: str, arguments: Optional[Dict[str, str]] = None) -> Any:
        """
        Get a prompt from the MCP server.
        
        Args:
            name: Name of the prompt
            arguments: Arguments for the prompt
            
        Returns:
            Prompt result
        """
        async def _get_prompt(session: ClientSession) -> Any:
            return await session.get_prompt(name, arguments)
        
        if self.server_type == "stdio":
            return await run_stdio_client(self.command, self.args, _get_prompt, self.sampling_callback)
        else:
            return await run_sse_client(self.url, _get_prompt, self.sampling_callback)
    
    async def generate_with_model(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate text using the MCP server's sampling capability.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            
        Returns:
            Generated text
        """
        async def _generate(session: ClientSession) -> str:
            # Create a message request
            request = CreateMessageRequest(
                messages=[
                    {"role": "user", "content": {"type": "text", "text": prompt}}
                ],
                systemPrompt=system_prompt
            )
            
            # Send the request through the sampling callback
            if self.sampling_callback:
                result = await self.sampling_callback(request)
                if isinstance(result.content, TextContent):
                    return result.content.text
                return str(result.content)
            
            return "No sampling callback provided"
        
        if self.server_type == "stdio":
            return await run_stdio_client(self.command, self.args, _generate, self.sampling_callback)
        else:
            return await run_sse_client(self.url, _generate, self.sampling_callback)
    
    async def __aenter__(self) -> 'McpProvider':
        """Async context manager entry."""
        # For context manager usage, we establish a single session
        if self.server_type == "stdio":
            self.session = await run_stdio_client(self.command, self.args, sampling_callback=self.sampling_callback)
        else:
            self.session = await run_sse_client(self.url, sampling_callback=self.sampling_callback)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        # The session is automatically closed by the context manager
        self.session = None


# Factory function for creating MCP providers
def create_mcp_provider(
    model_id: str,
    server_type: str = "stdio",
    command: Optional[str] = None,
    args: Optional[List[str]] = None,
    url: Optional[str] = None,
    default_model: Optional[str] = None,
) -> McpProvider:
    """
    Factory function to create an MCP provider.
    
    Args:
        model_id: Model ID for the provider (e.g., "mcp:filesystem")
        server_type: Type of MCP server connection ("stdio" or "sse")
        command: Command to run for stdio servers
        args: Arguments for the command
        url: URL for SSE servers
        default_model: Optional default model to use for sampling
        
    Returns:
        An initialized MCP provider
    """
    # Create provider info with additional metadata
    provider_info = {
        "name": "mcp",
        "server_type": server_type,
        "default_model": default_model
    }
    
    model_info = ModelInfo(id=model_id, provider=provider_info)
    
    return McpProvider(
        model_info=model_info,
        server_type=server_type,
        command=command,
        args=args,
        url=url
    ) 