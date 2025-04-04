"""
MCP Provider Implementation

Provides a provider interface for Model Context Protocol (MCP) servers.
This module enables Ember applications to interact with MCP servers through
a standardized provider interface, consistent with Ember's architecture.
"""

from __future__ import annotations

import abc
import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Awaitable, cast

# Use the actual MCP package structure
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from mcp.types import ClientCapabilities, SseClientParameters
from mcp.types import (
    Resource, Tool, Prompt, Root,
    CreateMessageRequest, CreateMessageResult,
    CallToolResult, GetPromptResult,
    TextContent, ImageContent
)

from ember.core.registry.model.providers.base_provider import BaseProviderModel
from ember.core.registry.model.base.schemas.model_info import ModelInfo


class McpProvider(BaseProviderModel):
    """
    Provider for Model Context Protocol (MCP) servers.
    
    This provider implements the BaseProviderModel interface to allow
    access to MCP servers within the Ember framework. It handles connection
    management, resource access, tool invocation, and other MCP functionality.
    
    Unlike model providers like OpenAI or Anthropic, the MCP provider connects
    to custom servers that expose resources, tools, and prompts rather than
    directly to language models.
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
        name: str = "ember-mcp-provider",
        version: str = "1.0.0",
        logging_level: int = logging.INFO,
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
            name: Name of the client
            version: Version of the client
            logging_level: Logging level
            sampling_callback: Callback for handling sampling requests
        """
        super().__init__(model_info=model_info)
        
        self.server_type = server_type
        self.command = command
        self.args = args or []
        self.url = url
        self.name = name
        self.version = version
        self.sampling_callback = sampling_callback
        self.session: Optional[ClientSession] = None
        self._read_stream = None
        self._write_stream = None
        self._lock = asyncio.Lock()
        self._event_loop = None
        
        # Configure logging
        self.logger = logging.getLogger(f"mcp_provider.{name}")
        self.logger.setLevel(logging_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate the provider configuration."""
        if self.server_type == "stdio" and not self.command:
            raise ValueError("Command must be provided for stdio server type")
        elif self.server_type == "sse" and not self.url:
            raise ValueError("URL must be provided for SSE server type")
    
    async def _ensure_session(self) -> ClientSession:
        """Ensure a session exists and is initialized."""
        async with self._lock:
            if self.session is None:
                self.logger.info(f"Creating new MCP session for {self.name}")
                await self._connect()
            return self.session
    
    async def _connect(self) -> None:
        """Connect to the MCP server."""
        if self.server_type == "stdio":
            await self._connect_stdio()
        elif self.server_type == "sse":
            await self._connect_sse()
        else:
            raise ValueError(f"Unsupported server type: {self.server_type}")
    
    async def _connect_stdio(self) -> None:
        """Connect to an MCP server via stdio."""
        if not self.command:
            raise ValueError("Command must be provided for stdio connection")
        
        self.logger.info(f"Connecting to MCP server via stdio: {self.command} {self.args}")
        params = StdioServerParameters(
            command=self.command,
            args=self.args
        )
        
        async with stdio_client(params) as (read_stream, write_stream):
            self._read_stream = read_stream
            self._write_stream = write_stream
            
            self.session = ClientSession(
                read_stream=read_stream,
                write_stream=write_stream,
                sampling_callback=self.sampling_callback
            )
            
            initialize_result = await self.session.initialize(
                client_name=self.name,
                client_version=self.version
            )
            
            self.logger.info(f"Connected to MCP server: {initialize_result.server_info.name} v{initialize_result.server_info.version}")
    
    async def _connect_sse(self) -> None:
        """Connect to an MCP server via SSE."""
        if not self.url:
            raise ValueError("URL must be provided for SSE connection")
        
        self.logger.info(f"Connecting to MCP server via SSE: {self.url}")
        params = SseClientParameters(url=self.url)
        
        async with sse_client(params) as (read_stream, write_stream):
            self._read_stream = read_stream
            self._write_stream = write_stream
            
            self.session = ClientSession(
                read_stream=read_stream,
                write_stream=write_stream,
                sampling_callback=self.sampling_callback
            )
            
            initialize_result = await self.session.initialize(
                client_name=self.name,
                client_version=self.version
            )
            
            self.logger.info(f"Connected to MCP server: {initialize_result.server_info.name} v{initialize_result.server_info.version}")
    
    async def close(self) -> None:
        """Close the connection to the MCP server."""
        if self.session:
            self.logger.info("Closing MCP session")
            await self.session.close()
            self.session = None
    
    async def list_resources(self) -> List[Resource]:
        """List available resources from the MCP server."""
        session = await self._ensure_session()
        return await session.list_resources()
    
    async def read_resource(self, uri: str) -> Tuple[str, str]:
        """
        Read a resource from the MCP server.
        
        Args:
            uri: URI of the resource
            
        Returns:
            Tuple of (content, mime_type)
        """
        session = await self._ensure_session()
        resources = await session.read_resource(uri)
        
        if not resources:
            raise ValueError(f"No content returned for resource: {uri}")
        
        resource = resources[0]  # Get the first resource
        
        # Handle text content
        if isinstance(resource.content, TextContent):
            return resource.content.text, resource.mime_type or "text/plain"
        # Handle image content
        elif isinstance(resource.content, ImageContent):
            return f"<image data: {len(resource.content.data)} bytes>", resource.mime_type or "image/unknown"
        # Handle other content types
        else:
            return str(resource.content), resource.mime_type or "application/octet-stream"
    
    async def list_tools(self) -> List[Tool]:
        """List available tools from the MCP server."""
        session = await self._ensure_session()
        return await session.list_tools()
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call a tool on the MCP server.
        
        Args:
            name: Name of the tool
            arguments: Arguments for the tool
            
        Returns:
            Result of the tool call
        """
        session = await self._ensure_session()
        result = await session.call_tool(name, arguments)
        
        # Extract text content if present
        if isinstance(result.content, TextContent):
            return result.content.text
        # Otherwise return the raw content
        return result.content
    
    async def list_prompts(self) -> List[Prompt]:
        """List available prompts from the MCP server."""
        session = await self._ensure_session()
        return await session.list_prompts()
    
    async def get_prompt(self, name: str, arguments: Optional[Dict[str, str]] = None) -> GetPromptResult:
        """
        Get a prompt from the MCP server.
        
        Args:
            name: Name of the prompt
            arguments: Arguments for the prompt
            
        Returns:
            The prompt result
        """
        session = await self._ensure_session()
        return await session.get_prompt(name, arguments)
    
    async def create_message(self, request: Dict[str, Any]) -> CreateMessageResult:
        """
        Create a message using sampling.
        
        Args:
            request: Message request parameters
            
        Returns:
            The message result
        """
        session = await self._ensure_session()
        return await session.create_message(request)
    
    async def add_roots(self, roots: List[str]) -> None:
        """
        Add roots to the MCP server.
        
        Args:
            roots: List of root URIs
        """
        session = await self._ensure_session()
        await session.add_roots(roots)
    
    async def remove_roots(self, roots: List[str]) -> None:
        """
        Remove roots from the MCP server.
        
        Args:
            roots: List of root URIs
        """
        session = await self._ensure_session()
        await session.remove_roots(roots)
    
    async def list_roots(self) -> List[Root]:
        """List roots from the MCP server."""
        session = await self._ensure_session()
        return await session.list_roots()

    # BaseProviderModel Implementation
    
    def create_client(self) -> Any:
        """Create the client for this provider.
        
        Note: MCP provider uses async client creation, so this
        synchronous method will return None. Use _ensure_session instead.
        """
        # MCP uses async initialization, so this is a no-op
        return None
    
    def forward(self, request: Any) -> Any:
        """Process a request with this provider.
        
        This method provides compatibility with the BaseProviderModel interface.
        It handles synchronous requests by running the async methods in a new event loop.
        
        Args:
            request: The request to process, typically a string prompt or dict
            
        Returns:
            The response from the MCP server
        """
        import asyncio
        
        # If we're already in an event loop, we can't create a new one
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                raise RuntimeError(
                    "MCP provider cannot be used synchronously from within an async context. "
                    "Use the async methods directly instead."
                )
        except RuntimeError:
            # No event loop exists, which is fine for this case
            pass
        
        # Create a new event loop for this request
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Handle different request types
            if isinstance(request, str):
                # Simple string prompt
                message_request = {
                    "messages": [{
                        "role": "user", 
                        "content": {"type": "text", "text": request}
                    }],
                    "includeContext": "thisServer"
                }
                result = loop.run_until_complete(self.create_message(message_request))
                return result.content.text if hasattr(result.content, "text") else str(result.content)
            
            elif isinstance(request, dict) and "messages" in request:
                # Already formatted as a message request
                result = loop.run_until_complete(self.create_message(request))
                return result.content.text if hasattr(result.content, "text") else str(result.content)
            
            elif isinstance(request, dict) and "tool" in request:
                # Tool call request
                result = loop.run_until_complete(
                    self.call_tool(request["tool"], request.get("arguments", {}))
                )
                return result
            
            else:
                raise ValueError(
                    f"Unsupported request type: {type(request)}. "
                    "Must be a string prompt or a dict with 'messages' or 'tool' key."
                )
        finally:
            # Clean up the event loop
            loop.close()
    
    # Context manager support
    
    async def __aenter__(self) -> McpProvider:
        """Support for async context manager."""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Clean up resources when used as a context manager."""
        await self.close()

    async def generate_with_model(
        self, 
        prompt: str, 
        model_name: str = None, 
        temperature: float = 0.7,
        max_tokens: int = 1000,
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate text using a specific model through MCP sampling.
        
        This method provides a more standard LLM interface similar to other providers.
        
        Args:
            prompt: The text prompt to send to the model
            model_name: Optional specific model to request (if None, uses server default)
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt
            
        Returns:
            The generated text response
        """
        model_preferences = None
        if model_name:
            model_preferences = {
                "hints": [{"name": model_name}]
            }
        
        request = {
            "messages": [{
                "role": "user",
                "content": {
                    "type": "text",
                    "text": prompt
                }
            }],
            "modelPreferences": model_preferences,
            "systemPrompt": system_prompt,
            "includeContext": "thisServer",
            "temperature": temperature,
            "maxTokens": max_tokens
        }
        
        result = await self.create_message(request)
        return result.content.text if hasattr(result.content, "text") else str(result.content)


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