#TODO First making with stdio, will implement SSE later
#TODO Only MCP tool usage is implemented, rest will be added later
from typing import Optional, Tuple

import logging
from typing import Any, Dict, Final, List, Optional, cast

import asyncio
from pydantic import Field, field_validator
from requests.exceptions import HTTPError
from tenacity import retry, stop_after_attempt, wait_exponential

from ember.core.exceptions import ModelProviderError, ValidationError
from ember.core.registry.model.base.schemas.chat_schemas import (
    ChatRequest,
    ChatResponse,
    ProviderParams,
)
from ember.core.registry.model.base.schemas.usage import UsageStats
from ember.core.registry.model.base.schemas.model_info import ModelInfo
from ember.core.registry.model.base.utils.model_registry_exceptions import (
    InvalidPromptError,
    ProviderAPIError,
)
from ember.core.registry.model.base.utils.usage_calculator import DefaultUsageCalculator
from ember.core.registry.model.providers.base_provider import (
    BaseChatParameters,
    BaseProviderModel,
)
from ember.plugin_system import provider

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from mcp.server.fastmcp.prompts import base
from mcp.shared.exceptions import McpError

logger: logging.Logger = logging.getLogger(__name__)

@provider("MCP")
class McpClient():
    # Store model for MCP
    _model: Optional[BaseProviderModel] = None # Optional as a workaround so we can call registry.get_model(), have to inject manually
    
    # Store session and context managers
    _session: Optional[ClientSession] = None
    _stdio_client: Optional[stdio_client] = None
    _read: Optional[asyncio.StreamReader] = None
    _write: Optional[asyncio.StreamWriter] = None
    _server_params: StdioServerParameters
    
    # Store capabilities and available features
    _server_capabilities: Optional[types.ServerCapabilities] = None
    _client_capabilities: Optional[types.ClientCapabilities] = None
    
    # Store available tools, prompts, and resources
    _tool_names: List[str] = []
    _tool_details: Dict[str, Any] = {}
    _prompt_names: List[str] = []
    _prompt_details: Dict[str, Any] = {}
    _resource_uris: List[str] = []
    _resource_details: Dict[str, Any] = {}

    def __init__(
    self,
    model: BaseProviderModel,
    server_command: str,
    server_args: Optional[List[str]] = None,
    server_env: Optional[Dict[str, str]] = None,
    logger: Optional[logging.Logger] = None
) -> None:
        """
        Initialize MCP client utility.
        
        Args:
            server_command: Command to start the MCP server (required)
            server_args: Optional list of arguments for the server command
            server_env: Optional environment variables for the server process
            logger: Optional logger instance (defaults to module-level logger)
        
        Raises:
            Exception: If server_command is not provided
        """
        self.logger = logger # Use the module-level logger

        # Validate required command
        # TODO do we want general exceptions or more specific to MCP?
        if not server_command:
            raise Exception("MCP client requires server_command")
    
        # Set up server parameters
        self._server_params = StdioServerParameters(
            command=server_command,
            args=server_args or [],  # Convert None to empty list
            env=server_env,  # Allow custom env vars
        )

        self.logger.debug(f"Set up server parameters: {self._server_params}") 


         # Initialize client capabilities with sampling support
        self._client_capabilities = types.ClientCapabilities(
            sampling=types.SamplingCapability(),
            experimental=None,
            roots=None
        )

        self.logger.debug(f"Initialized client capabilities: {self._client_capabilities}") 

        self.model = model # Can also be optional

        # Initialize async-related state (same as provider)
        self._session: Optional[ClientSession] = None
        self._stdio_client: Optional[stdio_client] = None
        self._read: Optional[asyncio.StreamReader] = None
        self._write: Optional[asyncio.StreamWriter] = None
        
        # Initialize server capabilities (same as provider)
        self._server_capabilities: Optional[types.ServerCapabilities] = None
        
        # Initialize feature storage (same as provider)
        self._tool_names: List[str] = []
        self._tool_details: Dict[str, Any] = {}
        self._prompt_names: List[str] = []
        self._prompt_details: Dict[str, Any] = {}
        self._resource_uris: List[str] = []
        self._resource_details: Dict[str, Any] = {}
        
        self.logger.debug(
            f"Initialized MCP client with command: {server_command} "
            f"args: {server_args or []} env: {server_env or {}}" 
        )

    async def connect(self) -> None:
        """Initializes the MCP connection and session using context managers."""
        if self._session:
            self.logger.warning("MCP session already initialized.")
            return

        try:
            self.logger.info(f"Starting MCP server via: {self._server_params.command} {' '.join(self._server_params.args)}")
            self._stdio_client = stdio_client(self._server_params)
            self._read, self._write = await self._stdio_client.__aenter__()
            self.logger.debug("stdio_client context entered.")

            self._session = ClientSession(self._read, self._write)
            self.logger.debug("ClientSession created")
            
            await self._session.__aenter__()
            self.logger.debug("ClientSession context entered")

            # Send initialization request with proper capabilities
            self.logger.info("Sending initialization request...")
            init_result = await self._session.initialize()
            self.logger.info(f"Server initialized with capabilities: {init_result}")

            # Send initialized notification
            self.logger.debug("Sending initialized notification...")
            initialized_notification = types.InitializedNotification(
                method="notifications/initialized"
            )
            await self._session.send_notification(
                types.ClientNotification(
                    root=initialized_notification  # Add the root structure
                )
            )
            
            self.logger.info("MCP session initialized")

            # After initialization, analyze and log server capabilities (just for logging purposes)
            def _analyze_server_capabilities(self, init_result: types.InitializeResult) -> None:
                """Analyze and log server capabilities for debugging purposes."""
                self._server_capabilities = init_result.capabilities
                
                # Log detailed server information and capabilities
                self.logger.info(f"Server info: {init_result.serverInfo}")
                self.logger.info(f"Server capabilities: {init_result.capabilities}")
                
                # Log individual capabilities for better visibility in logs
                if hasattr(init_result.capabilities, 'prompts') and init_result.capabilities.prompts:
                    self.logger.info("Server supports prompts API")
                
                if hasattr(init_result.capabilities, 'tools') and init_result.capabilities.tools:
                    self.logger.info("Server supports tools API")
                
                if hasattr(init_result.capabilities, 'resources') and init_result.capabilities.resources:
                    self.logger.info("Server supports resources API")

            _analyze_server_capabilities(init_result) # Just for logging

            # If tools API is available, fetch available tools
            self._fetch_tools()

            # TODO prompts and resources API not implemented yet

            # If prompts API is available, fetch available prompts
            self._fetch_prompts()
            
            # If resources API is available, fetch available resources
            self._fetch_resources()

        except Exception as e:
            self.logger.error(f"Error during MCP initialization: {e}", exc_info=True)
            raise ModelProviderError(f"Initialization failed: {e}")

    async def _fetch_tools(self):
        if hasattr(self._server_capabilities, 'tools') and self._server_capabilities.tools:
                try:
                    tools_result = await self.list_tools()
                    self._tool_names = [tool.name for tool in tools_result]
                    self._tool_details = {tool.name: tool for tool in tools_result}
                    self.logger.info(f"Available tools: {self._tool_names}")
                except Exception as e:
                    self.logger.warning(f"Failed to fetch available tools: {e}")
                    self._tool_names = []
                    self._tool_details = {}
        else:
            self.logger.info("Server does not support tools")

    async def _fetch_prompts(self):
        if hasattr(self._server_capabilities, 'prompts') and self._server_capabilities.prompts:
                try:
                    prompts_result = await self.list_prompts()
                    self._prompt_names = [prompt.name for prompt in prompts_result]
                    self._prompt_details = {prompt.name: prompt for prompt in prompts_result}
                    self.logger.info(f"Available prompts: {self._prompt_names}")
                except Exception as e:
                    self.logger.warning(f"Failed to fetch available prompts: {e}")
                    self._prompt_names = []
                    self._prompt_details = {}
        else:
            self.logger.info("Server does not support prompts")

    async def _fetch_resources(self):
        if hasattr(self._server_capabilities, 'resources') and self._server_capabilities.resources:
                try:
                    resources_result = await self.list_resources()
                    self._resource_uris = [resource.uri for resource in resources_result]
                    self._resource_details = {resource.uri: resource for resource in resources_result}
                    self.logger.info(f"Available resources: {self._resource_uris}")
                except Exception as e:
                    self.logger.warning(f"Failed to fetch available resources: {e}")
                    self._resource_uris = []
                    self._resource_details = {}
        else:
            self.logger.info("Server does not support resources")
