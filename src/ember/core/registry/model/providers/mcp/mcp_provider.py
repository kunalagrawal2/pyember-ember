#TODO First making with stdio, will implement SSE later
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

class McpProviderParams(ProviderParams):
    #make it based on the provider?
    pass

# Define MCP specific parameters if needed (placeholder for now)
class McpChatParameters(BaseChatParameters):
    # Explicitly include common parameters we intend to map
    temperature: Optional[float] = None
    stop_sequences: Optional[List[str]] = None
    # Add MCP-specific fields corresponding to CreateMessageRequestParams
    include_context: Optional[types.IncludeContext] = None
    metadata: Optional[Dict[str, Any]] = None
    # Add any other MCP specific parameters here later
    pass

# Helper exception for configuration issues
class ConfigurationError(Exception):
    pass

@provider("MCP")
class McpClient(BaseProviderModel):
    # Store session and context managers
    _model: Optional[BaseProviderModel] = None # Optional as a workaround so we can call registry.get_model(), have to inject manually
    _session: Optional[ClientSession] = None
    _stdio_client: Optional[stdio_client] = None
    _read: Optional[asyncio.StreamReader] = None
    _write: Optional[asyncio.StreamWriter] = None
    _server_params: StdioServerParameters # Required for launching the server

    # Define default parameters
    DEFAULT_PARAMS: Final[Dict[str, Any]] = {} # Add MCP defaults if any

    def __init__(self, model_info: ModelInfo, model: Optional[BaseProviderModel] = None):
        """
        Initializes the McpProvider.
        
        Args:
            model_info: Information about the specific MCP model/server.
            model: Optional model instance to use
        """
        # Call BaseProviderModel's __init__ AFTER setting up _server_params
        # because create_client might theoretically depend on it, even if
        # it currently doesn't do much.
        self.logger = logger # Use the module-level logger

        # --- Configuration for StdioServerParameters ---
        provider_config = model_info.provider.custom_args or {}
        server_command = provider_config.get("command")

        # Get 'args' as a string from custom_args and split it into a list
        server_args_str = provider_config.get("args", "") # Default to empty string
        server_args = server_args_str.split() # Split by space

        # Env should ideally be a Dict[str, str], handle if stored differently
        # For now, assume env is not passed via custom_args or handled separately
        server_env = None # Simplification: Assume env is not configured via custom_args for now

        if not server_command:
            raise ConfigurationError(
                "MCP provider requires 'command' in provider.custom_args."
            )

        self._server_params = StdioServerParameters(
            command=server_command,
            args=server_args, # Pass the parsed list
            env=server_env,
        )
        # --- End Configuration ---

        # Now call the base class __init__ which will call our create_client
        super().__init__(model_info)

        self._model = model #Can be none for now

        # Initialize async-related state
        self._session = None
        self._stdio_client = None
        self._read = None
        self._write = None

        # MCP Server capabilities
        self._server_capabilities = None

        # Ember Client capabilites
        self._client_capabilities = None

        # Store prompt details 
        self._prompt_names = []  # Will store available prompt names if prompts API is available
        self._prompt_details = {}  # Will store prompt details
        # Store tools
        self._tool_names = []
        self._tool_details = {}
        # Store resources
        self._resource_uris = []
        self._resource_details = {}

    def create_client(self) -> Any:
        """
        Satisfies the BaseProviderModel requirement for a synchronous client creator.

        For MCP, the actual client (ClientSession) is created asynchronously
        during initialize_session. This method currently does minimal setup
        and returns None. The BaseProviderModel assigns this to self.client.
        """
        self.logger.debug("MCP Provider create_client called (returns None). Session created in initialize_session.")
        
        # Initialize client capabilities with sampling support
        self._client_capabilities = types.ClientCapabilities(
            sampling=types.SamplingCapability(),
            experimental=None,
            roots=None
        )
        
        self.logger.debug(f"Initialized client capabilities with sampling support: {self._client_capabilities}")
        
        return None
    
    def set_wrapped_model(self, model: BaseProviderModel) -> None:
        self.logger.debug(f"Setting wrapped model: {model.model_info.id}")
        self._model = model
    

    async def initialize_session(self) -> None:
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
            init_result = await self._session.send_request(
                types.InitializeRequest(
                    method="initialize",
                    params=types.InitializeRequestParams(
                        protocolVersion=types.LATEST_PROTOCOL_VERSION,
                        clientInfo=types.Implementation(
                            name="ember-mcp-client",
                            version="0.1.0"
                        ),
                        capabilities=types.ClientCapabilities(
                            sampling=types.SamplingCapability(),
                            experimental=None,
                            roots=None
                        )
                    )
                ),
                types.InitializeResult
            )
            self.logger.info(f"Server initialized with capabilities: {init_result}")

            # After server initialization, examine capabilities in more detail
            self.logger.info(f"Server capabilities detail: {init_result.capabilities}")
            if hasattr(init_result.capabilities, 'prompts'):
                self.logger.info(f"Prompts capability: {init_result.capabilities.prompts}")
            # Look for any endpoint information in the capabilities

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
            
            self.logger.info("MCP session fully initialized and ready")

            # After initialization, analyze and log server capabilities
            self._analyze_server_capabilities(init_result) # Just for logging

            # If prompts API is available, fetch available prompts
            if hasattr(self._server_capabilities, 'prompts') and self._server_capabilities.prompts:
                try:
                    list_prompts_request = types.ListPromptsRequest(
                        method="prompts/list",
                        params=None
                    )
                    prompts_result = await self._session.send_request(
                        list_prompts_request, 
                        types.ListPromptsResult
                    )
                    self._prompt_names = [prompt.name for prompt in prompts_result.prompts]
                    self._prompt_details = {prompt.name: prompt for prompt in prompts_result.prompts}
                    self.logger.info(f"Available prompts: {self._prompt_names}")
                except Exception as e:
                    self.logger.warning(f"Failed to fetch available prompts: {e}")
                    self._prompt_names = []
                    self._prompt_details = {}

                #TODO Refactor to in built functions in ClientSession
                # If tools API is available, fetch available tools
                if hasattr(self._server_capabilities, 'tools') and self._server_capabilities.tools:
                    try:
                        list_tools_request = types.ListToolsRequest(
                            method="tools/list",
                            params=None
                        )
                        tools_result = await self._session.send_request(
                            list_tools_request, 
                            types.ListToolsResult
                        )
                        self._tool_names = [tool.name for tool in tools_result.tools]
                        self._tool_details = {tool.name: tool for tool in tools_result.tools}
                        self.logger.info(f"Available tools: {self._tool_names}")
                    except Exception as e:
                        self.logger.warning(f"Failed to fetch available tools: {e}")
                        self._tool_names = []
                        self._tool_details = {}
                
                # If resources API is available, fetch available resources
                if hasattr(self._server_capabilities, 'resources') and self._server_capabilities.resources:
                    try:
                        list_resources_request = types.ListResourcesRequest(
                            method="resources/list",
                            params=None
                        )
                        resources_result = await self._session.send_request(
                            list_resources_request, 
                            types.ListResourcesResult
                        )
                        self._resource_uris = [resource.uri for resource in resources_result.resources]
                        self._resource_details = {resource.uri: resource for resource in resources_result.resources}
                        self.logger.info(f"Available resources: {self._resource_uris}")
                    except Exception as e:
                        self.logger.warning(f"Failed to fetch available resources: {e}")
                        self._resource_uris = []
                        self._resource_details = {}

        except Exception as e:
            self.logger.error(f"Failed to initialize MCP session: {str(e)}", exc_info=True)
            await self.terminate()
            raise ModelProviderError(f"MCP session initialization failed: {e}") from e

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
        

    # --- Implement BaseProviderModel abstract methods ---

    def get_default_params(self) -> Dict[str, Any]:
        """Returns default parameters for the MCP provider."""
        # Return a copy of the class's default parameters.
        # Do not call super() if the base class doesn't have this method.
        return self.DEFAULT_PARAMS.copy()

    def _validate_request(
        self, request: ChatRequest, params: McpChatParameters
    ) -> None:
        """Validate the chat request and parameters for MCP."""
        # Call super() for base validation if it exists and is needed
        # super()._validate_request(request, params) # Assuming BaseProviderModel has this
        if not request.prompt: # Example basic validation
             raise InvalidPromptError("Prompt cannot be empty.")
        # Add any MCP specific validation if needed

    def _prepare_params(
        self,
        request: ChatRequest,
        params: Optional[ProviderParams],
        param_class: type[BaseChatParameters],
    ) -> BaseChatParameters:
        """Prepares the final parameters by merging request, defaults, and provider params."""
        # Start with relevant fields from the request itself
        # Exclude provider_params as it's handled separately/merged later
        # Exclude none to avoid overwriting defaults with None from request
        combined_params = request.model_dump(exclude={'provider_params'}, exclude_none=True)

        # Merge defaults (defaults should not override explicit request values)
        default_params = self.get_default_params()
        for key, value in default_params.items():
            combined_params.setdefault(key, value) # Only set if key is not already present

        # Merge provider_params from the request (overrides defaults and request fields)
        if request.provider_params:
            combined_params.update(request.provider_params)

        # Merge explicit 'params' argument (highest precedence)
        if params:
            combined_params.update(params.model_dump(exclude_unset=True))

        # Validate and return using the specific parameter class
        try:
            # Filter params to only include those defined in the target param_class
            valid_keys = param_class.model_fields.keys()
            filtered_params = {k: v for k, v in combined_params.items() if k in valid_keys}
            return param_class(**filtered_params)
        # Catch the correct Pydantic error
        except ValidationError as e:
            self.logger.error(f"Parameter validation failed: {e}", exc_info=True)
            # Re-raise as ValidationError or a more specific custom error if desired
            # Raising ProviderAPIError here might also be valid depending on desired error hierarchy
            raise ValidationError(f"Parameter validation failed for {param_class.__name__}: {e}") from e
        

    # Will use this method for the agent to determine to use tools, read resources, etc. Used in forward
    # Just using tools for now
    async def process_query(self, query: str) -> str:
        """
        Process a query using MCP with automatic tool handling.
        
        Args:
            query: The user's text query
            
        Returns:
            A string containing the final response, including tool usage info
        """
        # Ensure session is initialized
        if not hasattr(self, '_session') or self._session is None:
            await self.initialize_session()
        
        final_text = []
        
        try:
            # 1. Get available tools from MCP server using the existing method
            available_tools = []
            if hasattr(self._server_capabilities, 'tools') and self._server_capabilities.tools:
                try:
                    available_tools = await self.list_tools()
                    self.logger.info(f"Available tools: {[tool.name for tool in available_tools]}")
                except Exception as e:
                    self.logger.warning(f"Failed to fetch available tools: {e}")
                    available_tools = []
            
            # 2. Prepare the system prompt to inform the model about available tools
            tool_instructions = ""
            if available_tools:
                tool_descriptions = []
                for tool in available_tools:
                    tool_descriptions.append(f"{tool.name}: {tool.description}")
                
                tool_instructions = f"""
                The following tools are available. If you need to use a tool, respond with:
                Call tool: tool_name(param1=value1, param2=value2)
                
                Available tools:
                {"".join(tool_descriptions)}
                """.strip('\n')
        

            # Create the ChatRequest with proper provider_params
            chat_request = ChatRequest(
                prompt=query,
                context=tool_instructions if tool_instructions else None,
            )
            print(f"Chat Request: {chat_request}")
            # 4. Send initial request to the model through MCP
           
            response = self._model.forward(chat_request) # model forwards aren't async
            
            final_text.append(response.data)
            
            # 5. Check if the response contains tool calls
            tool_calls = self._extract_tool_calls(response.data)
            
            # 6. Process tool calls if any
            if tool_calls and available_tools:
                for tool_call in tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    
                    self.logger.info(f"Executing tool: {tool_name} with args: {tool_args}")
                    final_text.append(f"\n[Calling tool: {tool_name}]")

                    #Just with tools for now
                    # Execute the tool using existing method
                    try:
                        tool_result = await self.call_tool(tool_name, tool_args)
                        
                        # Format the tool result for display
                        if isinstance(tool_result, list):
                            # Handle content list if returned
                            tool_output = "\n".join([str(item) for item in tool_result])
                        else:
                            tool_output = str(tool_result)
                        
                        final_text.append(f"[Tool result: {tool_output}]")
                        
                        # Send follow-up with tool results
                        followup_prompt = f"""
                        Previous query: {query}
                        Previous response: {response.data}
                        
                        Tool call: {tool_name}({', '.join([f'{k}={v}' for k, v in tool_args.items()])})
                        Tool result: {tool_output}
                        
                        Please provide your final answer based on this tool result.
                        """
                        
                        # Process the follow-up through the appropriate channel
                        if self._model:
                            followup_response = await self._model.forward(ChatRequest(prompt=followup_prompt))
                        
                        final_text.append(followup_response.data)
                        
                    except Exception as e:
                        error_msg = f"Error calling tool {tool_name}: {str(e)}"
                        self.logger.error(error_msg)
                        final_text.append(f"[{error_msg}]")
            
            return "".join(final_text)
            
        except Exception as e:
            error_msg = f"Error in process_query: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return f"Error processing your query: {error_msg}"

    def _extract_tool_calls(self, text: str) -> list:
        """
        Extract tool calls from model response text.
        
        Args:
            text: Response text from the model
            
        Returns:
            List of tool call dictionaries with name and args
        """
        tool_calls = []
        
        import re
        # Match "Call tool: tool_name(param1=value1, param2=value2)"
        pattern = r"Call tool->\s*(\w+)\s*\((.*?)\)"
        matches = re.findall(pattern, text)
        
        for match in matches:
            tool_name = match[0]
            args_str = match[1]
            
            # Parse arguments string into a dictionary
            args = {}
            for arg_pair in args_str.split(','):
                if '=' in arg_pair:
                    key, value = arg_pair.split('=', 1)
                    args[key.strip()] = value.strip().strip('"\'')  # Remove quotes if present
            
            tool_calls.append({"name": tool_name, "args": args})
        
        return tool_calls

    async def forward(
        self, request: ChatRequest, params: Optional[ProviderParams] = None
    ) -> ChatResponse:
        """
        Processes a chat request through the MCP provider or its underlying model.
        
        Args:
            request: The chat request containing the prompt and parameters
            params: Optional provider-specific parameters
            
        Returns:
            ChatResponse: The response from processing the request
        """
        # Check if we have an underlying model
        print(f"Checking Model ID: {self._model.model_info.id}")
        if self._model is None:
            raise ModelProviderError("No underlying model provided")
        else:
            try:
                # If we have a wrapped model but need tool capabilities, use process_query
                if hasattr(self, '_server_capabilities') and self._server_capabilities:
                    has_tools = (hasattr(self._server_capabilities, 'tools') and 
                                self._server_capabilities.tools)
                                
                    # If the MCP server has tools, use our process_query method to handle tool calls
                    if has_tools:
                        self.logger.info("Using process_query to handle potential tool usage")
                        result_text = await self.process_query(request.prompt)
                        return ChatResponse(
                            data=result_text,
                            model_id=self.model_info.id
                        )
                    
                    # Otherwise, directly use the wrapped model (bypass MCP protocol)
                    self.logger.info("Forwarding directly to underlying model")
                    return await self._model.forward(request)
                else:
                    # No server capabilities, so use the wrapped model directly
                    self.logger.info("No MCP server capabilities, using underlying model directly")
                    return await self._model.forward(request)
            except Exception as e:
                self.logger.error(f"Error using underlying model: {e}", exc_info=True)
                raise ModelProviderError(f"Forward to underlying model failed: {e}")
        
        # No underlying model, use our own implementation via chat method
        self.logger.info("No underlying model, using MCP provider's own implementation")
        return await self.chat(request)

    def _create_chat_response_from_sampling_result(
        self, result: types.CreateMessageResult, request: ChatRequest
    ) -> ChatResponse:
        """Convert sampling API result to ChatResponse."""
        if hasattr(result.content, "text"):
            return ChatResponse(
                data=result.content.text,
                model=f"{self.model_info.id}/{result.model}",
                raw=result,
                usage=UsageStats(),  # Fill wixth actual usage if available
                request=request
            )
        raise ModelProviderError("Unexpected result format from sampling API")

    def _create_chat_response_from_prompt_result(
        self, result: types.GetPromptResult, request: ChatRequest
    ) -> ChatResponse:
        """Convert prompt API result to ChatResponse."""
        # Extract assistant message from result.messages
        for message in result.messages:
            if message.role == "assistant" and hasattr(message.content, "text"):
                return ChatResponse(
                    data=message.content.text,
                    model=self.model_info.id,  # Prompt API might not provide model info
                    raw=result,
                    usage=UsageStats(),  # Fill with actual usage if available
                    request=request
                )
        
        # If no assistant message found, use last message or concatenate all
        if result.messages and hasattr(result.messages[-1].content, "text"):
            return ChatResponse(
                data=result.messages[-1].content.text,
                model=self.model_info.id,
                raw=result,
                usage=UsageStats(),
                request=request
            )
        
        raise ModelProviderError("No text content found in prompt result")

    async def terminate(self) -> None:
        """
        Cleanly terminates the MCP client session and transport by exiting contexts.
        """
        self.logger.info("Terminating MCP provider session...")
        # Exit contexts in reverse order of entry
        exit_exception = None
        try:
            if self._session:
                self.logger.debug("Exiting ClientSession context...")
                await self._session.__aexit__(None, None, None)
                self.logger.debug("ClientSession context exited.")
        except Exception as e:
            exit_exception = e
            self.logger.error(f"Error exiting ClientSession context: {e}", exc_info=True)
        finally:
            self._session = None
            self._session = None

        try:
            if self._stdio_client:
                self.logger.debug("Exiting stdio_client context...")
                await self._stdio_client.__aexit__(None, None, None)
                self.logger.debug("stdio_client context exited.")
        except Exception as e:
            # Prioritize the first exception if multiple occur
            if not exit_exception:
                exit_exception = e
            self.logger.error(f"Error exiting stdio_client context: {e}", exc_info=True)
        finally:
            self._read = None
            self._write = None
            self._stdio_client = None

        if exit_exception:
             # Re-raise the first exception encountered during cleanup
             raise ModelProviderError(f"Error during MCP termination: {exit_exception}") from exit_exception
        else:
             self.logger.info("MCP provider terminated successfully.")

    async def list_tools(self) -> List[types.Tool]:
        """List available tools provided by the server."""
        if not hasattr(self, '_session') or self._session is None:
            await self.initialize_session()
        
        if not hasattr(self._server_capabilities, 'tools') or not self._server_capabilities.tools:
            raise ModelProviderError("Server doesn't support tools API")
        
        try:
            list_tools_request = types.ListToolsRequest(
                method="tools/list",
                params=None
            )
            tools_result = await self._session.send_request(
                list_tools_request, 
                types.ListToolsResult
            )
            self.logger.info(f"Available tools: {[tool.name for tool in tools_result.tools]}")
            return tools_result.tools
        except Exception as e:
            self.logger.error(f"Failed to list tools: {e}")
            raise ModelProviderError(f"Failed to list tools: {e}")

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool provided by the server.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool
            
        Returns:
            The result returned by the tool
            
        Raises:
            ModelProviderError: If the server doesn't support tools or the tool call fails
        """
        if not hasattr(self, '_session') or self._session is None:
            await self.initialize_session()
        
        if not self._server_capabilities or not hasattr(self._server_capabilities, 'tools') or not self._server_capabilities.tools:
            raise ModelProviderError("Server doesn't support tools API")
        
        try:
            call_tool_request = types.CallToolRequest(
                method="tools/call",
                params=types.CallToolRequestParams(
                    name=tool_name,
                    arguments=arguments
                )
            )
            result = await self._session.send_request(
                call_tool_request, 
                types.CallToolResult
            )
            return result.content #Do we want to return as a content, or extract the data stored in content object?
        except Exception as e:
            self.logger.error(f"Failed to call tool '{tool_name}': {e}")
            raise ModelProviderError(f"Failed to call tool '{tool_name}': {e}")

    async def list_resources(self) -> List[types.Resource]:
        """List available resources provided by the server.
        
        Returns:
            List of Resource objects describing available resources
            
        Raises:
            ModelProviderError: If the server doesn't support resources or the request fails
        """
        if not hasattr(self, '_session') or self._session is None:
            await self.initialize_session()
        
        if not self._server_capabilities or not hasattr(self._server_capabilities, 'resources') or not self._server_capabilities.resources:
            raise ModelProviderError("Server doesn't support resources API")
        
        try:
            list_resources_request = types.ListResourcesRequest(
                method="resources/list",
                params=None
            )
            resources_result = await self._session.send_request(
                list_resources_request, 
                types.ListResourcesResult
            )
            self.logger.info(f"Available resources: {[resource.uri for resource in resources_result.resources]}")
            return resources_result.resources
        except Exception as e:
            self.logger.error(f"Failed to list resources: {e}")
            raise ModelProviderError(f"Failed to list resources: {e}")

    async def read_resource(self, uri: str) -> Any:
        """Read a resource from the server.
        
        Args:
            uri: URI of the resource to read
            
        Returns:
            Any

        Raises:
            ModelProviderError: If the server doesn't support resources or the read fails
        """
        if not hasattr(self, '_session') or self._session is None:
            await self.initialize_session()
        
        if not self._server_capabilities or not hasattr(self._server_capabilities, 'resources') or not self._server_capabilities.resources:
            raise ModelProviderError("Server doesn't support resources API")
        
        try:
            read_resource_request = types.ReadResourceRequest(
                method="resources/read",
                params=types.ReadResourceRequestParams(
                    uri=uri
                )
            )
            result = await self._session.send_request(
                read_resource_request, 
                types.ReadResourceResult
            )
            return result.contents
        except Exception as e:
            self.logger.error(f"Failed to read resource '{uri}': {e}")
            raise ModelProviderError(f"Failed to read resource '{uri}': {e}")

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """
        Sends a chat request to the MCP provider and returns the response.
        
        Args:
            request: The chat request containing prompt and parameters.
            
        Returns:
            The chat response from the MCP provider.
        """
        # Ensure session is initialized
        if not hasattr(self, '_session') or self._session is None:
            await self.initialize_session()
        
        
        try:
            # Try sampling API first if available
            if hasattr(self._server_capabilities, 'sampling') and self._server_capabilities.sampling:
                try:
                    # Sampling API code...
                    pass
                except Exception as e:
                    self.logger.warning(f"Sampling API request failed: {e}, trying prompts API next")
                    # Fall through to prompts API
            
            # Use prompts API if available
            if hasattr(self._server_capabilities, 'prompts') and self._server_capabilities.prompts:
                # Prompts API code...
                pass
            
            # If we reach here, neither sampling nor prompts API worked
            raise ModelProviderError("Server doesn't support compatible APIs for chat")
            
        except McpError as mcp_err:
            pass
