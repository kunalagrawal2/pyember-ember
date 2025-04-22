#TODO First making with stdio, will implement SSE later
from typing import Optional

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
        # If env needs to be passed, it might need JSON stringification/parsing
        # For now, assume env is not passed via custom_args or handled separately
        server_env = None # Simplification: Assume env is not configured via custom_args for now
        # If env IS needed via custom_args, you'd need:
        # env_str = provider_config.get("env")
        # server_env = json.loads(env_str) if env_str else None

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

        self._model = model

        # Initialize async-related state
        self._session = None
        self._stdio_client = None
        self._read = None
        self._write = None

        # Will be determined after initialization based on server capabilities
        self._server_capabilities = None
        self._use_sampling_api = False
        self._use_prompts_api = False
        self._prompt_names = []  # Will store available prompt names if using prompts API
        self._prompt_details = {}  # Will store prompt details

    #TODO Super Jank work around
    def create_client(self) -> Any:
        """
        Satisfies the BaseProviderModel requirement for a synchronous client creator.

        For MCP, the actual client (ClientSession) is created asynchronously
        during initialize_session. This method currently does minimal setup
        and returns None. The BaseProviderModel assigns this to self.client.
        """
        self.logger.debug("MCP Provider create_client called (returns None). Session created in initialize_session.")
        # In the future, could perform some synchronous setup if needed.
        # For now, the main setup happens in __init__ and initialize_session.
        return None # The actual session is created asynchronously

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

            # After initialization, based on server capabilities:
            self._analyze_server_capabilities(init_result)

            # If prompts API is available, fetch available prompts
            if self._use_prompts_api:
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

        except Exception as e:
            self.logger.error(f"Failed to initialize MCP session: {str(e)}", exc_info=True)
            await self.terminate()
            raise ModelProviderError(f"MCP session initialization failed: {e}") from e

    def _analyze_server_capabilities(self, init_result: types.InitializeResult) -> None:
        """Analyze server capabilities and set appropriate flags."""
        self._server_capabilities = init_result.capabilities
        
        # Log detailed capabilities for debugging
        self.logger.info(f"Server info: {init_result.serverInfo}")
        self.logger.info(f"Server capabilities: {init_result.capabilities}")
        
        # Check for sampling API support
        if hasattr(init_result.capabilities, 'sampling') and init_result.capabilities.sampling:
            self.logger.info("Server supports sampling API")
            self._use_sampling_api = True
        
        # Check for prompts API support
        if hasattr(init_result.capabilities, 'prompts') and init_result.capabilities.prompts:
            self.logger.info("Server supports prompts API")
            self._use_prompts_api = True
            # Optionally fetch available prompts immediately
            # This could be done asynchronously in initialize_session
        
        if not (self._use_sampling_api or self._use_prompts_api):
            self.logger.warning("Server doesn't support sampling or prompts APIs")

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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        # retry_error_cls=ProviderAPIError, # Removed
        reraise=True,
    )
    async def forward(
        self, request: ChatRequest, params: Optional[ProviderParams] = None
    ) -> ChatResponse:
        """
        Processes a chat request using the initialized MCP session.
        """
        if not self._session:
            # Attempt to initialize if not already done
            self.logger.warning("MCP session not initialized. Attempting initialization...")
            await self.initialize_session()
            if not self._session: # Check again after attempt
                 raise ModelProviderError("MCP session is not initialized. Cannot forward request.")

        # 1. Prepare Parameters
        mcp_params: McpChatParameters = self._prepare_params(request, params, McpChatParameters)

        # 2. Validate Request
        self._validate_request(request, mcp_params)

        # 3. Convert Ember ChatRequest to MCP message format
        mcp_messages: List[types.SamplingMessage] = []
        system_prompt_text: Optional[str] = "You are a helpful AI assistant. Please provide clear and accurate responses." # Default system prompt

        # Extract context if it exists, override default system prompt
        if request.context:
            system_prompt_text = request.context

        # Add the current user prompt
        mcp_messages.append(types.SamplingMessage(role="user", content=types.TextContent(type="text", text=request.prompt)))

        self.logger.debug(f"mcp_messages: {mcp_messages}")
        self.logger.debug(f"system_prompt_text: {system_prompt_text}")

        # Check if max_tokens is available and required
        if mcp_params.max_tokens is None:
            # setting a default
            mcp_params.max_tokens = 1024 #TODO What should the default be?

        # 4. Prepare MCP Request Parameters object
        mcp_request_params = types.CreateMessageRequestParams(
            # Required fields
            messages=mcp_messages,
            maxTokens=mcp_params.max_tokens,

            # Optional fields explicitly set
            systemPrompt=system_prompt_text,
            temperature=mcp_params.temperature if mcp_params.temperature is not None else None,
            stopSequences=mcp_params.stop_sequences if mcp_params.stop_sequences else None,
            metadata=mcp_params.metadata if mcp_params.metadata is not None else None,
        )

        # Initialize usage with a default UsageStats object
        usage = UsageStats()

        # 5. Call MCP using send_request
        self.logger.debug(f"Sending createMessage request to MCP: {mcp_request_params}")
    
        try:
            # Select API based on server capabilities
            result = None
            
            # Try sampling API if available
            if self._use_sampling_api:
                self.logger.info("Using sampling API")
                try:
                    create_message_req_instance = types.CreateMessageRequest(
                        method="sampling/createMessage",
                        params=mcp_request_params
                    )
                    result = await self._session.send_request(
                        create_message_req_instance, 
                        types.CreateMessageResult
                    )
                    # Process result for sampling API
                    return self._create_chat_response_from_sampling_result(result, request)
                except Exception as e:
                    self.logger.warning(f"Sampling API request failed: {e}")
                    # If sampling fails and prompts API is available, fall back
                    if not self._use_prompts_api:
                        raise
            
            # Use prompts API if available (or as fallback)
            if self._use_prompts_api:
                self.logger.info("Using prompts API")
                
                # First check if we have any prompts, if not, try to fetch them
                if not hasattr(self, '_prompt_names') or not self._prompt_names:
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
                        self.logger.warning(f"Failed to fetch prompts list: {e}")
                        self._prompt_names = []
                        self._prompt_details = {}
                
                # If we don't have any prompts, we can't continue
                if not self._prompt_names:
                    raise ModelProviderError("No prompts available on server")
                
                # Use prompt specified in model_info.provider.custom_args if available
                custom_prompt = None
                if hasattr(self.model_info, 'provider') and hasattr(self.model_info.provider, 'custom_args'):
                    custom_prompt = self.model_info.provider.custom_args.get('prompt_name')
                
                # Determine which prompt to use
                prompt_name = None
                if custom_prompt and custom_prompt in self._prompt_names:
                    # Use the custom prompt from model_info if available
                    prompt_name = custom_prompt
                    self.logger.info(f"Using configured prompt '{prompt_name}'")
                else:
                    # Otherwise, use the first available prompt
                    prompt_name = self._prompt_names[0]
                    self.logger.info(f"Using default prompt '{prompt_name}'")
                
                # Determine appropriate arguments based on prompt schema if possible
                prompt_args = {"input": request.prompt}  # Default fallback
                
                # Check if we have prompt details that might help with argument names
                if prompt_name in self._prompt_details:
                    prompt = self._prompt_details[prompt_name]
                    # If the prompt has defined argument schemas, try to match them
                    if hasattr(prompt, 'arguments') and prompt.arguments:
                        # Look for common argument names in the schema
                        arg_names = [arg.name for arg in prompt.arguments]
                        
                        # Try common input argument names
                        common_input_names = ["input", "message", "prompt", "query", "text", "user_input"]
                        for arg_name in common_input_names:
                            if arg_name in arg_names:
                                prompt_args = {arg_name: request.prompt}
                                self.logger.debug(f"Using '{arg_name}' as prompt input argument")
                                break
                
                # Create prompt request with the best arguments we can determine
                try:
                    # Create prompt request
                    prompt_request = types.GetPromptRequest(
                        method="prompts/get",
                        params=types.GetPromptRequestParams(
                            name=prompt_name,
                            arguments=prompt_args
                        )
                    )
                    
                    # Send the request and get the result
                    result = await self._session.send_request(
                        prompt_request, 
                        types.GetPromptResult
                    )
                    
                    # Process result from prompt API
                    return self._create_chat_response_from_prompt_result(result, request)
                except Exception as e:
                    self.logger.error(f"Prompt API request failed: {e}")
                    raise ModelProviderError(f"Failed to get response from prompt '{prompt_name}': {e}")
            
            # If we reach here, neither sampling nor prompts API worked
            raise ModelProviderError("Server doesn't support compatible APIs")

        except McpError as mcp_err:
             self.logger.error(f"MCP Error during createMessage: {mcp_err.error}", exc_info=True)
             # You might want to map McpError codes to Ember exceptions
             raise ProviderAPIError(f"MCP request failed: {mcp_err.error.message}") from mcp_err
        except ValidationError as val_err:
             self.logger.error(f"Pydantic validation error creating CreateMessageRequest: {val_err}", exc_info=True)
             raise ModelProviderError(f"Internal error creating MCP request structure: {val_err}") from val_err
        except Exception as e:
             # Catch other potential errors during the request/send_request call
             self.logger.error(f"Unexpected error during MCP createMessage call: {e}", exc_info=True)
             raise ModelProviderError(f"Failed to send message via MCP: {e}") from e

    def _create_chat_response_from_sampling_result(
        self, result: types.CreateMessageResult, request: ChatRequest
    ) -> ChatResponse:
        """Convert sampling API result to ChatResponse."""
        if hasattr(result.content, "text"):
            return ChatResponse(
                data=result.content.text,
                model=f"{self.model_info.id}/{result.model}",
                raw=result,
                usage=UsageStats(),  # Fill with actual usage if available
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
