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

logger: logging.Logger = logging.getLogger(__name__)

class McpProviderParams(ProviderParams):
    #make it based on the provider?
    pass

# Define MCP specific parameters if needed (placeholder for now)
class McpChatParameters(BaseChatParameters):
    # Add any MCP specific parameters here later
    pass

# Helper exception for configuration issues
class ConfigurationError(Exception):
    pass

@provider("MCP")
class McpClient(BaseProviderModel):
    # Store session and context managers
    _model: BaseProviderModel # Required, the model that will be invoked after the MCP connection
    _session: Optional[ClientSession] = None
    _stdio_client: Optional[stdio_client] = None
    _read: Optional[asyncio.StreamReader] = None
    _write: Optional[asyncio.StreamWriter] = None
    _server_params: StdioServerParameters # Required for launching the server

    # Define default parameters
    DEFAULT_PARAMS: Final[Dict[str, Any]] = {} # Add MCP defaults if any

    def __init__(self, model_info: ModelInfo, model: BaseProviderModel):
        """
        Initializes the McpProvider.

        Args:
            model_info: Information about the specific MCP model/server.
                        We'll need to define how server command/args are passed,
                        perhaps via model_info.provider.config or similar.
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
        """
        Initializes the MCP connection and session using context managers.
        This replaces the old 'forward' method for lifecycle setup.
        """
        if self._session:
            self.logger.warning("MCP session already initialized.")
            return

        try:
            self.logger.info(f"Starting MCP server via: {self._server_params.command} {' '.join(self._server_params.args)}")
            # Create and enter the stdio_client context
            self._stdio_client = stdio_client(self._server_params)
            self._read, self._write = await self._stdio_client.__aenter__()
            self.logger.debug("stdio_client context entered.")

            # Define callbacks (example: sampling)
            # TODO: Make callbacks configurable or part of the provider logic
            async def handle_sampling_message(
                message: types.CreateMessageRequestParams,
            ) -> types.CreateMessageResult:
                 # This basic callback just echoes - replace with actual model logic
                 self.logger.warning("Using basic echo sampling callback.")
                 return types.CreateMessageResult(
                     role="assistant",
                     content=message.messages[-1].content, # Echo last message content
                     model=self.model_info.id, # Use configured model ID
                     stopReason="endTurn",
                 )

            # Create and enter the ClientSession context
            self._session = ClientSession(
                self._read, self._write, sampling_callback=handle_sampling_message
            )
            self._session = await self._session.__aenter__() 
            self.logger.debug("ClientSession context entered.")

            # Initialize the MCP connection (part of ClientSession context)
            # The example calls session.initialize() explicitly, but it might
            # be handled by ClientSession.__aenter__ depending on the library version.
            # Let's call it explicitly for clarity based on the example.
            self.logger.info("Initializing MCP session...")
            await self._session.initialize()
            self.logger.info("MCP session successfully initialized.")

        except Exception as e:
            self.logger.error(f"Failed to initialize MCP session: {str(e)}", exc_info=True)
            # Attempt cleanup if initialization failed
            await self.terminate()
            raise ModelProviderError(f"MCP session initialization failed: {e}") from e

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
        retry_error_cls=ProviderAPIError,
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
        system_prompt_text: Optional[str] = None # Variable to hold system prompt

        # Extract context if it exists, but DON'T add it to mcp_messages here
        if request.context:
            system_prompt_text = request.context

        # Add previous messages if any (ensure they are user/assistant)
        # for msg in request.history:
        #     if msg.role in ("user", "assistant"): # Filter roles if necessary
        #         mcp_messages.append(types.SamplingMessage(role=msg.role, content=types.TextContent(type="text", text=msg.content)))

        # Add the current user prompt
        mcp_messages.append(types.SamplingMessage(role="user", content=types.TextContent(type="text", text=request.prompt)))

        self.logger.debug(f"mcp_messages: {mcp_messages}")
        self.logger.debug(f"system_prompt_text: {system_prompt_text}")

        # Check if max_tokens is available and required
        if mcp_params.max_tokens is None:
            # setting a default
            mcp_params.max_tokens = 1024 #TODO What should the default be?

        # 4. Prepare MCP Request Parameters
        mcp_request_params = types.CreateMessageRequestParams(
            messages=mcp_messages,
            model=self.model_info.id,
            maxTokens=mcp_params.max_tokens,
            systemPrompt=system_prompt_text, # Pass context here
            # Add other optional fields from mcp_params if they exist and are needed
            # temperature=mcp_params.temperature,
            # stopSequences=mcp_params.stop_sequences,
            # ...
        )

        # Initialize usage with a default UsageStats object
        usage = UsageStats()

        # 5. Call MCP create_message
        self.logger.debug(f"Sending message to MCP: {mcp_request_params}")
        result: types.CreateMessageResult = await self._session.create_message(mcp_request_params)
        self.logger.debug(f"Received message from MCP: {result}")

        # --- Start of post-call processing ---
        # (Optional: Add a new try...except here if needed for result processing errors)
        try:
            if not isinstance(result.content, types.TextContent):
                 raise ModelProviderError(f"Received non-text content from MCP: {type(result.content)}")

            # 7. Format Response
            return ChatResponse(
                data=result.content.text,
                usage=usage, # Pass the initialized UsageStats object
                raw_output=result.model_dump(), # Store raw MCP response
                provider_params=mcp_params.model_dump(),
            )
        except Exception as processing_error:
            # Handle errors during response processing specifically
            self.logger.error(f"Error processing MCP response: {processing_error}", exc_info=True)
            # Decide how to handle processing errors - maybe raise a different error type?
            raise ModelProviderError(f"Failed to process MCP response: {processing_error}") from processing_error
        # --- End of post-call processing ---

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
