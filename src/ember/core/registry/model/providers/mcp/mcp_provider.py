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
        # Combine base defaults with MCP specific ones if any
        defaults = super().get_default_params()
        defaults.update(self.DEFAULT_PARAMS)
        return defaults

    def _validate_request(
        self, request: ChatRequest, params: McpChatParameters
    ) -> None:
        """Validate the chat request and parameters for MCP."""
        super()._validate_request(request, params)
        # Add any MCP specific validation if needed

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry_error_callback=lambda retry_state: logger.error(
            f"Retrying MCP request after error: {retry_state.outcome.exception()}"
        ),
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
        mcp_params = self._prepare_params(request, params, McpChatParameters)

        # 2. Validate Request
        self._validate_request(request, mcp_params)

        # 3. Convert Ember ChatRequest to MCP message format
        # This conversion depends heavily on the expected MCP message structure.
        # Assuming a simple conversion for now.
        mcp_messages: List[types.Message] = []
        if request.context:
            mcp_messages.append(types.Message(role="system", content=types.TextContent(type="text", text=request.context)))
        # Add previous messages if any (assuming request.history follows similar structure)
        # for msg in request.history:
        #     mcp_messages.append(types.Message(role=msg.role, content=types.TextContent(type="text", text=msg.content)))
        mcp_messages.append(types.Message(role="user", content=types.TextContent(type="text", text=request.prompt)))

        # 4. Prepare MCP Request Parameters
        mcp_request_params = types.CreateMessageRequestParams(
            messages=mcp_messages,
            model=self.model_info.id, # Use the specific model ID
            # Map Ember params to MCP params if applicable
            # temperature=mcp_params.temperature,
            # maxTokens=mcp_params.max_tokens,
            # ... other MCP sampling params
        )

        # Initialize usage with a default UsageStats object as defined in usage.py
        # This creates an instance with total_tokens=0, prompt_tokens=0, etc.
        usage = UsageStats()

        # 5. Call MCP create_message (or equivalent)
        try:
            self.logger.debug(f"Sending message to MCP: {mcp_request_params}")
            # Assuming create_message exists and follows request/response
            # The example uses a sampling_callback, implying the server might initiate
            # the message creation. If the client needs to initiate, the API might differ.
            # Let's assume a client-initiated `create_message` for now.
            # This part needs verification against the actual mcp library API for chat.
            # If chat is purely handled by sampling_callback, this `forward` needs
            # a different approach (e.g., triggering the server somehow).

            # *** Use the actual session method to send the message ***
            result: types.CreateMessageResult = await self._session.create_message(mcp_request_params)
            self.logger.debug(f"Received message from MCP: {result}")

            if not isinstance(result.content, types.TextContent):
                 raise ModelProviderError(f"Received non-text content from MCP: {type(result.content)}")

            # 6. Calculate Usage (if available from MCP response)
            # Placeholder: If MCP provided token counts, update the 'usage' object here.
            # Example (if result had token info):
            # usage = UsageStats(
            #     prompt_tokens=result.usage.prompt_tokens,
            #     completion_tokens=result.usage.completion_tokens,
            #     total_tokens=result.usage.total_tokens
            # )
            # Since MCP doesn't seem to provide this, we stick with the default UsageStats().

            # 7. Format Response
            return ChatResponse(
                model_id=self.model_info.id,
                data=result.content.text,
                usage=usage, # Pass the initialized UsageStats object
                raw_output=result.model_dump(), # Store raw MCP response
                provider_params=mcp_params.model_dump(),
            )

        except Exception as e:
            self.logger.error(f"Error during MCP forward request: {str(e)}", exc_info=True)
            # Convert specific MCP errors if possible
            raise ProviderAPIError(f"MCP API error: {e}") from e


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
