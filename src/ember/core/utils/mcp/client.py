# TODO First making with stdio, will implement SSE later
# TODO Only MCP tool usage is implemented, rest will be added later
import asyncio
import logging
from typing import Any, Dict, Final, List, Optional, Tuple, cast

from mcp.client.stdio import stdio_client
from mcp.server.fastmcp.prompts import base

from ember.core.registry.model.base.schemas.chat_schemas import (
    ChatRequest,
    ChatResponse,
    ProviderParams,
)
from ember.core.registry.model.providers.base_provider import (
    BaseChatParameters,
    BaseProviderModel,
)
from mcp import ClientSession, StdioServerParameters, types

logger: logging.Logger = logging.getLogger(__name__)


class McpClient:
    # Store model for MCP
    _model: Optional[BaseProviderModel] = (
        None  # Optional as a workaround so we can call registry.get_model(), have to inject manually
    )

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
    tools: List[str] = []
    prompts: List[str] = []
    resources: List[str] = []

    def __init__(
        self,
        model: BaseProviderModel,
        server_command: str,
        server_args: Optional[List[str]] = None,
        server_env: Optional[Dict[str, str]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Initialize MCP client utility.

        Args:
            model: The underlying model to use for processing requests
            server_command: Command to start the MCP server (required)
            server_args: Optional list of arguments for the server command
            server_env: Optional environment variables for the server process
            logger: Optional logger instance (defaults to module-level logger)

        Raises:
            Exception: If server_command is not provided
        """
        self.logger = logger  # Use the module-level logger

        # Validate required command
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
            sampling=types.SamplingCapability(), experimental=None, roots=None
        )

        self.logger.debug(
            f"Initialized client capabilities: {self._client_capabilities}"
        )

        # Store the underlying model
        self._model = model

        # Initialize async-related state
        self._session: Optional[ClientSession] = None
        self._stdio_client: Optional[stdio_client] = None
        self._read: Optional[asyncio.StreamReader] = None
        self._write: Optional[asyncio.StreamWriter] = None

        # Initialize server capabilities
        self._server_capabilities: Optional[types.ServerCapabilities] = None

        # Initialize feature storage
        self.tools: List[str] = []
        self.prompts: List[str] = []
        self.resources: List[str] = []

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
            self.logger.info(
                f"Starting MCP server via: {self._server_params.command} {' '.join(self._server_params.args)}"
            )
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
            def _analyze_server_capabilities(
                init_result: types.InitializeResult,
            ) -> None:
                """Analyze and log server capabilities for debugging purposes."""
                self._server_capabilities = init_result.capabilities

                # Log detailed server information and capabilities
                self.logger.info(f"Server info: {init_result.serverInfo}")
                self.logger.info(f"Server capabilities: {init_result.capabilities}")

                # Log individual capabilities for better visibility in logs
                if (
                    hasattr(init_result.capabilities, "prompts")
                    and init_result.capabilities.prompts
                ):
                    self.logger.info("Server supports prompts API")

                if (
                    hasattr(init_result.capabilities, "tools")
                    and init_result.capabilities.tools
                ):
                    self.logger.info("Server supports tools API")

                if (
                    hasattr(init_result.capabilities, "resources")
                    and init_result.capabilities.resources
                ):
                    self.logger.info("Server supports resources API")

            _analyze_server_capabilities(init_result)  # Just for logging

            # If tools API is available, fetch available tools
            await self.fetch_tools()

            # TODO prompts and resources API not implemented yet

            # If prompts API is available, fetch available prompts
            await self.fetch_prompts()

            # If resources API is available, fetch available resources
            await self.fetch_resources()

        except Exception as e:
            self.logger.error(f"Error during MCP initialization: {e}", exc_info=True)
            raise Exception(f"Initialization failed: {e}")

    async def fetch_tools(self):
        if (
            hasattr(self._server_capabilities, "tools")
            and self._server_capabilities.tools
        ):
            try:
                tools_result = await self.list_tools()
                self.tools = [tool.name for tool in tools_result]
                self.logger.info(f"Available tools: {self.tools}")
            except Exception as e:
                self.logger.warning(f"Failed to fetch available tools: {e}")
                self.tools = []
        else:
            self.logger.info("Server does not support tools")

    async def fetch_prompts(self):
        if (
            hasattr(self._server_capabilities, "prompts")
            and self._server_capabilities.prompts
        ):
            try:
                prompts_result = await self.list_prompts()
                self.prompts = [prompt.name for prompt in prompts_result]
                self.logger.info(f"Available prompts: {self.prompts}")
            except Exception as e:
                self.logger.warning(f"Failed to fetch available prompts: {e}")
                self.prompts = []
        else:
            self.logger.info("Server does not support prompts")

    async def fetch_resources(self):
        if (
            hasattr(self._server_capabilities, "resources")
            and self._server_capabilities.resources
        ):
            try:
                resources_result = await self.list_resources()
                self.resources = [resource.uri for resource in resources_result]
                self.logger.info(f"Available resources: {self.resources}")
            except Exception as e:
                self.logger.warning(f"Failed to fetch available resources: {e}")
                self.resources = []
        else:
            self.logger.info("Server does not support resources")

    async def list_tools(self) -> List[types.Tool]:
        """List available tools provided by the server."""
        if not hasattr(self, "_session") or self._session is None:
            await self.connect()

        if (
            not hasattr(self._server_capabilities, "tools")
            or not self._server_capabilities.tools
        ):
            raise Exception("Server doesn't support tools API")
        try:
            tools_result = await self._session.list_tools()
            self.logger.info(
                f"Available tools: {[tool.name for tool in tools_result.tools]}"
            )
            return tools_result.tools
        except Exception as e:
            self.logger.error(f"Failed to list tools: {e}")
            raise Exception(f"Failed to list tools: {e}")

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool provided by the server."""
        if not hasattr(self, "_session") or self._session is None:
            await self.connect()

        if (
            not self._server_capabilities
            or not hasattr(self._server_capabilities, "tools")
            or not self._server_capabilities.tools
        ):
            raise Exception("Server doesn't support tools API")

        try:
            result = await self._session.call_tool(name=tool_name, arguments=arguments)
            return result.content
        except Exception as e:
            self.logger.error(f"Failed to call tool '{tool_name}': {e}")
            raise Exception(f"Failed to call tool '{tool_name}': {e}")

    # Only tools are supported for now
    # Prompts and Resources are not supported yet

    async def list_prompts(self) -> List[types.Prompt]:
        """List available prompts provided by the server."""
        if not hasattr(self, "_session") or self._session is None:
            await self.connect()

        if (
            not hasattr(self._server_capabilities, "prompts")
            or not self._server_capabilities.prompts
        ):
            raise Exception("Server doesn't support prompts API")

        try:
            prompts_result = await self._session.list_prompts()
            self.logger.info(
                f"Available prompts: {[prompt.name for prompt in prompts_result.prompts]}"
            )
            return prompts_result.prompts
        except Exception as e:
            self.logger.error(f"Failed to list prompts: {e}")
            raise Exception(f"Failed to list prompts: {e}")

    async def get_prompt(
        self, prompt_name: str, parameters: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Get a prompt from the server.

        Args:
            prompt_name: Name of the prompt to retrieve
            parameters: Optional parameters to pass to the prompt

        Returns:
            The prompt result from the server

        Raises:
            Exception: If not connected to server, prompt request fails, or prompt is not found/invalid
        """
        if not hasattr(self, "_session") or self._session is None:
            await self.connect()

        if (
            not self._server_capabilities
            or not hasattr(self._server_capabilities, "prompts")
            or not self._server_capabilities.prompts
        ):
            raise Exception("Server doesn't support prompts API")

        try:
            result = await self._session.get_prompt(
                name=prompt_name, arguments=parameters
            )
            return result.content
        except Exception as e:
            self.logger.error(f"Failed to get prompt '{prompt_name}': {e}")
            raise Exception(f"Failed to get prompt '{prompt_name}': {e}")

    async def list_resources(self) -> List[types.Resource]:
        """List available resources provided by the server."""
        if not hasattr(self, "_session") or self._session is None:
            await self.connect()

        if (
            not self._server_capabilities
            or not hasattr(self._server_capabilities, "resources")
            or not self._server_capabilities.resources
        ):
            raise Exception("Server doesn't support resources API")

        try:
            resources_result = await self._session.list_resources()
            self.logger.info(
                f"Available resources: {[resource.uri for resource in resources_result.resources]}"
            )
            return resources_result.resources
        except Exception as e:
            self.logger.error(f"Failed to list resources: {e}")
            raise Exception(f"Failed to list resources: {e}")

    async def read_resource(self, uri: str) -> Any:
        """Read a resource from the server."""
        if not hasattr(self, "_session") or self._session is None:
            await self.connect()

        if (
            not self._server_capabilities
            or not hasattr(self._server_capabilities, "resources")
            or not self._server_capabilities.resources
        ):
            raise Exception("Server doesn't support resources API")

        try:
            result = await self._session.read_resource(uri=uri)
            return result.contents
        except Exception as e:
            self.logger.error(f"Failed to read resource '{uri}': {e}")
            raise Exception(f"Failed to read resource '{uri}': {e}")

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
            for arg_pair in args_str.split(","):
                if "=" in arg_pair:
                    key, value = arg_pair.split("=", 1)
                    args[key.strip()] = value.strip().strip(
                        "\"'"
                    )  # Remove quotes if present

            tool_calls.append({"name": tool_name, "args": args})

        return tool_calls

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
        if not hasattr(self, "_session") or self._session is None:
            await self.connect()

        final_text = []

        try:
            await self.fetch_tools()

            # Prepare the system prompt to inform the model about available tools
            tool_instructions = ""
            if self.tools:
                tools_result = await self.list_tools()
                tool_descriptions = [
                    f"{tool.name}: {tool.description}" for tool in tools_result
                ]

                tool_instructions = f"""
                The following tools are available. If you need to use a tool, respond with:
                Call tool: tool_name(param1=value1, param2=value2)
                
                Available tools:
                {"".join(tool_descriptions)}
                """.strip(
                    "\n"
                )

            # Create the ChatRequest with proper provider_params
            chat_request = ChatRequest(
                prompt=query,
                context=tool_instructions if tool_instructions else None,
            )
            print(f"Chat Request: {chat_request}")
            # 4. Send initial request to the model through MCP

            response = self._model.forward(chat_request)  # model forwards aren't async

            final_text.append(response.data)

            # 5. Check if the response contains tool calls
            tool_calls = self._extract_tool_calls(response.data)

            # 6. Process tool calls if any
            if tool_calls and self.tools:
                for tool_call in tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]

                    self.logger.info(
                        f"Executing tool: {tool_name} with args: {tool_args}"
                    )
                    final_text.append(f"\n[Calling tool: {tool_name}]")

                    # Just with tools for now
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
                            followup_response = await self._model.forward(
                                ChatRequest(prompt=followup_prompt)
                            )

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

        Raises:
            Exception: If no model is provided or if processing fails
        """
        if not self._model:
            raise Exception("No underlying model provided")

        try:
            if self.tools:  # tools should only be truthy if server supports tools
                self.logger.info("Using process_query to handle potential tool usage")
                result_text = await self.process_query(request.prompt)
                return ChatResponse(
                    data=result_text, model_id=self._model.model_info.id
                )

            self.logger.info("Using underlying model directly")
            return self._model.forward(request)

        except Exception as e:
            self.logger.error(f"Error processing request: {e}", exc_info=True)
            raise Exception(f"Request processing failed: {e}")

    async def terminate(self) -> None:
        """
        Cleanly terminates the MCP client session and transport by exiting contexts.

        Raises:
            Exception: If there are any errors during termination
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
            self.logger.error(
                f"Error exiting ClientSession context: {e}", exc_info=True
            )
        finally:
            self._session = None
            self._session = None

        try:
            if self._stdio_client:
                self.logger.debug("Exiting stdio_client context...")
                await self._stdio_client.__aexit__(None, None, None)
                self.logger.debug("stdio_client context exited.")
        except Exception as e:
            if not exit_exception:
                exit_exception = e
            self.logger.error(f"Error exiting stdio_client context: {e}", exc_info=True)
        finally:
            self._read = None
            self._write = None
            self._stdio_client = None

        if exit_exception:
            raise Exception(
                f"Error during MCP termination: {exit_exception}"
            ) from exit_exception
        else:
            self.logger.info("MCP provider terminated successfully.")
