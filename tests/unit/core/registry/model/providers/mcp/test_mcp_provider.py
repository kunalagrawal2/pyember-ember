#!/usr/bin/env python3
"""Unit tests for the MCP client implementation using stdio_client."""

import asyncio
from typing import Any, AsyncGenerator
import pytest
from pytest_asyncio import fixture as async_fixture # Import the specific decorator
from unittest.mock import AsyncMock, MagicMock, patch

# Ember imports
from ember.core.registry.model.base.schemas.chat_schemas import ChatRequest, ChatResponse
from ember.core.registry.model.base.schemas.model_info import ModelInfo
from ember.core.registry.model.base.schemas.provider_info import ProviderInfo
# Import the renamed class and other components
from ember.core.registry.model.providers.mcp.mcp_provider import (
    McpClient, # Renamed from McpProvider
    ConfigurationError,
)
from ember.core.exceptions import ProviderAPIError, ModelProviderError
# Import base class for mocking
from ember.core.registry.model.providers.base_provider import BaseProviderModel

# MCP imports
from mcp import ClientSession, StdioServerParameters, types

# --- Test Helper Classes (Optional, can use MagicMock directly) ---
class DummyMessage:
    def __init__(self, content: str) -> None:
        self.content = content

class DummyChoice:
    def __init__(self, message_content: str) -> None:
        self.message = DummyMessage(message_content)

class DummyOpenAIResponse: # Keep if needed for mocking the wrapped model
    def __init__(self) -> None:
        self.choices = [DummyChoice("Test response.")]
        self.usage = type(
            "Usage", (), {"total_tokens": 100, "prompt_tokens": 40, "completion_tokens": 60}
        )

# --- Helper Functions & Fixtures ---

def create_dummy_model_info(custom_args: dict = None) -> ModelInfo:
    """Creates a dummy ModelInfo for testing."""
    if custom_args is None:
        # Define the default custom_args needed for MCP
        custom_args = {
            "command": "dummy_server_command",
            # Store args as a single space-separated string
            "args": "--port 1234"
            # Add "env": '{"VAR": "value"}' here if needed (JSON string)
        }
    # Ensure all values in custom_args are strings
    string_custom_args = {k: str(v) for k, v in custom_args.items()}

    return ModelInfo(
        id="mcp:dummy-model",
        name="dummy-model",
        # Use custom_args here
        provider=ProviderInfo(name="mcp", custom_args=string_custom_args),
    )

@pytest.fixture
def model_info() -> ModelInfo:
    """Provides a standard dummy ModelInfo."""
    return create_dummy_model_info()

@pytest.fixture
def mock_wrapped_model() -> MagicMock:
    """Provides a mock BaseProviderModel to be wrapped by McpClient."""
    mock_model = MagicMock(spec=BaseProviderModel)
    # Mock methods if the McpClient interacts with the wrapped model directly
    # mock_model.forward = AsyncMock(...)
    return mock_model

@pytest.fixture # This one is synchronous, so @pytest.fixture is fine
def mock_stdio_client_cm() -> MagicMock:
    """Mocks the stdio_client async context manager instance."""
    mock_cm = MagicMock()
    mock_cm.__aenter__ = AsyncMock(return_value=(AsyncMock(spec=asyncio.StreamReader), AsyncMock(spec=asyncio.StreamWriter)))
    mock_cm.__aexit__ = AsyncMock(return_value=None)
    return mock_cm

@async_fixture # Use @pytest_asyncio.fixture (imported as async_fixture)
async def mock_stdio_client_constructor(mock_stdio_client_cm: MagicMock) -> AsyncGenerator[MagicMock, None]:
    """Patches the stdio_client constructor."""
    with patch("ember.core.registry.model.providers.mcp.mcp_provider.stdio_client", return_value=mock_stdio_client_cm) as mock_constructor:
        yield mock_constructor

@pytest.fixture # Synchronous fixture
def mock_session_object() -> MagicMock:
    """Provides a mock ClientSession object (the result of __aenter__)."""
    mock_session = MagicMock(spec=ClientSession)
    mock_session.initialize = AsyncMock()
    mock_session.create_message = AsyncMock(
        return_value=types.CreateMessageResult(
            role="assistant",
            content=types.TextContent(type="text", text="Test MCP response."),
            model="dummy-model",
            stopReason="endTurn",
        )
    )
    mock_session.__aexit__ = AsyncMock(return_value=None)
    return mock_session

@pytest.fixture # Synchronous fixture
def mock_session_cm(mock_session_object: MagicMock) -> MagicMock:
    """Provides a mock ClientSession context manager instance."""
    mock_cm = MagicMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_session_object)
    mock_cm.__aexit__ = AsyncMock(return_value=None)
    return mock_cm

@async_fixture # Use @pytest_asyncio.fixture
async def mock_session_constructor(mock_session_cm: MagicMock) -> AsyncGenerator[MagicMock, None]:
    """Patches the ClientSession constructor."""
    with patch("ember.core.registry.model.providers.mcp.mcp_provider.ClientSession", return_value=mock_session_cm) as mock_constructor:
        yield mock_constructor


@async_fixture # Use @pytest_asyncio.fixture for the main async generator fixture
async def mcp_client_fixture(
    model_info: ModelInfo,
    mock_wrapped_model: MagicMock,
    mock_stdio_client_constructor: MagicMock,
    mock_session_constructor: MagicMock,
) -> AsyncGenerator[McpClient, None]:
    """Fixture that provides an McpClient instance with mocked dependencies."""
    client = McpClient(model_info, mock_wrapped_model)
    yield client
    try:
        await client.terminate()
    except Exception as e:
        # Log or ignore termination errors during test cleanup if necessary
        print(f"Ignoring error during fixture cleanup: {e}") # Added print for visibility
        pass


# --- Test Cases ---

def test_init_success(model_info: ModelInfo, mock_wrapped_model: MagicMock) -> None:
    """Test successful initialization with valid config."""
    client = McpClient(model_info, mock_wrapped_model)
    assert client.model_info == model_info
    # Check command from custom_args
    assert client._server_params.command == model_info.provider.custom_args["command"]
    # Check args: compare the parsed list in _server_params
    # with the expected list derived from the string in custom_args
    expected_args = model_info.provider.custom_args["args"].split()
    assert client._server_params.args == expected_args
    assert client._session is None
    assert client._model == mock_wrapped_model

def test_init_failure_missing_command(mock_wrapped_model: MagicMock) -> None:
    """Test initialization failure when 'command' is missing."""
    # Create info with custom_args missing the 'command' key
    # Ensure 'args' value is also a string
    bad_info = create_dummy_model_info(custom_args={"args": "--port 5678"})
    with pytest.raises(ConfigurationError, match="requires 'command'"):
        McpClient(bad_info, mock_wrapped_model)

# Make this test async because it uses the async mcp_client_fixture
@pytest.mark.asyncio
async def test_create_client(mcp_client_fixture: McpClient) -> None:
    """Test that create_client returns None as expected."""
    # Now mcp_client_fixture should be the resolved McpClient instance
    assert mcp_client_fixture.create_client() is None
    assert mcp_client_fixture.client is None

@pytest.mark.asyncio
async def test_initialize_session_success(
    mcp_client_fixture: McpClient,
    mock_stdio_client_constructor: MagicMock,
    mock_session_constructor: MagicMock,
    mock_session_object: MagicMock,
) -> None:
    """Test successful session initialization."""
    # mcp_client_fixture should now be the McpClient instance
    await mcp_client_fixture.initialize_session()

    # Check constructors were called
    mock_stdio_client_constructor.assert_called_once()
    mock_session_constructor.assert_called_once()

    # Check context manager methods were awaited
    mock_stdio_client_constructor.return_value.__aenter__.assert_awaited_once()
    mock_session_constructor.return_value.__aenter__.assert_awaited_once()

    # Check session object methods were awaited
    mock_session_object.initialize.assert_awaited_once()

    # Check internal state
    assert mcp_client_fixture._session == mock_session_object # Should hold the session object
    assert mcp_client_fixture._stdio_client is not None # Holds the stdio context manager

@pytest.mark.asyncio
async def test_initialize_session_failure(
    mcp_client_fixture: McpClient,
    mock_stdio_client_cm: MagicMock,
    mock_session_cm: MagicMock,
    mock_session_object: MagicMock,
) -> None:
    """Test failure during session.initialize()."""
    mock_session_object.initialize.side_effect = Exception("init failed")

    with pytest.raises(ModelProviderError, match="MCP session initialization failed: init failed"):
        # mcp_client_fixture should now be the McpClient instance
        await mcp_client_fixture.initialize_session()

    # Check __aexit__ was called on the context managers/objects used in terminate
    mock_session_object.__aexit__.assert_awaited_once() # Called on the session object
    mock_stdio_client_cm.__aexit__.assert_awaited_once() # Called on the stdio client cm

    assert mcp_client_fixture._session is None # Should be cleaned up

@pytest.mark.asyncio
async def test_forward_success(mcp_client_fixture: McpClient, mock_session_object: MagicMock) -> None:
    """Test successful forward call (implicitly initializes session)."""
    request = ChatRequest(prompt="Hello MCP", context="System prompt")
    # mcp_client_fixture should now be the McpClient instance
    response = await mcp_client_fixture.forward(request)

    # Verify session was initialized and create_message called
    mock_session_object.initialize.assert_awaited_once() # Implicit init
    mock_session_object.create_message.assert_awaited_once()

    # Verify response
    assert isinstance(response, ChatResponse)
    assert response.data == "Test MCP response."

    # Verify call arguments to create_message
    call_args, _ = mock_session_object.create_message.call_args
    mcp_params = call_args[0]
    assert isinstance(mcp_params, types.CreateMessageRequestParams)
    assert mcp_params.messages[0].content.text == "Hello MCP"

@pytest.mark.asyncio
async def test_forward_api_call_failure(mcp_client_fixture: McpClient, mock_session_object: MagicMock) -> None:
    """Test forward when the MCP API call (create_message) fails."""
    await mcp_client_fixture.initialize_session()
    # This mock will now be reached
    mock_session_object.create_message.side_effect = ProviderAPIError("MCP API Error")
    request = ChatRequest(prompt="Hello MCP")

    # Expect ProviderAPIError because retry_error_cls is set.
    # Match against the message from the *mocked* exception, which will be wrapped.
    with pytest.raises(Exception):
        await mcp_client_fixture.forward(request)

    # Now that the AttributeError is fixed, create_message will be called by retry.
    # Verify it was called exactly 3 times before failing.
    assert mock_session_object.create_message.call_count == 3

@pytest.mark.asyncio
async def test_terminate_success(
    mcp_client_fixture: McpClient,
    mock_stdio_client_cm: MagicMock,
    mock_session_object: MagicMock,
) -> None:
    """Test successful termination after initialization."""
    # mcp_client_fixture should now be the McpClient instance
    await mcp_client_fixture.initialize_session()
    stdio_cm = mcp_client_fixture._stdio_client # Should hold the stdio cm
    session_obj = mcp_client_fixture._session # Should hold the session object

    assert stdio_cm is not None
    assert session_obj is not None

    # Terminate
    await mcp_client_fixture.terminate()

    # Verify __aexit__ was called on the correct objects based on terminate() impl
    session_obj.__aexit__.assert_awaited_once_with(None, None, None)
    stdio_cm.__aexit__.assert_awaited_once_with(None, None, None)

    # Verify state is reset
    assert mcp_client_fixture._session is None
    assert mcp_client_fixture._stdio_client is None

@pytest.mark.asyncio
async def test_terminate_failure(
    mcp_client_fixture: McpClient,
    mock_stdio_client_cm: MagicMock,
    mock_session_object: MagicMock,
) -> None:
    """Test termination when a context exit fails."""
    # mcp_client_fixture should now be the McpClient instance
    await mcp_client_fixture.initialize_session()
    stdio_cm = mcp_client_fixture._stdio_client
    session_obj = mcp_client_fixture._session

    # Setup failure on the session object's __aexit__
    session_obj.__aexit__.side_effect = Exception("Session exit failed")

    # Expect exception and check both exits were still attempted
    with pytest.raises(ModelProviderError, match="Session exit failed"):
        await mcp_client_fixture.terminate()

    session_obj.__aexit__.assert_awaited_once()
    stdio_cm.__aexit__.assert_awaited_once()

    # Verify state is reset despite error
    assert mcp_client_fixture._session is None
    assert mcp_client_fixture._stdio_client is None 