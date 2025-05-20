#!/usr/bin/env python3
"""Unit tests for the MCP client implementation."""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ember.core.registry.model.base.schemas.chat_schemas import (
    ChatRequest,
    ChatResponse,
)
from ember.core.registry.model.base.schemas.model_info import ModelInfo
from ember.core.registry.model.base.schemas.provider_info import ProviderInfo
from ember.core.registry.model.providers.base_provider import BaseProviderModel
from ember.core.utils.mcp.client import McpClient


class DummyMcpResponse:
    def __init__(self):
        self.capabilities = type(
            "Capabilities", (), {"prompts": True, "tools": True, "resources": True}
        )


class DummyPromptResponse:
    def __init__(self):
        self.prompts = [type("Prompt", (), {"name": "test_prompt"})]


class DummyToolResponse:
    def __init__(self):
        self.tools = [type("Tool", (), {"name": "test_tool"})]


class DummyResourceResponse:
    def __init__(self):
        self.resources = [type("Resource", (), {"uri": "test_resource"})]


class DummyModel(BaseProviderModel):
    def __init__(self):
        self.model_info = ModelInfo(
            id="test-model",
            name="test-model",
            provider=ProviderInfo(name="test-provider"),
        )

    def forward(self, request: ChatRequest) -> ChatResponse:
        return ChatResponse(data="Test response", model_id=self.model_info.id)

    def create_client(self) -> None:
        """Required implementation of abstract method."""
        pass  # No client needed for dummy model


@pytest.fixture
def mock_logger():
    """Returns a mock logger for testing."""
    logger = MagicMock(spec=logging.Logger)
    return logger


@pytest.fixture
def mcp_client(mock_logger):
    """Returns an MCP client for testing."""
    model = DummyModel()
    return McpClient(
        model=model,
        server_command="echo",
        server_args=["test"],
        server_env={"TEST": "1"},
        logger=mock_logger,
    )


@pytest.mark.asyncio
async def test_connect(mcp_client):
    """Test that connect properly initializes the session."""
    mock_client_session = AsyncMock()
    mock_stdio = AsyncMock()
    mock_init_result = MagicMock()
    mock_init_result.capabilities = DummyMcpResponse().capabilities

    # Patch the session and stdio client creation
    with (
        patch("ember.core.utils.mcp.client.stdio_client", return_value=mock_stdio),
        patch(
            "ember.core.utils.mcp.client.ClientSession",
            return_value=mock_client_session,
        ),
    ):

        mock_stdio.__aenter__.return_value = (AsyncMock(), AsyncMock())
        mock_client_session.initialize.return_value = mock_init_result

        # Call connect
        await mcp_client.connect()

        # Verify initialization
        assert mcp_client._session == mock_client_session
        assert mcp_client._stdio_client == mock_stdio
        assert hasattr(mcp_client._server_capabilities, "prompts")
        assert mcp_client._server_capabilities.prompts is True


@pytest.mark.asyncio
async def test_list_prompts(mcp_client):
    """Test that list_prompts returns the expected prompts."""
    with patch.object(mcp_client, "_session", new_callable=AsyncMock) as mock_session:
        mock_session.list_prompts.return_value = DummyPromptResponse()

        # Set server capabilities to include prompts
        mcp_client._server_capabilities = type("Capabilities", (), {"prompts": True})

        prompts = await mcp_client.list_prompts()
        assert len(prompts) == 1
        assert prompts[0].name == "test_prompt"


@pytest.mark.asyncio
async def test_forward_with_model(mcp_client):
    """Test that forward correctly forwards requests to the underlying model."""
    # Model is already set in fixture
    response = await mcp_client.forward(ChatRequest(prompt="Test prompt"))
    assert response.data == "Test response"


@pytest.mark.asyncio
async def test_forward_no_model(mcp_client):
    """Test that forward raises an error when no model is provided."""
    mcp_client._model = None
    with pytest.raises(Exception, match="No underlying model provided"):
        await mcp_client.forward(ChatRequest(prompt="Test prompt"))


@pytest.mark.asyncio
async def test_terminate(mcp_client):
    """Test that terminate properly cleans up resources."""
    # Create mocks that will be properly cleaned up
    session_mock = AsyncMock()
    stdio_mock = AsyncMock()

    # Create completed futures for __aexit__
    completed_future = asyncio.get_event_loop().create_future()
    completed_future.set_result(None)

    # Set up the mocks
    session_mock.__aexit__.return_value = completed_future
    stdio_mock.__aexit__.return_value = completed_future

    # Patch just the method calls that would exit the context managers
    with (
        patch.object(mcp_client, "_session", session_mock),
        patch.object(mcp_client, "_stdio_client", stdio_mock),
    ):

        # Call terminate
        await mcp_client.terminate()

        # Verify that attributes are reset
        assert mcp_client._session is None
        assert mcp_client._stdio_client is None
