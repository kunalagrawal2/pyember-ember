"""
Simple MCP Server

A simplified MCP server with minimal functionality for testing connection issues.
"""

import asyncio
import logging
import sys
from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("simple_mcp_server")
logger.debug("Starting server setup")

# Disable output buffering
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
logger.debug("Disabled output buffering")

# Create the server
mcp = FastMCP(name="Simple MCP Server", version="0.1")
logger.debug("Created FastMCP instance")

# Add a simple resource
@mcp.resource("test://hello")
def hello_resource():
    """A simple test resource."""
    logger.debug("Serving hello resource")
    return "Hello, world!"

# Add a simple tool
@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    logger.debug(f"Adding {a} + {b}")
    return a + b

if __name__ == "__main__":
    logger.debug("Starting MCP server")
    
    try:
        # Run the server
        asyncio.run(mcp.run())
    except Exception as e:
        logger.error(f"Error running server: {e}") 