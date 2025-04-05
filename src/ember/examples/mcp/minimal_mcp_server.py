"""
Minimal MCP Server

A bare-bones MCP server for debugging connection issues.
"""

import asyncio
import logging
import sys
from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("minimal_mcp_server")

# Disable output buffering
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Create a minimal server
mcp = FastMCP(name="Minimal MCP Server", version="0.1")

# Add a single test resource
@mcp.resource("test://hello")
def hello_resource():
    """A simple test resource."""
    return "Hello, world!"

# Simple sampling handler
async def handle_sample(request):
    """Handle sampling requests."""
    return {
        "content": {
            "type": "text",
            "text": "This is a test response."
        }
    }

# Try to register sampling handler in different ways
try:
    # Try registering with register_sampling method
    mcp.register_sampling(handle_sample)
    logger.debug("Registered sampling handler with register_sampling")
except Exception as e:
    logger.error(f"Error registering sampling with register_sampling: {e}")
    
    try:
        # Try alternative registration method
        mcp._sampling_handler = handle_sample
        logger.debug("Set sampling handler directly")
    except Exception as e:
        logger.error(f"Error setting sampling handler directly: {e}")

if __name__ == "__main__":
    logger.debug("Starting minimal MCP server")
    
    # Print server configuration
    logger.debug(f"Server info: {mcp.name} v{mcp.version}")
    logger.debug(f"Resources: {len(mcp._resources)}")
    
    # Run the server
    asyncio.run(mcp.run()) 