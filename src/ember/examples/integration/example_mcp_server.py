# Basic Echo Server to show Ember MCP Integration

from mcp.server.fastmcp import FastMCP
import logging
import sys

# Configure more verbose logging
logging.basicConfig(
    level=logging.DEBUG,  # Change to DEBUG level
    format='%(asctime)s [MCP Server - %(levelname)s] %(message)s',
    stream=sys.stderr  # Explicitly write to stderr
)
logger = logging.getLogger(__name__)

# Flush stdout/stderr immediately
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

mcp = FastMCP("Echo")

@mcp.resource("echo://{message}")
def echo_resource(message: str) -> str:
    """Echo a message as a resource"""
    return f"Resource echo: {message}"


@mcp.tool()
def echo_tool(message: str) -> str:
    """Echo a message as a tool"""
    return f"Tool echo: {message}"


@mcp.prompt()
def echo_prompt(message: str) -> str:
    """Create an echo prompt"""
    logger.info(f"Received prompt request with message: '{message}'")
    response_text = f"Server echo: {message}"
    logger.info(f"Sending response: '{response_text}'")
    sys.stdout.flush()  # Ensure response is sent immediately
    return response_text

if __name__ == "__main__":
    logger.info("Starting MCP Echo Server...")
    try:
        mcp.run()
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)