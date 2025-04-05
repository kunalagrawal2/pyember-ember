"""
Minimal MCP Test

An extremely simplified test that focuses on just connecting to the MCP server.
"""

import asyncio
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_callback(session):
    """Simple callback to test the connection."""
    print("Connected to MCP server! Running callback...")
    
    # Initialize the session
    await session.initialize()
    print("Initialization complete")
    
    # List resources 
    resources = await session.list_resources()
    print(f"Found {len(resources)} resources")
    
    # List tools
    tools = await session.list_tools()
    print(f"Found {len(tools)} tools")
    
    return "Test completed successfully"

async def main():
    """Main function."""
    print(f"Starting test from {__file__}")
    
    # Create parameters for the server
    params = StdioServerParameters(
        command="python",
        args=["-m", "ember.examples.mcp.simple_mcp_server"],
        capture_stderr=True
    )
    
    # Connect directly without going through our wrapper
    try:
        print("Connecting to MCP server...")
        async with stdio_client(params) as (read, write):
            print("Connection established")
            
            # Create and use the session
            session = ClientSession(read, write)
            try:
                result = await test_callback(session)
                print(f"Result: {result}")
            finally:
                # Make sure we close the session
                await session.close()
                print("Session closed")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")

# Run the test directly without using any of our wrapper code
if __name__ == "__main__":
    print(f"Python version: {sys.version}")
    asyncio.run(main()) 