"""
Direct MCP Test without Ember Provider 

Tests the MCP server using the raw MCP client, bypassing our wrapper code.
"""

import asyncio
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CreateMessageRequest, CreateMessageResult, TextContent


async def main():
    print("Setting up MCP connection...")
    
    # Create server parameters with potential stderr logging
    params = StdioServerParameters(
        command="python",
        args=["-m", "ember.examples.mcp.example_mcp_server"],
        capture_stderr=True  # Capture stderr for debugging
    )
    
    # Connect directly using the MCP client SDK
    print("Connecting to server...")
    async with stdio_client(params) as (read, write):
        print("Connection established, creating session...")
        
        # Use ClientSession as a context manager
        async with ClientSession(read, write) as session:
            # Skip explicit initialization which might be redundant
            # Directly list resources
            print("\nListing resources...")
            try:
                resources = await asyncio.wait_for(session.list_resources(), timeout=5.0)
                print(f"Found {len(resources)} resources:")
                for r in resources:
                    print(f"  - {r.name}: {r.description}")
            except asyncio.TimeoutError:
                print("Resource listing timed out!")
            except Exception as e:
                print(f"Error listing resources: {e}")
    print("Setting up MCP connection...")
        
    # Initialize the session
    print("Initializing session...")
    try:
        init_result = await asyncio.wait_for(session.initialize(), timeout=5.0)
        print(f"Connected to: {init_result.server_info.name} v{init_result.server_info.version}")
    except asyncio.TimeoutError:
        print("Session initialization timed out!")
        return
    except Exception as e:
        print(f"Error initializing session: {e}")
        return
    
    # Try listing resources
    print("\nListing resources...")
    try:
        resources = await asyncio.wait_for(session.list_resources(), timeout=5.0)
        print(f"Found {len(resources)} resources:")
        for r in resources:
            print(f"  - {r.name}: {r.description}")
    except asyncio.TimeoutError:
        print("Resource listing timed out!")
    except Exception as e:
        print(f"Error listing resources: {e}")
    
    # Try listing tools
    print("\nListing tools...")
    try:
        tools = await asyncio.wait_for(session.list_tools(), timeout=5.0)
        print(f"Found {len(tools)} tools:")
        for t in tools:
            print(f"  - {t.name}: {t.description}")
    except asyncio.TimeoutError:
        print("Tool listing timed out!")
    except Exception as e:
        print(f"Error listing tools: {e}")
    
    # Close the session properly
    print("\nClosing session...")
    await session.close()


if __name__ == "__main__":
    print("Starting direct MCP test...")
    asyncio.run(main()) 