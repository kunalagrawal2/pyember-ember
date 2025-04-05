"""
Simple MCP Client

A simplified MCP client for testing connection to the simple server.
"""

import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    print("Setting up MCP connection...")
    
    # Create server parameters
    params = StdioServerParameters(
        command="python",
        args=["-m", "ember.examples.mcp.simple_mcp_server"],
        capture_stderr=True
    )
    
    try:
        # Connect to the server with a timeout
        print("Connecting to server...")
        connection_task = stdio_client(params)
        
        # Add a timeout to the connection
        async with await asyncio.wait_for(connection_task, timeout=5.0) as (read, write):
            print("Connection established")
            
            # Create session with explicit timeout
            async with ClientSession(read, write) as session:
                # Initialize the session
                print("Initializing session...")
                init_result = await asyncio.wait_for(session.initialize(), timeout=5.0)
                print(f"Connected to: {init_result.server_info.name} v{init_result.server_info.version}")
                
                # List resources
                print("\nListing resources...")
                resources = await asyncio.wait_for(session.list_resources(), timeout=5.0)
                print(f"Found {len(resources)} resources:")
                for r in resources:
                    print(f"  - {r.name}: {r.description}")
                
                # List tools
                print("\nListing tools...")
                tools = await asyncio.wait_for(session.list_tools(), timeout=5.0)
                print(f"Found {len(tools)} tools:")
                for t in tools:
                    print(f"  - {t.name}: {t.description}")
                
                # Call a tool
                print("\nCalling tool...")
                result = await asyncio.wait_for(session.call_tool("add", {"a": 5, "b": 3}), timeout=5.0)
                print(f"5 + 3 = {result}")
                
                print("\nTests completed successfully!")
    
    except asyncio.TimeoutError:
        print("Connection or operation timed out!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("Starting simple MCP client test...")
    asyncio.run(main()) 