"""
Example MCP Server

This module implements a simple MCP server using the FastMCP framework.
It provides resources, tools, and prompts that can be used to test the
McpProvider implementation.

To run this server:
    python -m src.ember.examples.mcp.example_mcp_server

Then in another terminal, you can connect to it using the McpProvider.
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent, ImageContent

# Configure the MCP server
mcp = FastMCP(name="Ember Example MCP Server", version="1.0")

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("example_mcp_server")

# Log when server starts
logger.debug("Starting MCP server setup")

# Important: Disable output buffering to prevent stdio deadlocks
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
logger.debug("Disabled output buffering")

# Sample data for our server
SAMPLE_DATA = {
    "users": [
        {"id": 1, "name": "Alice", "email": "alice@example.com"},
        {"id": 2, "name": "Bob", "email": "bob@example.com"},
        {"id": 3, "name": "Charlie", "email": "charlie@example.com"}
    ],
    "products": [
        {"id": 101, "name": "Laptop", "price": 999.99},
        {"id": 102, "name": "Smartphone", "price": 499.99},
        {"id": 103, "name": "Headphones", "price": 149.99}
    ]
}

# ===== Resources =====

@mcp.resource("data://users")
def get_users() -> str:
    """Get the list of users as JSON."""
    return json.dumps(SAMPLE_DATA["users"], indent=2)

@mcp.resource("data://products")
def get_products() -> str:
    """Get the list of products as JSON."""
    return json.dumps(SAMPLE_DATA["products"], indent=2)

@mcp.resource("data://users/{user_id}")
def get_user(user_id: int) -> str:
    """Get a specific user by ID."""
    for user in SAMPLE_DATA["users"]:
        if user["id"] == user_id:
            return json.dumps(user, indent=2)
    return json.dumps({"error": "User not found"}, indent=2)

@mcp.resource("file://{path}")
def get_file(path: str) -> str:
    """Get the contents of a file."""
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

# ===== Tools =====

@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b

@mcp.tool()
def search_users(query: str) -> List[Dict[str, Any]]:
    """Search for users by name or email."""
    results = []
    query = query.lower()
    for user in SAMPLE_DATA["users"]:
        if query in user["name"].lower() or query in user["email"].lower():
            results.append(user)
    return results

@mcp.tool()
def get_current_time() -> str:
    """Get the current server time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ===== Prompts =====

@mcp.prompt("greeting")
def greeting_prompt(name: str = "User") -> str:
    """Generate a greeting prompt."""
    return f"""
Hello, {name}!

I'm an AI assistant connected to the Ember Example MCP Server.
I can help you with various tasks like:
- Retrieving user and product information
- Performing calculations
- Searching for users
- Getting the current time

How can I assist you today?
"""

@mcp.prompt("data_analysis")
def data_analysis_prompt() -> str:
    """Generate a data analysis prompt."""
    return """
I'll help you analyze the available data. Here's what we have:

1. Users data: Access with data://users
2. Products data: Access with data://products

What specific analysis would you like me to perform?
"""

# ===== Sampling =====
async def handle_sampling(request: Dict[str, Any]) -> Dict[str, Any]:
    """Handle sampling requests by simulating an AI response."""
    # Extract the user message
    messages = request.get("messages", [])
    user_message = "No message provided"
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", {})
            if isinstance(content, dict) and "text" in content:
                user_message = content["text"]
            elif isinstance(content, str):
                user_message = content
            break
    
    # Generate a simple response
    response = f"You said: {user_message}\n\nThis is a simulated AI response from the MCP server."
    
    # If there's a system prompt, acknowledge it
    system_prompt = request.get("systemPrompt")
    if system_prompt:
        response += f"\n\nI'm following this system prompt: {system_prompt}"
    
    # Return the response in the expected format
    return {
        "content": {
            "type": "text",
            "text": response
        }
    }

# Register sampling# Register sampling
mcp.set_sampling_handler(handle_sampling)

# # ===== Main =====

if __name__ == "__main__":
    # Run the server
    print("Starting MCP server on stdio...")
    print("Use Ctrl+C to stop the server")
    
    # Create a directory for testing file resources if it doesn't exist
    os.makedirs("mcp_test_files", exist_ok=True)
    
    # Create a sample file for testing
    with open("mcp_test_files/sample.txt", "w") as f:
        f.write("This is a sample file for testing the MCP file resource.")
    
    print(f"Created sample file at: {os.path.abspath('mcp_test_files/sample.txt')}")
    print("You can access this file using the resource: file://mcp_test_files/sample.txt")
    
    # Run the server
    asyncio.run(mcp.run()) 