from mcp import ClientSession, StdioServerParameters
import asyncio
import langchain_mcp_adapters
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.agents.react.agent import create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain import hub

# Specify server parameters
server_params = StdioServerParameters(
  command="python",
  args=["math_server.py"],
)

#Specify a model
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o")

#Import stdio client

from mcp.client.stdio import stdio_client

#Start client session with server parameters

async def run_agent():
  async with stdio_client(server_params) as (read, write):
    # Open an MCP session to interact with the math_server.py tool.
    async with ClientSession(read, write) as session:
      # Initialize the session.
      await session.initialize()
      # Load tools
      tools = await load_mcp_tools(session)
      # Create a ReAct agent.

      #default prompt
      prompt = hub.pull("hwchase17/react") #from langchain hub

      agent = create_react_agent(model, tools, prompt)
      # Run the agent.
      agent_response = await agent.ainvoke(
        {
          "input": "what's (4 + 6) x 14?",
          "intermediate_steps": [] #required by ReAct agent
        }
      )
      
      # Return the response - handle AgentAction object correctly
      if hasattr(agent_response, 'output'):
          return agent_response.output
      elif hasattr(agent_response, 'return_values'):
          return agent_response.return_values.get('output')
      else:
          return str(agent_response)  # Fallback to string representation
    

# run async function

if __name__ == "__main__":
  result = asyncio.run(run_agent())
  print(result)