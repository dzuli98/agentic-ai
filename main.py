import os
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from tavily import TavilyClient


from dotenv import load_dotenv
load_dotenv()

tavily = TavilyClient()
# Define a simple search tool
# always add the description for the tool as it helps the agent to decide when to use it
@tool
def search(query: str) -> str:
    """
    Tool that searches over the internet.
    Args:
        query: The query to search for.
    Returns:
        The search result
    """
    print(f"Search results for '{query}'")
    return tavily.search(query)

llm = ChatOllama(
        model="qwen3:0.6b",  
        max_retries=0
    )
tools = [search]
agent = create_agent(model = llm, tools = tools)

def main():
    print('Hello!')
    result = agent.invoke(
        {
            "messages": [HumanMessage(content="What's the weather like in Tokyo?")]
        }
    )
    print(f"Agent result: {result}")
if __name__ == "__main__":
    main()
