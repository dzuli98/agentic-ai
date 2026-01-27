from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from typing import List
from pydantic import BaseModel, Field
from langchain_tavily import TavilySearch


from dotenv import load_dotenv

load_dotenv()

class Source(BaseModel):
    url: str = Field(..., description="The URL of the source document.")

class AgentResponse(BaseModel):
    answer: str = Field(..., description="The final answer to the user's question.")
    sources: List[Source] = Field(default_factory=list, description="List of source documents used to generate the answer.")


llm = ChatOllama(model="qwen3:0.6b", max_retries=0)
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse) # this ollama model doenst support this but OpenAI


def main():
    print("Hello!")
    result = agent.invoke(
        {"messages": [HumanMessage(content="Search for 2 job postings using langchain for python developer in belgrade on linkedin?")]}
    )
    print(result)


if __name__ == "__main__":
    main()
