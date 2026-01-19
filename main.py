import os
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()


def main():
    print("Hello from langchain-course!")
    information = """
    Elon Musk is a businessman who leads Tesla, SpaceX, and xAI.
    Born in South Africa in 1971, he's now the world's wealthiest person.
    """
    summary_template = """Given the information {information} about a person, I want you to create:
1. A short summary
2. 2 interesting facts about them"""
    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    ) # it is better then using fstring as it will be easier to debug if prompt is broken
    
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        max_retries=0
    )
    # runnable chain
    # summary_prompt_template -> llm
    chain = summary_prompt_template | llm # langchain expression language
    response = chain.invoke({"information": information})
    print("Response from LLM:", response.content)

if __name__ == "__main__":
    main()
