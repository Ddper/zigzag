import os
import json

from enum import Enum, auto
from typing import Union, Callable, Dict, List
from pydantic import BaseModel, Field
from openai import OpenAI
from tavily import TavilyClient

from dotenv import load_dotenv

from zigzag.prompts import system_prompt
from zigzag.settings import settings

load_dotenv()

Observation = Union[str, Exception]


class Message(BaseModel):
    """
    Represents a message with sender role and content.
    """
    role: str = Field(..., description="The role of the message sender.")
    content: str = Field(..., description="The content of the message.")


class Tool:
    """
    A wrapper class for tools used by the agent, executing a function based on tool type.
    """

    def __init__(self, name: str, description: str, func: Callable[[str], str]):
        """
        Initializes a Tool with a name and an associated function.
        :param name: The name of the tool.
        :param func: The function associated with the tool.
        """
        self.name = name
        self.description = description
        self.func = func

    def use(self, query: str) -> Observation:
        """
        Executes the tool's function with the provided query.
        :param query: The input query for the tool.
        :return: Result of the tool's function or an error message if an exception occurs.
        """
        return self.func(query)


class ReActAgent:

    def __init__(self, client: OpenAI, tools: List[Tool]) -> None:
        self.client = client
        self.tools = tools
        self.max_iterations = 5
        self.template = system_prompt
        tool_desc = "\n\n".join([f"Name: {tool.name}, Description: {tool.description}" for tool in tools])
        tool_names = ",".join([tool.name for tool in tools])
        self.messages = [{"role": "system", "content": self.template.format(tool_desc=tool_desc, tool_names=tool_names)}]

    def call_llm(self) -> Dict[str, str | Dict[str, str]]:
        response = self.client.chat.completions.create(
            model=settings.openai_model_name,
            messages=self.messages,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": content})
        d = json.loads(content)
        for k, v in d.items():
            print(f"{k}: {v}")
        return d

    def run(self, query: str):
        self.messages.append({"role": "user", "content": query})
        current_iteration = 0
        while current_iteration < self.max_iterations:
            print(f"current_iteration = {current_iteration}")
            response = self.call_llm()
            if response.get("Answer"):
                return response["Answer"]
            if response.get("Action"):
                observation = web_search(response["Action Input"]["input"])
                print(f"Observation: {observation}")
                print("\n")
                self.messages.append({
                    "role": "user",
                    "content": json.dumps({"Observation": observation})
                })

            current_iteration += 1


def add(a: int, b: int) -> int:
    """ Add two integers and returns the result integer"""
    return a + b


def web_search(query: str) -> str:
    client = TavilyClient(api_key=settings.tavily_api_key)
    response = client.search(
        query=query,
        include_answer=True
    )
    return response.get("answer")


def main():
    client = OpenAI(
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
    )
    agent = ReActAgent(client, [
        # Tool("add", "Add two integers and returns the result integer", add),
        Tool("web_search", "Tavily search for more information", web_search)
    ])
    # print(agent.run("What is 5+3+2?"))
    # print(agent.run("What is the age of the oldest tree in the country that has won the most FIFA World Cup titles?"))
    print(agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?"))


if __name__ == "__main__":
    main()
