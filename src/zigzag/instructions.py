"""
Chat instructions generator for zigzag agents.

This module provides functionality to generate formatted instructions
for interacting with various agent types in the zigzag framework.
"""

from typing import List, Optional, Any


def generate_chat_instructions(
    agent_type: str = "react",
    tools: Optional[List[Any]] = None,
    custom_context: Optional[str] = None
) -> str:
    """
    Generate formatted chat instructions for an agent.
    
    Args:
        agent_type: The type of agent (e.g., 'react')
        tools: List of tools available to the agent
        custom_context: Optional custom context to include
        
    Returns:
        Formatted instruction string for chat interactions
    """
    if agent_type.lower() == "react":
        return _generate_react_instructions(tools, custom_context)
    else:
        return _generate_generic_instructions(agent_type, tools, custom_context)


def _generate_react_instructions(
    tools: Optional[List[Any]] = None,
    custom_context: Optional[str] = None
) -> str:
    """Generate instructions for ReAct agent."""
    instructions = []
    
    instructions.append("# Chat Instructions for ReAct Agent\n")
    instructions.append("## Overview")
    instructions.append("This agent uses the ReAct (Reasoning and Acting) paradigm to solve problems.")
    instructions.append("It thinks step-by-step and uses tools to gather information.\n")
    
    instructions.append("## How to Interact")
    instructions.append("1. Ask your question in natural language")
    instructions.append("2. The agent will think and decide which tools to use")
    instructions.append("3. The agent will provide a final answer based on gathered information\n")
    
    if tools:
        instructions.append("## Available Tools")
        for tool in tools:
            tool_name = getattr(tool, 'name', str(tool))
            tool_desc = getattr(tool, 'description', 'No description available')
            if not tool_desc:  # Handle empty string case
                tool_desc = 'No description available'
            instructions.append(f"- **{tool_name}**: {tool_desc}")
        instructions.append("")
    
    instructions.append("## Example Questions")
    instructions.append("- What is the age of the oldest tree in the country that has won the most FIFA World Cup titles?")
    instructions.append("- How many seconds would it take for a leopard at full speed to run through Pont des Arts?")
    instructions.append("- What is the current weather in Tokyo?\n")
    
    if custom_context:
        instructions.append("## Custom Context")
        instructions.append(custom_context)
        instructions.append("")
    
    instructions.append("## Tips")
    instructions.append("- Be specific in your questions")
    instructions.append("- Complex questions work well with this agent")
    instructions.append("- The agent may take multiple steps to answer your question")
    
    return "\n".join(instructions)


def _generate_generic_instructions(
    agent_type: str,
    tools: Optional[List[Any]] = None,
    custom_context: Optional[str] = None
) -> str:
    """Generate generic instructions for any agent type."""
    instructions = []
    
    instructions.append(f"# Chat Instructions for {agent_type.title()} Agent\n")
    instructions.append("## Overview")
    instructions.append(f"This is a {agent_type} agent in the zigzag framework.\n")
    
    instructions.append("## How to Interact")
    instructions.append("- Ask questions or provide tasks in natural language")
    instructions.append("- The agent will process your input and provide responses\n")
    
    if tools:
        instructions.append("## Available Tools")
        for tool in tools:
            tool_name = getattr(tool, 'name', str(tool))
            tool_desc = getattr(tool, 'description', 'No description available')
            if not tool_desc:  # Handle empty string case
                tool_desc = 'No description available'
            instructions.append(f"- **{tool_name}**: {tool_desc}")
        instructions.append("")
    
    if custom_context:
        instructions.append("## Custom Context")
        instructions.append(custom_context)
        instructions.append("")
    
    return "\n".join(instructions)


def format_conversation_instructions(role: str = "user") -> str:
    """
    Generate instructions for formatting conversation messages.
    
    Args:
        role: The role in the conversation (user, assistant, system)
        
    Returns:
        Instructions for formatting messages
    """
    instructions = []
    
    instructions.append("# Conversation Message Format\n")
    instructions.append("## Message Structure")
    instructions.append("Each message should have:")
    instructions.append("- **role**: The sender's role (user, assistant, or system)")
    instructions.append("- **content**: The message content\n")
    
    instructions.append("## Example")
    instructions.append("```json")
    instructions.append("{")
    instructions.append(f'  "role": "{role}",')
    instructions.append('  "content": "Your message here"')
    instructions.append("}")
    instructions.append("```")
    
    return "\n".join(instructions)


def print_instructions(agent_type: str = "react", tools: Optional[List[Any]] = None):
    """
    Print chat instructions to console.
    
    Args:
        agent_type: The type of agent
        tools: List of tools available to the agent
    """
    instructions = generate_chat_instructions(agent_type, tools)
    print(instructions)
