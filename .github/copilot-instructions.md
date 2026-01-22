# GitHub Copilot Instructions for Zigzag

## Project Overview

Zigzag is a composable agents framework built in Python. The project implements AI agents that can use tools to solve complex problems, starting with a ReAct (Reasoning and Acting) agent pattern.

## Project Structure

```
zigzag/
├── src/zigzag/
│   ├── __init__.py          # Main entry point
│   ├── settings.py          # Configuration using pydantic-settings
│   ├── agents/              # Agent implementations
│   │   ├── react.py         # ReAct agent implementation
│   │   └── __init__.py
│   └── prompts/             # Prompt templates
│       └── __init__.py      # System prompts for agents
├── pyproject.toml           # Project configuration and dependencies
└── README.md
```

## Technology Stack

- **Python 3.12+**: Minimum required version
- **OpenAI API**: For LLM interactions (supports custom base URLs)
- **Pydantic**: For data validation and settings management
- **Tavily**: For web search functionality
- **python-dotenv**: For environment variable management

## Key Components

### 1. ReAct Agent (`src/zigzag/agents/react.py`)

The ReAct agent follows the Reasoning and Acting paradigm:
- Uses a thought-action-observation loop
- Can use tools to gather information
- Maximum 5 iterations by default
- Returns JSON responses with `Thought`, `Action`, `Action Input`, and `Answer` fields

**Key Classes:**
- `Tool`: Wrapper for agent tools with name, description, and callable function
- `Message`: Pydantic model for chat messages with role and content
- `ReActAgent`: Main agent class that orchestrates the reasoning loop

### 2. Settings (`src/zigzag/settings.py`)

Configuration is managed via pydantic-settings:
- Loads from `.env` file in the zigzag source directory
- Required settings:
  - `TAVILY_API_KEY`: API key for Tavily search
  - `OPENAI_BASE_URL`: Base URL for OpenAI-compatible API
  - `OPENAI_API_KEY`: API key for OpenAI
  - `OPENAI_MODEL_NAME`: Model name (default: "deepseek-chat")

### 3. Prompts (`src/zigzag/prompts/__init__.py`)

Contains the `system_prompt` template that defines:
- Agent behavior and capabilities
- Tool usage format
- Output format requirements (JSON with Thought/Action/Answer)

## Code Style and Conventions

### Python Style
- Use type hints for function parameters and return values
- Follow PEP 8 naming conventions
- Use Pydantic models for data validation
- Use docstrings for classes and complex functions

### Agent Implementation
- Agents should be composable and extensible
- Tools should be independent and reusable
- Use the Tool class wrapper for all agent tools
- Maintain the thought-action-observation pattern

### Settings Management
- All configuration should use pydantic-settings
- Environment variables should be loaded from `.env` file
- Provide sensible defaults where appropriate

## Development Guidelines

### Adding New Agents
1. Create a new file in `src/zigzag/agents/`
2. Implement the agent class with appropriate methods
3. Use the existing Tool class for tool integration
4. Add corresponding prompts if needed

### Adding New Tools
1. Define a function that takes a string query and returns a string result
2. Wrap it in a Tool instance with descriptive name and description
3. Pass the Tool to the agent's tools list
4. Document the tool's purpose and expected input/output

### Error Handling
- Use Union types for operations that might return errors (e.g., `Observation = Union[str, Exception]`)
- Catch and handle exceptions appropriately in tool functions
- Provide meaningful error messages to the user

### Dependencies
- Keep dependencies minimal and well-justified
- Use `>=` for version constraints unless a specific version is required
- Document any new dependencies and their purpose

## Testing Considerations

When adding tests:
- Test agent behavior with mock LLM responses
- Test tool functionality independently
- Test settings loading from environment variables
- Test error handling and edge cases

## Environment Setup

For development:
```bash
# Install in editable mode
pip install -e .

# Create .env file in src/zigzag/
cp src/zigzag/.env.example src/zigzag/.env
# Edit .env with your API keys
```

## Common Patterns

### Creating an Agent
```python
from openai import OpenAI
from zigzag.agents.react import ReActAgent, Tool
from zigzag.settings import settings

client = OpenAI(
    base_url=settings.openai_base_url,
    api_key=settings.openai_api_key,
)

tools = [Tool("tool_name", "Tool description", tool_function)]
agent = ReActAgent(client, tools)
response = agent.run("Your question here")
```

### Creating a Tool
```python
def my_tool(query: str) -> str:
    """Tool implementation that returns a string result."""
    # Process query and return result
    return result

tool = Tool("my_tool", "Description of what the tool does", my_tool)
```

## Notes for Copilot

- The project uses OpenAI API but supports custom base URLs (e.g., DeepSeek)
- JSON format for LLM responses is enforced using `response_format={"type": "json_object"}`
- The agent maintains conversation history in `self.messages` list
- Tools are invoked synchronously within the agent loop
- The system is designed to be framework-agnostic for LLM providers

## Future Considerations

- Additional agent types beyond ReAct
- Async tool execution
- Tool composition and chaining
- Agent memory and context management
- Multi-agent collaboration patterns
