# zigzag
Composable agents

## Features

- **ReAct Agent**: A reasoning and acting agent that uses tools to solve complex problems
- **Chat Instructions Generator**: Generate formatted instructions for interacting with agents

## Usage

### Generate Chat Instructions

Generate formatted instructions for interacting with agents:

```bash
# Generate instructions for the default ReAct agent
zigzag instructions

# Generate instructions for a specific agent type
zigzag instructions react
zigzag instructions custom
```

### Using the Instructions Module Programmatically

```python
from zigzag.instructions import generate_chat_instructions, print_instructions

# Print instructions to console
print_instructions("react")

# Get instructions as a string
instructions = generate_chat_instructions("react")
```

### Using ReAct Agent

See `examples/generate_instructions.py` for a complete example of generating chat instructions.

## Installation

```bash
pip install -e .
```

## Requirements

- Python >= 3.12
- OpenAI API key
- Tavily API key (for web search)
