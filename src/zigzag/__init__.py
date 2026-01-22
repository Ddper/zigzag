import sys
from zigzag.instructions import generate_chat_instructions, print_instructions


def main() -> None:
    """Main entry point for zigzag CLI."""
    if len(sys.argv) > 1 and sys.argv[1] == "instructions":
        # Generate and print chat instructions
        agent_type = sys.argv[2] if len(sys.argv) > 2 else "react"
        print_instructions(agent_type=agent_type)
    else:
        print("Hello from zigzag!")
        print("\nUsage:")
        print("  zigzag instructions [agent_type]  - Generate chat instructions")
        print("\nExample:")
        print("  zigzag instructions react")
