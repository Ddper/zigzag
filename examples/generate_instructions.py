#!/usr/bin/env python3
"""
Example script demonstrating how to generate chat instructions.
"""

from zigzag.instructions import generate_chat_instructions, print_instructions


def main():
    print("=" * 70)
    print("Example 1: Generate instructions for ReAct agent")
    print("=" * 70)
    print_instructions("react")
    
    print("\n" + "=" * 70)
    print("Example 2: Generate instructions for custom agent")
    print("=" * 70)
    print_instructions("custom")
    
    print("\n" + "=" * 70)
    print("Example 3: Generate instructions programmatically")
    print("=" * 70)
    instructions = generate_chat_instructions("react")
    print(f"Generated {len(instructions)} characters of instructions")
    print("\nFirst 200 characters:")
    print(instructions[:200])


if __name__ == "__main__":
    main()
