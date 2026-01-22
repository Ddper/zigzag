import sys
import argparse
from zigzag.instructions import generate_chat_instructions, print_instructions


def main() -> None:
    """Main entry point for zigzag CLI."""
    parser = argparse.ArgumentParser(
        prog='zigzag',
        description='Composable agents framework'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Instructions subcommand
    instructions_parser = subparsers.add_parser(
        'instructions',
        help='Generate chat instructions for an agent'
    )
    instructions_parser.add_argument(
        'agent_type',
        nargs='?',
        default='react',
        help='Agent type (default: react)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command == 'instructions':
        print_instructions(agent_type=args.agent_type)
    else:
        # Show help if no command provided
        parser.print_help()
