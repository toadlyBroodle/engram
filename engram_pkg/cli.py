#!/usr/bin/env python3
"""
Command Line Interface for Engram

Provides a CLI for interacting with the Engram memory system.
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from .integration import MemoryIntegration


def create_parser():
    """Create argument parser for the CLI interface"""
    parser = argparse.ArgumentParser(
        description="🧠 Engram - AI Memory Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  engram query "How does the memory system work?"
  engram add "New insight" --importance 0.8
  engram stats
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Query command
    query_parser = subparsers.add_parser('query', help='Query relevant memories')
    query_parser.add_argument('query', help='The topic or question to search for')
    query_parser.add_argument('--max-memories', type=int, default=3, help='Maximum memories to return (default: 3)')

    # Add command
    add_parser = subparsers.add_parser('add', help='Add a new memory')
    add_parser.add_argument('content', help='Memory content to add')
    add_parser.add_argument('--importance', type=float, default=0.5, help='Importance score 0.0-1.0 (default: 0.5)')

    # Stats command
    subparsers.add_parser('stats', help='Show memory system statistics')

    return parser


def main():
    """Main entry point for the CLI"""
    parser = create_parser()

    try:
        args = parser.parse_args()
    except SystemExit:
        if len(sys.argv) == 1:
            parser.print_help()
        return

    if not args.command:
        parser.print_help()
        return

    try:
        # Initialize memory system
        memory = MemoryIntegration()

        if args.command == 'query':
            # Add the query as user context
            memory.update_conversation("user", args.query)

            # Get relevant memories
            result = memory.get_context_memories(max_memories=args.max_memories)

            if result["memory_count"] == 0:
                print(f"💭 No relevant memories found for '{args.query}'.")
                return

            print(f"🧠 **Relevant Memories** ({result['memory_count']} found):")
            print(result["formatted_context"])

        elif args.command == 'add':
            memory_id = memory.add_memory(args.content, importance=args.importance)
            print(f"✅ Memory added (ID: {memory_id[:8]})")

        elif args.command == 'stats':
            stats = memory.get_system_status()
            print("📊 **Engram Stats:**")
            print(f"- Total memories: {stats['memory_system']['total_memories']}")
            print(f"- Integration rate: {stats['integration_layer']['integration_rate']:.1%}")
            print(f"- Context window: {stats['context_manager']['usage_percentage']:.1f}% utilized")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Try running with --help for usage information")
        sys.exit(1)


if __name__ == "__main__":
    main()
