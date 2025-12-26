#!/usr/bin/env python3
"""
Quick Memory Query Interface

Fast access to relevant memories during conversations.
Use this to get instant memory suggestions without starting the full assistant.
"""

import sys
import argparse
from memory_integration import MemoryIntegration

# Global instance for performance - lazy loaded
_memory_integration = None


def get_memory_integration():
    """Get the shared MemoryIntegration instance (lazy loaded)"""
    global _memory_integration
    if _memory_integration is None:
        try:
            _memory_integration = MemoryIntegration()
        except ImportError as e:
            raise RuntimeError(f"Missing required dependencies: {e}\n💡 Try: pip install -r requirements.txt")
        except FileNotFoundError as e:
            raise RuntimeError(f"Memory system files not found: {e}\n💡 Try: python auto_bootstrap.py")
        except PermissionError as e:
            raise RuntimeError(f"Permission denied accessing memory files: {e}\n💡 Check file permissions in the project directory")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize memory system: {e}\n💡 Try: python quick_memory.py tool")
    return _memory_integration


def get_memory_suggestion(query: str, max_memories: int = 3) -> str:
    """
    Get quick memory suggestions for a query

    Args:
        query: The topic or question to get memories for
        max_memories: Maximum number of memories to return

    Returns:
        Formatted memory suggestions
    """
    if not query or not query.strip():
        return "❌ Query cannot be empty. Please provide a topic or question to search for."

    if max_memories < 1 or max_memories > 10:
        return "❌ max_memories must be between 1 and 10."

    try:
        integration = get_memory_integration()

        # Add the query as user context
        integration.update_conversation("user", query)

        # Get relevant memories
        result = integration.get_context_memories(max_memories=max_memories)

        if result["memory_count"] == 0:
            return f"💭 No relevant memories found for '{query}'. Consider adding some knowledge about this topic."

        response = f"🧠 **Relevant Memories** ({result['memory_count']} found, {result['tokens_used']} tokens):\n\n"
        response += result["formatted_context"]
        return response

    except ConnectionError as e:
        return f"❌ Network error accessing memory system: {e}\n💡 Check your internet connection"
    except TimeoutError as e:
        return f"❌ Timeout error: {e}\n💡 Try again in a moment"
    except ValueError as e:
        return f"❌ Invalid query format: {e}\n💡 Try rephrasing your question"
    except Exception as e:
        return f"❌ Unexpected error accessing memory system: {e}\n💡 Try: python quick_memory.py tool"


def add_memory_quick(content: str, tags: str = "", importance: float = 0.5):
    """
    Quickly add a memory to the system

    Args:
        content: Memory content
        tags: Comma-separated tags
        importance: Importance score (0.0-1.0)
    """
    if not content or not content.strip():
        return "❌ Memory content cannot be empty. Please provide meaningful content to remember."

    if not (0.0 <= importance <= 1.0):
        return "❌ Importance must be between 0.0 and 1.0."

    try:
        integration = get_memory_integration()
        tag_list = [tag.strip() for tag in tags.split(",")] if tags else []

        memory_id = integration.add_memory(content, importance=importance, tags=tag_list)
        tag_info = f" with tags: {', '.join(tag_list)}" if tag_list else ""
        return f"✅ Memory added (ID: {memory_id[:8]}){tag_info}"

    except ValueError as e:
        return f"❌ Invalid memory format: {e}\n💡 Check your content and importance values"
    except PermissionError as e:
        return f"❌ Permission denied saving memory: {e}\n💡 Check file permissions"
    except Exception as e:
        return f"❌ Error adding memory: {e}\n💡 Try: python quick_memory.py tool"


def delete_memory_quick(memory_id: str):
    """
    Quickly delete a memory from the system

    Args:
        memory_id: ID of the memory to delete
    """
    if not memory_id or not memory_id.strip():
        return "❌ Memory ID cannot be empty. Please provide a valid memory ID."

    try:
        integration = get_memory_integration()
        deleted = integration.delete_memory(memory_id)
        if deleted:
            return f"🗑️  Memory {memory_id} successfully deleted and forgotten"
        else:
            return f"❌ Memory {memory_id} not found\n💡 Use 'python quick_memory.py stats' to see available memories"
    except ValueError as e:
        return f"❌ Invalid memory ID format: {e}\n💡 Memory IDs are typically 8-character strings"
    except PermissionError as e:
        return f"❌ Permission denied deleting memory: {e}\n💡 Check file permissions"
    except Exception as e:
        return f"❌ Error deleting memory: {e}\n💡 Try: python quick_memory.py tool"


def wipe_all_memories():
    """
    Delete all memories and start fresh

    Returns:
        Status message
    """
    try:
        integration = get_memory_integration()
        count = integration.wipe_all_memories()
        if count > 0:
            return f"🧹 Wiped {count} memories. Memory system is now empty and ready for fresh start."
        else:
            return "ℹ️  No memories to wipe. Memory system was already empty."
    except PermissionError as e:
        return f"❌ Permission denied wiping memories: {e}\n💡 Check file permissions"
    except Exception as e:
        return f"❌ Error wiping memories: {e}\n💡 Try: python quick_memory.py tool"


def show_memory_stats():
    """Show current memory system statistics"""
    try:
        integration = get_memory_integration()
        stats = integration.get_system_status()

        return f"""📊 **Memory System Stats:**
- Total memories: {stats['memory_system']['total_memories']}
- Integration rate: {stats['integration_layer']['integration_rate']:.1%}
- Context window: {stats['context_manager']['usage_percentage']:.1f}% utilized
- Memory performance: {stats.get('integration_layer', {}).get('performance', 'N/A')}"""

    except Exception as e:
        return f"❌ Error getting stats: {e}"


def check_autonomous_milestone(context: str = "manual_check") -> str:
    """Check if current system state warrants an autonomous milestone commit"""
    try:
        integration = get_memory_integration()

        if hasattr(integration.bootstrap_handler, 'check_and_commit_milestone'):
            success = integration.bootstrap_handler.check_and_commit_milestone(context)

            if success:
                return "🎯 AUTONOMOUS MILESTONE: Significant improvements detected and committed!"
            else:
                return "ℹ️  No milestone-worthy improvements detected at this time."
        else:
            return "❌ Autonomous milestone system not available"

    except Exception as e:
        return f"❌ Error checking milestone: {e}"


def get_conversation_bootstrap() -> str:
    """Get bootstrap knowledge for Cursor IDE conversations"""
    try:
        from bootstrap_memory import bootstrap_conversation_awareness
        return bootstrap_conversation_awareness()
    except Exception as e:
        return f"❌ Error getting bootstrap knowledge: {e}"


def bootstrap_tool() -> str:
    """CRITICAL BOOTSTRAP TOOL - Must be called first in every new conversation"""
    try:
        from bootstrap_memory import execute_bootstrap_tool
        return execute_bootstrap_tool()
    except Exception as e:
        return f"❌ CRITICAL ERROR: Bootstrap tool failed: {e}\n\nPlease manually run: python quick_memory.py bootstrap"


def get_memory_reminder() -> str:
    """Get a memory-saving reminder"""
    try:
        integration = get_memory_integration()

        if hasattr(integration.bootstrap_handler, 'get_memory_saving_reminder'):
            return integration.bootstrap_handler.get_memory_saving_reminder()
        else:
            return "🧠 **MEMORY REMINDER**: Save important insights!\nUse: python quick_memory.py add \"insight\" \"tags\" 0.8"

    except Exception as e:
        return f"❌ Error getting reminder: {e}"


def create_parser():
    """Create argument parser for the CLI interface"""
    parser = argparse.ArgumentParser(
        description="🧠 Quick Memory Query Interface - Fast access to relevant memories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🧠 **CRITICAL**: Start EVERY new conversation with:
  python quick_memory.py tool

Examples:
  python quick_memory.py query "How does the memory system work?"
  python quick_memory.py add "New insight" "tags,here" 0.8
  python quick_memory.py delete "memory_id"
  python quick_memory.py wipe
  python quick_memory.py stats
  python quick_memory.py milestone "manual_check"
  python quick_memory.py remind
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
    add_parser.add_argument('tags', nargs='?', default='', help='Comma-separated tags (e.g., "tag1,tag2")')
    add_parser.add_argument('importance', nargs='?', type=float, default=0.5, help='Importance score 0.0-1.0 (default: 0.5)')

    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete a memory')
    delete_parser.add_argument('memory_id', help='ID of the memory to delete')

    # Wipe command
    subparsers.add_parser('wipe', help='Delete all memories and start fresh')

    # Stats command
    subparsers.add_parser('stats', help='Show memory system statistics')

    # Milestone command
    milestone_parser = subparsers.add_parser('milestone', help='Check for autonomous milestone commit')
    milestone_parser.add_argument('--context', default='manual_check', help='Context for milestone check')

    # Bootstrap command
    subparsers.add_parser('bootstrap', help='Get conversation bootstrap knowledge')

    # Tool command
    subparsers.add_parser('tool', help='CRITICAL BOOTSTRAP TOOL - Run at start of every conversation')

    # Remind command
    subparsers.add_parser('remind', help='Get memory-saving reminder')

    return parser


def main():
    """Main entry point with improved argument parsing and error handling"""
    parser = create_parser()

    try:
        args = parser.parse_args()
    except SystemExit:
        # argparse handles --help and invalid arguments, but let's add a helpful message
        if len(sys.argv) == 1:
            print("🧠 **CRITICAL**: Start EVERY new conversation with: python quick_memory.py tool")
        return

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == 'query':
            result = get_memory_suggestion(args.query, args.max_memories)
            print(result)

        elif args.command == 'add':
            result = add_memory_quick(args.content, args.tags, args.importance)
            print(result)

        elif args.command == 'delete':
            result = delete_memory_quick(args.memory_id)
            print(result)

        elif args.command == 'wipe':
            result = wipe_all_memories()
            print(result)

        elif args.command == 'stats':
            result = show_memory_stats()
            print(result)

        elif args.command == 'milestone':
            result = check_autonomous_milestone(args.context)
            print(result)

        elif args.command == 'bootstrap':
            result = get_conversation_bootstrap()
            print(result)

        elif args.command == 'tool':
            result = bootstrap_tool()
            print(result)

        elif args.command == 'remind':
            result = get_memory_reminder()
            print(result)

    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled by user")
        sys.exit(1)
    except RuntimeError as e:
        # Handle initialization errors specifically
        print(f"❌ Memory system initialization failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("💡 Tip: Run 'python quick_memory.py tool' to bootstrap the memory system")
        print("🔍 For more help: python quick_memory.py --help")
        sys.exit(1)


if __name__ == "__main__":
    main()
