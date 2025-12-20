#!/usr/bin/env python3
"""
Quick Memory Query Interface

Fast access to relevant memories during conversations.
Use this to get instant memory suggestions without starting the full assistant.
"""

import sys
from memory_integration import MemoryIntegration


def get_memory_suggestion(query: str, max_memories: int = 3) -> str:
    """
    Get quick memory suggestions for a query

    Args:
        query: The topic or question to get memories for
        max_memories: Maximum number of memories to return

    Returns:
        Formatted memory suggestions
    """
    try:
        integration = MemoryIntegration()

        # Add the query as user context
        integration.update_conversation("user", query)

        # Get relevant memories
        result = integration.get_context_memories(max_memories=max_memories)

        if result["memory_count"] == 0:
            return "💭 No relevant memories found. Consider adding some knowledge about this topic."

        response = f"🧠 **Relevant Memories** ({result['memory_count']} found, {result['tokens_used']} tokens):\n\n"
        response += result["formatted_context"]
        return response

    except Exception as e:
        return f"❌ Error accessing memory system: {e}"


def add_memory_quick(content: str, tags: str = "", importance: float = 0.5):
    """
    Quickly add a memory to the system

    Args:
        content: Memory content
        tags: Comma-separated tags
        importance: Importance score (0.0-1.0)
    """
    try:
        integration = MemoryIntegration()
        tag_list = [tag.strip() for tag in tags.split(",")] if tags else []

        memory_id = integration.add_memory(content, importance=importance, tags=tag_list)
        return f"✅ Memory added (ID: {memory_id[:8]})"

    except Exception as e:
        return f"❌ Error adding memory: {e}"


def show_memory_stats():
    """Show current memory system statistics"""
    try:
        integration = MemoryIntegration()
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
        integration = MemoryIntegration()

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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python quick_memory.py query \"your question here\"")
        print("  python quick_memory.py add \"memory content\" \"tag1,tag2\" 0.8")
        print("  python quick_memory.py stats")
        print("  python quick_memory.py milestone [context]")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "query" and len(sys.argv) >= 3:
        query = " ".join(sys.argv[2:])
        result = get_memory_suggestion(query)
        print(result)

    elif command == "add" and len(sys.argv) >= 3:
        content = sys.argv[2]
        tags = sys.argv[3] if len(sys.argv) > 3 else ""
        importance = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
        result = add_memory_quick(content, tags, importance)
        print(result)

    elif command == "stats":
        result = show_memory_stats()
        print(result)

    elif command == "milestone":
        context = sys.argv[2] if len(sys.argv) > 2 else "manual_check"
        result = check_autonomous_milestone(context)
        print(result)

    else:
        print("❌ Invalid command. Use 'query', 'add', 'stats', or 'milestone'")
