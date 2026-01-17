#!/usr/bin/env python3
"""
🧠 Engram CLI - Memory-Enhanced Chat (RLM Edition)

A CLI interface for having conversations with Engram memory.
The AI actively queries its memory using tools, enabling iterative
retrieval and reasoning over stored knowledge.

Usage:
    python brain.py                    # Start interactive chat
    python brain.py --search "query"   # Search memories
    python brain.py --stats            # Show statistics
    python brain.py --add "memory"     # Manually add a memory
"""

import os
import sys
import argparse
from pathlib import Path

# Add project directory to path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))


def load_env():
    """Load environment variables from .env file"""
    if os.environ.get("GEMINI_API_KEY"):
        return
    
    env_path = project_dir / ".env"
    if env_path.exists():
        try:
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        if key not in os.environ:
                            os.environ[key] = value.strip('"\'')
            print(f"📂 Loaded environment from {env_path}")
        except Exception:
            pass


load_env()

from memory_agent import MemoryAgent, AgentConfig


def print_banner():
    """Print welcome banner"""
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║          🧠 ENGRAM - Memory-Enhanced Chat (RLM)          ║")
    print("║                                                          ║")
    print("║  Your AI assistant with active memory retrieval.         ║")
    print("║  It queries memory as needed, not just at prompt time.   ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()


def print_help():
    """Print in-chat help"""
    print("""
╭─────────────────────────────────────────────────────────────╮
│ Commands:                                                   │
│   /help          Show this help                            │
│   /memories      Search your memories                       │
│   /recent        Show recent memories                       │
│   /stats         Show session statistics                    │
│   /add <text>    Manually add a memory                     │
│   /clear         Clear conversation (keeps memories)        │
│   /quit          Exit the chat                             │
╰─────────────────────────────────────────────────────────────╯
""")


def format_memory(mem, index: int = None) -> str:
    """Format a memory for display"""
    prefix = f"{index}. " if index else "• "
    importance = "🔥" if mem.get("importance", 0) > 0.7 else "  "
    tags = ", ".join(mem.get("tags", [])[:3])
    tags_str = f" [{tags}]" if tags else ""
    content = mem.get('content', '')
    
    if len(content) > 80:
        return f"{importance}{prefix}{content[:80]}...{tags_str}"
    return f"{importance}{prefix}{content}{tags_str}"


def interactive_chat(agent: MemoryAgent):
    """Run interactive chat session"""
    print_banner()
    
    memory_count = len(agent.memory_system.memories)
    print(f"📚 Loaded {memory_count} memories from previous sessions")
    print("💡 Type /help for commands, or just start chatting!\n")
    
    while True:
        try:
            # Print any pending MemMan messages before prompt
            if hasattr(agent, 'extractor') and agent.extractor:
                agent.extractor.print_pending_messages()
            
            # Get user input
            user_input = input("\033[1;36mYou:\033[0m ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith('/'):
                cmd_parts = user_input[1:].split(' ', 1)
                cmd = cmd_parts[0].lower()
                arg = cmd_parts[1] if len(cmd_parts) > 1 else ""
                
                if cmd in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye! Your memories are saved.")
                    agent.shutdown()
                    break
                
                elif cmd == 'help':
                    print_help()
                
                elif cmd == 'memories':
                    query = arg if arg else "recent important"
                    memories = agent.search_memories(query, limit=5)
                    print(f"\n📚 Found {len(memories)} memories for '{query}':")
                    for i, mem in enumerate(memories, 1):
                        print(f"   {format_memory(mem, i)}")
                    print()
                
                elif cmd == 'recent':
                    memories = agent.memory_system.get_recent_memories(hours=24, limit=5)
                    print(f"\n🕐 Recent memories (last 24h):")
                    for i, mem in enumerate(memories, 1):
                        print(f"   {i}. {mem.content[:70]}...")
                    print()
                
                elif cmd == 'stats':
                    stats = agent.get_stats()
                    print(f"""
📊 Session Statistics:
   Messages sent: {stats['messages_processed']}
   Tool calls made: {stats['tool_calls_made']}
   Memories retrieved: {stats['memories_retrieved']}
   Memories stored: {stats['memories_stored']}
   Total memories: {stats['memory_count']}
   Memories extracted (async): {stats['extraction_stats']['memories_extracted']}
   Session duration: {stats['session_duration_seconds']:.0f}s
""")
                
                elif cmd == 'add':
                    if arg:
                        memory_id = agent.add_memory(arg, importance=0.7)
                        print(f"✅ Memory added: {memory_id[:8]}...")
                    else:
                        print("Usage: /add <memory content>")
                
                elif cmd == 'clear':
                    agent.clear_conversation()
                
                else:
                    print(f"❓ Unknown command: /{cmd}. Type /help for available commands.")
                
                continue
            
            # Regular chat message
            print("\033[1;33mAssistant:\033[0m ", end="", flush=True)
            
            # Get response (with active memory tool access)
            response = agent.chat(user_input)
            print(response)
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Your memories are saved.")
            agent.shutdown()
            break
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            continue


def search_memories(agent: MemoryAgent, query: str):
    """Search and display memories"""
    memories = agent.search_memories(query, limit=10)
    
    if not memories:
        print(f"No memories found for '{query}'")
        return
    
    print(f"\n📚 Found {len(memories)} memories for '{query}':\n")
    for i, mem in enumerate(memories, 1):
        print(format_memory(mem, i))
    print()


def show_stats(agent: MemoryAgent):
    """Show memory system statistics"""
    stats = agent.get_stats()
    memory_stats = agent.memory_system.get_memory_stats()
    
    print(f"""
🧠 Memory System Statistics
{'=' * 40}

Storage:
   Total memories: {stats['memory_count']}
   Oldest memory: {memory_stats.get('oldest_memory', 'N/A')}
   Newest memory: {memory_stats.get('newest_memory', 'N/A')}

Agent Activity:
   Tool calls made: {stats['tool_calls_made']}
   Memories retrieved: {stats['memories_retrieved']}
   Memories stored: {stats['memories_stored']}

Extraction:
   Exchanges processed: {stats['extraction_stats']['exchanges_processed']}
   Memories extracted: {stats['extraction_stats']['memories_extracted']}
   Errors: {stats['extraction_stats']['extraction_errors']}

Session:
   Messages: {stats['messages_processed']}
""")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="🧠 Engram - Memory-Enhanced Chat (RLM Edition)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python brain.py                     Start interactive chat
  python brain.py --search "python"   Search memories about python
  python brain.py --stats             Show memory statistics
  python brain.py --add "Remember..."  Add a memory
  python brain.py --remove abc123     Remove memory by ID
  python brain.py --merge             Merge similar memories
  python brain.py --wipe              Wipe all memories
        """
    )
    
    parser.add_argument("--search", "-s", type=str, help="Search memories")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--add", "-a", type=str, help="Add a memory")
    parser.add_argument("--remove", "-r", type=str, help="Remove a memory by ID")
    parser.add_argument("--merge", action="store_true", help="Merge similar memories")
    parser.add_argument("--wipe", action="store_true", help="Wipe all memories (requires confirmation)")
    parser.add_argument("--importance", "-i", type=float, default=0.7, help="Importance for added memory (0.0-1.0)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output (shows tool calls)")
    parser.add_argument("--brain-model", type=str, default="gemini-2.0-flash", help="Model for chat (default: gemini-2.0-flash)")
    parser.add_argument("--memman-model", type=str, default="gemini-2.0-flash-lite", help="Model for MemMan agent (default: gemini-2.0-flash-lite)")
    parser.add_argument("--no-extraction", action="store_true", help="Disable automatic memory extraction")
    parser.add_argument("--max-tool-calls", type=int, default=10, help="Max tool calls per turn (default: 10)")
    
    args = parser.parse_args()
    
    # Create config
    config = AgentConfig(
        verbose=args.verbose,
        model=args.brain_model,
        extraction_model=args.memman_model,
        extraction_enabled=not args.no_extraction,
        max_tool_calls=args.max_tool_calls
    )
    
    # Initialize agent
    try:
        agent = MemoryAgent(config=config)
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        sys.exit(1)
    
    # Handle commands
    if args.search:
        search_memories(agent, args.search)
    
    elif args.stats:
        show_stats(agent)
    
    elif args.add:
        memory_id = agent.add_memory(args.add, importance=args.importance)
        print(f"✅ Memory added: {memory_id}")
    
    elif args.remove:
        if agent.delete_memory(args.remove):
            print(f"✅ Memory removed: {args.remove}")
        else:
            print(f"❌ Memory not found: {args.remove}")
    
    elif args.merge:
        merged = agent.memory_system.merge_similar_memories(similarity_threshold=0.80)
        if merged:
            print(f"\n📦 Merged {sum(len(m[1]) for m in merged)} memories:")
            for kept_id, merged_ids in merged:
                kept_mem = agent.memory_system.memories.get(kept_id)
                content_preview = kept_mem.content[:50] if kept_mem else "?"
                print(f"   • {kept_id[:8]}: {content_preview}...")
                for mid in merged_ids:
                    print(f"     ← merged {mid[:8]}")
        else:
            print("✅ No similar memories to merge")
    
    elif args.wipe:
        count = len(agent.memory_system.memories)
        if count == 0:
            print("📭 No memories to wipe")
        else:
            confirm = input(f"⚠️  This will delete all {count} memories. Type 'yes' to confirm: ")
            if confirm.lower() == 'yes':
                # Wipe by rebuilding with empty list
                agent.memory_system._rebuild_index_with_memories([])
                print(f"🗑️  Wiped {count} memories")
            else:
                print("❌ Wipe cancelled")
    
    else:
        # Interactive chat
        interactive_chat(agent)


if __name__ == "__main__":
    main()
