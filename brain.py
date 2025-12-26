#!/usr/bin/env python3
"""
🧠 Engram CLI - Memory-Enhanced Chat

A CLI interface for having conversations with Engram memory.
The AI automatically remembers important things from your conversations.

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

# Load environment variables from .env
def load_env():
    if os.environ.get("GEMINI_API_KEY"):
        return
    
    # Load from local .env first
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

from memory_proxy import PassiveMemoryProxy, ProxyConfig


def print_banner():
    """Print welcome banner"""
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║             🧠 ENGRAM - Memory-Enhanced Chat              ║")
    print("║                                                          ║")
    print("║  Your AI assistant that actually remembers things.       ║")
    print("║  Everything important is automatically saved.            ║")
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
    
    return f"{importance}{prefix}{mem['content'][:80]}...{tags_str}" if len(mem['content']) > 80 else f"{importance}{prefix}{mem['content']}{tags_str}"


def interactive_chat(proxy: PassiveMemoryProxy):
    """Run interactive chat session"""
    print_banner()
    
    memory_count = len(proxy.memory.memory_system.memories)
    print(f"📚 Loaded {memory_count} memories from previous sessions")
    print("💡 Type /help for commands, or just start chatting!\n")
    
    while True:
        try:
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
                    proxy.shutdown()
                    break
                
                elif cmd == 'help':
                    print_help()
                
                elif cmd == 'memories':
                    query = arg if arg else "recent important"
                    memories = proxy.search_memories(query, limit=5)
                    print(f"\n📚 Found {len(memories)} memories for '{query}':")
                    for i, mem in enumerate(memories, 1):
                        print(f"   {format_memory(mem, i)}")
                    print()
                
                elif cmd == 'recent':
                    memories = proxy.memory.memory_system.get_recent_memories(hours=24, limit=5)
                    print(f"\n🕐 Recent memories (last 24h):")
                    for i, mem in enumerate(memories, 1):
                        print(f"   {i}. {mem.content[:70]}...")
                    print()
                
                elif cmd == 'stats':
                    stats = proxy.get_stats()
                    print(f"""
📊 Session Statistics:
   Messages sent: {stats['messages_processed']}
   Memories injected: {stats['memories_injected']}
   Total memories: {stats['memory_count']}
   Memories extracted this session: {stats['extraction_stats']['memories_extracted']}
   Session duration: {stats['session_duration_seconds']:.0f}s
""")
                
                elif cmd == 'add':
                    if arg:
                        memory_id = proxy.add_memory(arg, importance=0.7)
                        print(f"✅ Memory added: {memory_id[:8]}...")
                    else:
                        print("Usage: /add <memory content>")
                
                elif cmd == 'clear':
                    proxy.clear_conversation()
                
                else:
                    print(f"❓ Unknown command: /{cmd}. Type /help for available commands.")
                
                continue
            
            # Regular chat message
            print("\033[1;33mAssistant:\033[0m ", end="", flush=True)
            
            # Get response (with automatic memory injection)
            response = proxy.chat(user_input)
            print(response)
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Your memories are saved.")
            proxy.shutdown()
            break
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            continue


def search_memories(proxy: PassiveMemoryProxy, query: str):
    """Search and display memories"""
    memories = proxy.search_memories(query, limit=10)
    
    if not memories:
        print(f"No memories found for '{query}'")
        return
    
    print(f"\n📚 Found {len(memories)} memories for '{query}':\n")
    for i, mem in enumerate(memories, 1):
        print(format_memory(mem, i))
    print()


def show_stats(proxy: PassiveMemoryProxy):
    """Show memory system statistics"""
    stats = proxy.get_stats()
    memory_stats = proxy.memory.get_system_status()
    
    print(f"""
🧠 Memory System Statistics
{'=' * 40}

Storage:
   Total memories: {stats['memory_count']}
   Oldest memory: {memory_stats['memory_system'].get('oldest_memory', 'N/A')}
   Newest memory: {memory_stats['memory_system'].get('newest_memory', 'N/A')}

Extraction:
   Exchanges processed: {stats['extraction_stats']['exchanges_processed']}
   Memories extracted: {stats['extraction_stats']['memories_extracted']}
   Errors: {stats['extraction_stats']['extraction_errors']}

Session:
   Messages: {stats['messages_processed']}
   Memories injected: {stats['memories_injected']}
   Tokens used for memory: ~{stats['tokens_used_for_memory']}
""")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="🧠 Engram - Memory-Enhanced Chat",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python brain.py                     Start interactive chat
  python brain.py --search "python"   Search memories about python
  python brain.py --stats             Show memory statistics
  python brain.py --add "Remember to always use dark mode"
        """
    )
    
    parser.add_argument("--search", "-s", type=str, help="Search memories")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--add", "-a", type=str, help="Add a memory")
    parser.add_argument("--importance", "-i", type=float, default=0.7,
                       help="Importance for added memory (0.0-1.0)")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Verbose output")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash",
                       help="Model to use for chat")
    parser.add_argument("--no-extraction", action="store_true",
                       help="Disable automatic memory extraction")
    
    args = parser.parse_args()
    
    # Create config
    config = ProxyConfig(
        verbose=args.verbose,
        model=args.model,
        extraction_enabled=not args.no_extraction
    )
    
    # Initialize proxy
    try:
        proxy = PassiveMemoryProxy(config=config)
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        sys.exit(1)
    
    # Handle commands
    if args.search:
        search_memories(proxy, args.search)
    
    elif args.stats:
        show_stats(proxy)
    
    elif args.add:
        memory_id = proxy.add_memory(args.add, importance=args.importance)
        print(f"✅ Memory added: {memory_id}")
    
    else:
        # Interactive chat
        interactive_chat(proxy)


if __name__ == "__main__":
    main()

