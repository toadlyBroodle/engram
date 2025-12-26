#!/usr/bin/env python3
"""
Example: Using Engram as External Dependency

This shows how to use Engram in another project without installing it.
Just set PYTHONPATH to include the engram project directory.
"""

# Add the memory project to Python path (if not already done via environment)
import sys
from pathlib import Path

# Add the engram project path (adjust this path for your setup)
engram_path = Path(__file__).parent.parent
if str(engram_path) not in sys.path:
    sys.path.insert(0, str(engram_path))

# Now import normally
from engram_pkg import MemoryIntegration, VectorMemory

def demonstrate_usage():
    """Show how to use Engram in external projects"""

    print("🧠 Using Engram as External Dependency")
    print("=" * 60)

    # Initialize memory system
    print("📚 Initializing memory system...")
    memory = MemoryIntegration()

    # Add some example memories
    examples = [
        "Machine learning models improve with more data",
        "Python's asyncio enables concurrent programming",
        "REST APIs should use proper HTTP status codes",
        "Database indexes improve query performance"
    ]

    print("💾 Adding example memories...")
    for example in examples:
        memory_id = memory.add_memory(example, importance=0.7)
        print(f"  ✓ {example[:40]}... (ID: {memory_id[:8]})")

    # Query for relevant memories
    print("\n🔍 Searching for 'performance' related memories...")
    memory.update_conversation("user", "How can I improve application performance?")

    results = memory.get_context_memories(max_memories=2)

    if results["memory_count"] > 0:
        print(f"📋 Found {results['memory_count']} relevant memories:")
        print(results["formatted_context"])
    else:
        print("No relevant memories found.")

    # Show stats
    stats = memory.get_system_status()
    print("\n📊 Memory System Stats:")
    print(f"  Memories: {stats['memory_system']['total_memories']}")
    print("\n✅ External usage demonstration complete!")

if __name__ == "__main__":
    demonstrate_usage()
