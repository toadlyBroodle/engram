#!/usr/bin/env python3
"""
Basic Usage Example for Engram

This example demonstrates how to use Engram as a dependency in your own projects.
"""

from engram_pkg import MemoryIntegration


def main():
    """Demonstrate basic memory operations"""

    # Initialize the memory system
    print("🧠 Initializing Engram...")
    memory = MemoryIntegration()

    # Add some memories
    print("\n📚 Adding memories...")
    memories = [
        ("Python list comprehensions are more memory efficient than traditional loops", 0.8),
        ("Always validate user input to prevent security vulnerabilities", 0.9),
        ("Use type hints for better code maintainability", 0.7),
        ("Neural networks require proper data normalization", 0.8),
    ]

    for content, importance in memories:
        memory_id = memory.add_memory(content, importance=importance)
        print(f"  ✓ Added: {content[:50]}... (ID: {memory_id[:8]})")

    # Query for relevant memories
    print("\n🔍 Querying memories...")
    query = "How can I improve my Python code performance?"
    print(f"Query: {query}")

    # Add query to conversation context
    memory.update_conversation("user", query)

    # Get relevant memories
    results = memory.get_context_memories(max_memories=3)

    if results["memory_count"] > 0:
        print(f"\n🧠 Found {results['memory_count']} relevant memories:")
        print(results["formatted_context"])
    else:
        print("No relevant memories found.")

    # Show system stats
    print("\n📊 System Statistics:")
    stats = memory.get_system_status()
    print(f"  Total memories: {stats['memory_system']['total_memories']}")

    print("\n✅ Memory operations completed successfully!")


if __name__ == "__main__":
    main()
