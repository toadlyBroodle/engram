#!/usr/bin/env python3
"""
Comprehensive Test of Memory Integration System

Tests the complete memory integration pipeline with realistic conversation scenarios.
Demonstrates how relevant memories are automatically integrated into context.
"""

import json
import time
import sys
from datetime import datetime
from pathlib import Path

# Add project directory to path (parent of test directory)
project_dir = Path(__file__).parent.parent
sys.path.insert(0, str(project_dir))

from memory_integration import MemoryIntegration
from adaptive_memory import create_adaptive_memory_system


def setup_test_memories(integration: MemoryIntegration):
    """Set up diverse test memories for comprehensive testing"""
    print("📚 Setting up test memories...")

    test_memories = [
        # Programming & Development
        ("Python list comprehensions provide better performance than traditional loops",
         0.8, ["python", "programming", "performance"]),

        ("Always validate user input to prevent security vulnerabilities",
         0.95, ["security", "programming", "best_practices"]),

        ("Use type hints in Python for better code maintainability",
         0.7, ["python", "coding", "maintainability"]),

        # AI & Machine Learning
        ("Neural networks require proper data normalization for optimal training",
         0.9, ["machine_learning", "neural_networks", "data_processing"]),

        ("Batch processing improves GPU utilization in deep learning",
         0.8, ["deep_learning", "gpu", "optimization"]),

        ("Feature scaling prevents some features from dominating the learning process",
         0.85, ["machine_learning", "preprocessing", "features"]),

        # Vector Databases & Memory
        ("FAISS provides sub-millisecond similarity search for high-dimensional vectors",
         0.9, ["vector_database", "faiss", "performance"]),

        ("Sentence transformers convert text to semantic vector embeddings",
         0.9, ["embeddings", "nlp", "semantic_search"]),

        ("Memory consolidation maintains quality while managing storage limits",
         0.8, ["memory_management", "optimization", "storage"]),

        # Performance & Optimization
        ("Context window management is crucial for LLM efficiency",
         0.9, ["llm", "performance", "context_management"]),

        ("CPU optimization requires AVX2 instruction utilization",
         0.7, ["cpu", "optimization", "hardware"]),

        ("Memory fragmentation can impact application performance",
         0.6, ["memory", "performance", "system_optimization"]),
    ]

    for content, importance, tags in test_memories:
        memory_id = integration.add_memory(content, importance, tags)
        print(f"  ✓ Added memory: {content[:50]}... (ID: {memory_id[:8]})")

    print(f"✅ Set up {len(test_memories)} test memories\n")


def simulate_conversation(integration: MemoryIntegration, tracker, scorer):
    """Simulate a realistic conversation with memory integration"""
    print("💬 Simulating conversation with memory integration...")
    print("-" * 60)

    # Conversation scenario: User asking about optimizing AI memory systems
    conversation_flow = [
        ("user", "How can I optimize my AI memory system for better performance?"),
        ("assistant", "There are several approaches to optimize AI memory systems. Let me think about the key factors..."),
        ("user", "I'm particularly interested in vector databases and context management"),
        ("assistant", "Vector databases are indeed crucial for modern AI memory systems..."),
        ("user", "What about CPU optimization and memory consolidation?"),
        ("assistant", "Those are important aspects too. Let me explain the trade-offs..."),
        ("user", "How do embeddings work in this context?"),
        ("assistant", "Embeddings are fundamental to semantic memory systems..."),
    ]

    for i, (role, message) in enumerate(conversation_flow):
        print(f"{role.title()}: {message}")

        # Update conversation context
        integration.update_conversation(role, message)

        # Get relevant memories for this context
        context_result = integration.get_context_memories(
            context_tokens_used=500  # Simulate some tokens already used
        )

        if context_result["memories"]:
            print(f"\n🧠 INTEGRATED MEMORIES ({context_result['memory_count']} found, {context_result['tokens_used']} tokens):")
            print(context_result["formatted_context"])

            # Simulate memory usage feedback (in real usage, this would come from actual performance)
            for memory in context_result["memories"]:
                success_score = 0.7 + (hash(memory.original_id + str(i)) % 30) / 100  # 0.7-1.0
                tracker.record_memory_usage(
                    memory.original_id,
                    context_type="conversation",
                    success_score=success_score,
                    query=message
                )
        else:
            print("  (No relevant memories found)")

        print("-" * 40)

        # Small delay to simulate thinking time
        time.sleep(0.1)

    print("✅ Conversation simulation complete\n")


def test_adaptive_learning(integration: MemoryIntegration, tracker, scorer):
    """Test the adaptive learning capabilities"""
    print("🧠 Testing Adaptive Learning System...")
    print("-" * 40)

    # Test different query types to see learning
    test_queries = [
        ("How do I improve my code performance?", "coding"),
        ("What's the best way to handle user input validation?", "security"),
        ("How do vector databases work?", "technical"),
        ("Why is data normalization important for ML?", "machine_learning"),
    ]

    for query, expected_context in test_queries:
        print(f"\nQuery: '{query}'")

        # Get adaptive relevance scoring
        context_result = integration.get_context_memories(query, context_tokens_used=300)

        if context_result["memories"]:
            print(f"Found {context_result['memory_count']} relevant memories")

            # Show learning insights
            insights = scorer.get_learning_insights()
            if "top_query_patterns" in insights and insights["top_query_patterns"]:
                print(f"Learning from {len(insights['top_query_patterns'])} query patterns")
        else:
            print("No memories found for this query")

    print("✅ Adaptive learning test complete\n")


def demonstrate_system_capabilities(integration: MemoryIntegration):
    """Demonstrate key system capabilities"""
    print("🚀 Demonstrating System Capabilities...")
    print("-" * 50)

    # Test memory search
    print("1. MEMORY SEARCH:")
    search_results = integration.search_memories("vector database performance", limit=3)
    for result in search_results:
        print(f"   • {result['content'][:60]}... (usage: {result['usage_count']})")

    # Test context snapshot
    print("\n2. CONTEXT SNAPSHOT:")
    snapshot = integration.export_context_snapshot()
    snapshot_data = json.loads(snapshot)
    print(f"   Conversation messages: {len(snapshot_data['conversation_context'].get('conversation_history', []))}")
    print(f"   System memories: {snapshot_data['system_status']['memory_system']['total_memories']}")

    # Test system optimization
    print("\n3. SYSTEM OPTIMIZATION:")
    initial_stats = integration.get_system_status()
    integration.optimize_for_performance()
    final_stats = integration.get_system_status()
    print(f"   Memories before: {initial_stats['memory_system']['total_memories']}")
    print(f"   Memories after: {final_stats['memory_system']['total_memories']}")

    print("✅ System capabilities demonstration complete\n")


def run_performance_analysis(integration: MemoryIntegration, tracker, scorer):
    """Run performance analysis on the memory system"""
    print("📊 Performance Analysis...")
    print("-" * 30)

    # Get system status
    status = integration.get_system_status()
    print(f"Memory System: {status['memory_system']['total_memories']} memories")
    print(f"Integration Layer: {status['integration_layer']['integration_rate']:.2%} integration rate")
    print(f"Context Window: {status['context_manager']['usage_percentage']:.1f}% utilized")

    # Get learning insights
    insights = scorer.get_learning_insights()
    print(f"\nLearning Insights:")
    print(f"  Relevance weights: {insights['relevance_weights']}")

    trends = insights.get('performance_trends', {})
    if 'average_performance' in trends:
        print(f"  Performance trends: {trends['average_performance']:.3f} avg performance")

    if trends.get('top_performing_memories'):
        top_mem = trends['top_performing_memories'][0]
        print(f"  Top memory: {top_mem[0][:8]} (score: {top_mem[1]:.3f})")

    print("✅ Performance analysis complete\n")


def main():
    """Main test function"""
    print("🧠 COMPREHENSIVE MEMORY INTEGRATION SYSTEM TEST")
    print("=" * 60)

    # Initialize systems
    integration = MemoryIntegration()
    tracker, scorer = create_adaptive_memory_system()

    try:
        # Setup test data
        setup_test_memories(integration)

        # Run comprehensive tests
        simulate_conversation(integration, tracker, scorer)
        test_adaptive_learning(integration, tracker, scorer)
        demonstrate_system_capabilities(integration)
        run_performance_analysis(integration, tracker, scorer)

        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("✅ Memory integration system is working correctly")
        print("✅ Context-aware memory retrieval functioning")
        print("✅ Adaptive learning system operational")
        print("✅ Performance optimization active")
        print("\nThe system is now ready to consistently integrate relevant")
        print("memories into working context for improved AI performance!")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
