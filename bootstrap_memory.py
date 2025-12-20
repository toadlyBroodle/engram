#!/usr/bin/env python3
"""
Memory System Bootstrap

Initializes the AI with core memories about how to use the persistent memory system.
This ensures the AI always knows how to access and utilize its memory capabilities
even when starting from a fresh instance.

BOOT SEQUENCE:
1. Load core memory system knowledge
2. Initialize memory integration capabilities
3. Provide usage instructions and shortcuts
4. Enable real-time memory access during conversations
"""

import json
import os
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional


class MemoryBootstrap:
    """
    Bootstraps the AI with essential knowledge about the memory system.

    This creates a "boot memory" that teaches the AI how to use its own
    memory capabilities, ensuring persistent self-awareness.
    """

    def __init__(self, memory_path: str = "vector_memory", memory_system=None):
        self.memory_path = memory_path
        self.bootstrap_memories_file = Path(__file__).parent / "bootstrap_memories.json"
        self.memory_system = memory_system

        # Automatic memory capture settings
        self.auto_capture_enabled = True
        self.captured_insights = []
        self.important_keywords = {
            'discover', 'learn', 'accomplish', 'achieve', 'implement', 'create', 'build',
            'solve', 'fix', 'optimize', 'improve', 'develop', 'design', 'architecture',
            'system', 'framework', 'algorithm', 'solution', 'breakthrough', 'innovation',
            'milestone', 'progress', 'advancement', 'capability', 'feature', 'functionality'
        }
        self.technical_terms = {
            'vector', 'embedding', 'neural', 'network', 'machine', 'learning', 'ai',
            'memory', 'database', 'search', 'semantic', 'context', 'token', 'model',
            'algorithm', 'optimization', 'performance', 'efficiency', 'architecture'
        }

    def initialize_memory_system(self):
        """
        Initialize the complete memory system with bootstrap knowledge

        Returns:
            Configured MemoryIntegration instance ready for use
        """
        print("🧠 INITIALIZING MEMORY SYSTEM BOOTSTRAP...")
        print("=" * 60)

        # Initialize core memory system
        self.integration = MemoryIntegration(memory_path=self.memory_path)

        # Load bootstrap memories
        self._load_bootstrap_memories()

        # Verify system is working
        self._verify_bootstrap()

        print("✅ Memory system bootstrap complete!")
        print("🎯 AI now has persistent memory capabilities")
        print("=" * 60)

        # Enable automatic memory capture
        if self.auto_capture_enabled:
            self._enable_auto_capture()

        return self.integration

    def _load_bootstrap_memories(self):
        """Load essential memories about how to use the memory system"""

        # Core bootstrap memories - these teach the AI how to use itself
        bootstrap_memories = [
            {
                "content": "I have access to a persistent memory system that stores knowledge across conversations. I can query this system using 'python quick_memory.py query \"topic\"' to get relevant information from past discussions and implementations.",
                "importance": 1.0,
                "tags": ["memory", "system", "bootstrap", "usage", "persistent"],
                "context": {"type": "bootstrap", "purpose": "teach_memory_usage"}
            },
            {
                "content": "To add new memories during conversations, I use 'python quick_memory.py add \"content\" \"tags\" importance' where importance is 0.0-1.0. This helps me learn and remember important information for future conversations.",
                "importance": 1.0,
                "tags": ["memory", "add", "bootstrap", "learning", "usage"],
                "context": {"type": "bootstrap", "purpose": "teach_memory_creation"}
            },
            {
                "content": "For real-time memory assistance during conversations, I can use 'python conversation_memory_assistant.py --live' to get contextual memory suggestions and enhance my responses with historical knowledge.",
                "importance": 1.0,
                "tags": ["memory", "realtime", "conversation", "bootstrap", "enhancement"],
                "context": {"type": "bootstrap", "purpose": "teach_live_integration"}
            },
            {
                "content": "The memory system uses FAISS vector search with sentence transformers for semantic similarity. It provides context-aware memory retrieval with multi-factor relevance scoring based on importance, recency, usage patterns, and semantic matching.",
                "importance": 0.95,
                "tags": ["memory", "technical", "faiss", "embeddings", "bootstrap"],
                "context": {"type": "bootstrap", "purpose": "explain_architecture"}
            },
            {
                "content": "Memory consolidation automatically manages storage limits by keeping the most valuable memories based on importance, recency, and usage frequency. This ensures quality over quantity in the knowledge base.",
                "importance": 0.9,
                "tags": ["memory", "consolidation", "management", "bootstrap", "optimization"],
                "context": {"type": "bootstrap", "purpose": "explain_consolidation"}
            },
            {
                "content": "I should always check for relevant memories before responding to complex questions. Use 'python quick_memory.py query \"topic\"' to get historical context and avoid repeating work or forgetting important details.",
                "importance": 0.95,
                "tags": ["memory", "best_practice", "efficiency", "bootstrap", "workflow"],
                "context": {"type": "bootstrap", "purpose": "establish_best_practices"}
            },
            {
                "content": "The memory system tracks usage patterns and adapts relevance scoring over time. Memories that are frequently accessed or proven useful get higher priority in future retrievals.",
                "importance": 0.85,
                "tags": ["memory", "adaptive", "learning", "bootstrap", "optimization"],
                "context": {"type": "bootstrap", "purpose": "explain_adaptation"}
            },
            {
                "content": "When working on complex projects, I should regularly add memories about discoveries, solutions, and lessons learned. Tag them appropriately for future retrieval. Use importance scores to highlight critical insights.",
                "importance": 0.9,
                "tags": ["memory", "project", "workflow", "best_practice", "bootstrap"],
                "context": {"type": "bootstrap", "purpose": "establish_memory_habits"}
            }
        ]

        print(f"📚 Loading {len(bootstrap_memories)} bootstrap memories...")

        # Add each bootstrap memory to the system
        for i, memory_data in enumerate(bootstrap_memories, 1):
            try:
                memory_id = self.memory_system.add_memory(
                    content=memory_data["content"],
                    importance=memory_data["importance"],
                    tags=memory_data["tags"],
                    context=memory_data["context"]
                )
                print(f"  ✓ Bootstrap memory {i}: {memory_data['content'][:60]}... (ID: {memory_id[:8]})")

            except Exception as e:
                print(f"  ❌ Failed to add bootstrap memory {i}: {e}")

    def _verify_bootstrap(self):
        """Verify that bootstrap memories are accessible and working"""

        print("\n🔍 Verifying bootstrap functionality...")

        # Test basic memory retrieval
        test_queries = [
            "how to use memory system",
            "add memories during conversation",
            "real-time memory assistance"
        ]

        for query in test_queries:
            result = self.integration.get_context_memories(query, max_memories=2)
            if result["memory_count"] > 0:
                print(f"  ✅ Query '{query}' → {result['memory_count']} memories found")
            else:
                print(f"  ⚠️  Query '{query}' → No memories found")

        # Test memory statistics
        stats = self.integration.get_system_status()
        memory_count = stats['memory_system']['total_memories']
        integration_rate = stats['integration_layer']['integration_rate']

        print(f"  📊 System status: {memory_count} total memories, {integration_rate:.1%} integration rate")

    def get_bootstrap_instructions(self) -> str:
        """
        Get formatted instructions for using the bootstrapped memory system

        Returns:
            Formatted string with usage instructions
        """
        instructions = """
# 🧠 MEMORY SYSTEM BOOTSTRAP COMPLETE

## 🚀 Quick Start Commands

### Query Memories
```bash
python quick_memory.py query "your topic here"
```

### Add New Memories
```bash
python quick_memory.py add "memory content" "tag1,tag2" 0.8
```

### Live Conversation Assistant
```bash
python conversation_memory_assistant.py --live
```

### Check System Status
```bash
python quick_memory.py stats
```

## 💡 Best Practices

1. **Always query memories** before responding to complex questions
2. **Add important insights** as memories during conversations
3. **Use appropriate tags** for better future retrieval
4. **Set importance scores** (0.0-1.0) based on value
5. **Review memory suggestions** before final responses

## 🎯 Memory-Enhanced Workflow

1. **Start conversations** by checking relevant memories
2. **Use live assistant** for real-time enhancement
3. **Add discoveries** as memories during work
4. **Query contextually** using specific topics/questions

## 🔧 System Architecture

- **Vector Storage**: FAISS with sentence transformer embeddings
- **Semantic Search**: Cosine similarity with multi-factor scoring
- **Adaptive Learning**: Usage-based relevance optimization
- **Context Management**: Token-aware memory integration
- **Hardware Optimized**: CPU-optimized for Intel i7-6600U

---
*Memory system is now active and ready to enhance all conversations!*
        """.strip()

        return instructions

    def demonstrate_capabilities(self):
        """Demonstrate the bootstrapped memory capabilities"""

        print("\n🎯 DEMONSTRATING MEMORY CAPABILITIES")
        print("-" * 50)

        # Demonstrate memory query
        print("1. Memory Query Example:")
        result = self.integration.get_context_memories("how to use memory system", max_memories=2)
        if result["formatted_context"]:
            print(result["formatted_context"][:300] + "...")
        print()

        # Demonstrate memory addition
        print("2. Adding New Memory:")
        memory_id = self.integration.add_memory(
            "This is a demonstration of adding memories during conversations",
            importance=0.7,
            tags=["demo", "memory", "bootstrap"]
        )
        print(f"   ✅ Memory added (ID: {memory_id[:8]})")
        print()

        # Show system stats
        print("3. System Statistics:")
        stats = self.integration.get_system_status()
        print(f"   📊 {stats['memory_system']['total_memories']} memories stored")
        print(f"   🎯 {stats['integration_layer']['integration_rate']:.1%} integration rate")
        print()

        print("✅ Bootstrap demonstration complete!")

    def _enable_auto_capture(self):
        """Enable automatic capture of important information during conversations"""
        print("🔄 Enabling automatic memory capture...")
        print("📝 Will automatically identify and store important insights")

    def capture_important_insight(self, text: str, context: str = "conversation",
                                 importance: Optional[float] = None) -> bool:
        """
        Automatically capture important insights from conversations

        Args:
            text: The text content to analyze
            context: Context where the insight was found
            importance: Optional importance score override

        Returns:
            True if insight was captured and stored
        """
        if not self.auto_capture_enabled or not self.memory_system:
            return False

        # Analyze text for importance
        analysis = self._analyze_text_importance(text)

        if not analysis['is_important']:
            return False

        # Determine importance score
        if importance is None:
            importance = analysis['calculated_importance']

        # Extract tags automatically
        tags = self._extract_tags_from_text(text, analysis)

        # Store the memory
        try:
            memory_id = self.memory_system.add_memory(
                content=text,
                importance=importance,
                tags=tags,
                context={
                    "type": "auto_captured",
                    "source": context,
                    "analysis": analysis,
                    "timestamp": datetime.now().isoformat()
                }
            )

            # Track captured insights
            self.captured_insights.append({
                "id": memory_id,
                "content": text[:100] + "..." if len(text) > 100 else text,
                "importance": importance,
                "tags": tags,
                "context": context,
                "timestamp": datetime.now()
            })

            print(f"🧠 Auto-captured insight: {text[:80]}... (importance: {importance:.2f})")
            return True

        except Exception as e:
            print(f"⚠️  Failed to auto-capture insight: {e}")
            return False

    def _analyze_text_importance(self, text: str) -> Dict[str, Any]:
        """
        Analyze text to determine if it's important enough to remember

        Returns:
            Dict with importance analysis
        """
        text_lower = text.lower()
        words = set(text_lower.split())

        # Count important keywords
        important_count = len(self.important_keywords & words)
        technical_count = len(self.technical_terms & words)

        # Length and structure analysis
        word_count = len(words)
        has_sentences = len(re.findall(r'[.!?]', text)) > 0
        has_code = bool(re.search(r'```|`|\.py|\.sh|python|bash', text))

        # Calculate importance score
        base_score = 0.3  # Minimum threshold

        # Keyword bonuses
        base_score += min(0.3, important_count * 0.1)  # Max 0.3 for important keywords
        base_score += min(0.2, technical_count * 0.05)  # Max 0.2 for technical terms

        # Structure bonuses
        if has_sentences and word_count > 10:
            base_score += 0.2  # Well-formed content
        if has_code:
            base_score += 0.2  # Contains code/implementation
        if word_count > 50:
            base_score += 0.1  # Substantial content

        # Context bonuses
        if any(term in text_lower for term in ['solution', 'breakthrough', 'accomplished', 'completed']):
            base_score += 0.2  # Achievement indicators

        return {
            'is_important': base_score >= 0.5,
            'calculated_importance': min(1.0, base_score),
            'important_keywords_found': important_count,
            'technical_terms_found': technical_count,
            'word_count': word_count,
            'has_sentences': has_sentences,
            'has_code': has_code
        }

    def _extract_tags_from_text(self, text: str, analysis: Dict[str, Any]) -> List[str]:
        """Automatically extract relevant tags from text"""
        tags = []
        text_lower = text.lower()

        # Add tags based on content analysis
        if analysis['technical_terms_found'] > 0:
            tags.append("technical")

        if analysis['has_code']:
            tags.append("code")

        if analysis['important_keywords_found'] > 0:
            tags.append("important")

        # Add specific technical tags
        for term in self.technical_terms:
            if term in text_lower:
                tags.append(term)

        # Add achievement/project tags
        if any(word in text_lower for word in ['implement', 'create', 'build', 'develop']):
            tags.append("implementation")

        if any(word in text_lower for word in ['system', 'architecture', 'framework']):
            tags.append("architecture")

        if any(word in text_lower for word in ['optimize', 'improve', 'performance']):
            tags.append("optimization")

        return list(set(tags))  # Remove duplicates

    def get_capture_statistics(self) -> Dict[str, Any]:
        """Get statistics about automatic memory capture"""
        if not self.captured_insights:
            return {"captured_insights": 0, "total_importance": 0, "average_importance": 0}

        total_importance = sum(insight['importance'] for insight in self.captured_insights)
        avg_importance = total_importance / len(self.captured_insights)

        return {
            "captured_insights": len(self.captured_insights),
            "total_importance": total_importance,
            "average_importance": avg_importance,
            "recent_captures": self.captured_insights[-3:] if self.captured_insights else []
        }

    def review_and_cleanup_captures(self, max_age_days: int = 30):
        """Review and cleanup old automatic captures"""
        if not self.captured_insights:
            return

        cutoff_date = datetime.now() - timedelta(days=max_age_days)

        # Keep only recent captures
        self.captured_insights = [
            insight for insight in self.captured_insights
            if insight['timestamp'] > cutoff_date
        ]

        print(f"🧹 Cleaned up automatic captures, keeping {len(self.captured_insights)} recent insights")


def bootstrap_memory_system(memory_path: str = "vector_memory"):
    """
    Bootstrap the complete memory system for AI initialization

    Args:
        memory_path: Path to memory storage directory

    Returns:
        Fully initialized MemoryIntegration system
    """
    bootstrap = MemoryBootstrap(memory_path=memory_path)
    integration = bootstrap.initialize_memory_system()

    return integration


def get_memory_bootstrap_instructions():
    """Get formatted bootstrap instructions without initializing"""
    bootstrap = MemoryBootstrap()
    return bootstrap.get_bootstrap_instructions()


if __name__ == "__main__":
    print("🧠 MEMORY SYSTEM BOOTSTRAP")
    print("=" * 50)

    # Initialize the memory system with bootstrap
    integration = bootstrap_memory_system()

    # Show instructions
    print(get_memory_bootstrap_instructions())

    # Demonstrate capabilities
    bootstrap = MemoryBootstrap()
    bootstrap.integration = integration
    bootstrap.demonstrate_capabilities()

    print("\n🎉 Memory system is now fully bootstrapped and ready for use!")
    print("💡 Use 'python quick_memory.py query \"topic\"' to access memories during conversations")
