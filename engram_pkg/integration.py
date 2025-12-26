"""
Engram Integration Layer

Seamlessly integrates Engram memory into AI decision-making and conversations.
This is the main interface for ensuring relevant memories are consistently available
in working context.

Usage:
    integrator = MemoryIntegration()
    context_memories = integrator.get_context_memories("current conversation topic")
    # Memories are now available for inclusion in responses
"""

import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime

from .core import VectorMemory
from .context import MemoryContextIntegrator, create_memory_context_integrator
from .window import ContextWindowManager, create_default_context_manager


class MemoryIntegration:
    """
    Main integration layer for Engram memory in AI systems.

    Provides a unified interface for:
    - Context-aware memory retrieval
    - Efficient context window management
    - Adaptive learning from usage patterns
    - Seamless integration with conversations
    """

    def __init__(self, memory_path: str = "vector_memory",
                 context_window_tokens: int = 8000,
                 auto_save: bool = True):
        """
        Initialize the memory integration system

        Args:
            memory_path: Path to vector memory storage
            context_window_tokens: Maximum context window size
            auto_save: Whether to automatically save memory changes
        """
        # Initialize core components
        self.memory_system = VectorMemory(memory_path)
        self.context_integrator = create_memory_context_integrator(self.memory_system)
        self.context_manager = create_default_context_manager(context_window_tokens)

        self.auto_save = auto_save
        self.conversation_history = []

        # Initialize automatic memory capture (optional)
        # Import here to avoid circular dependency
        try:
            from bootstrap_memory import MemoryBootstrap
            self.bootstrap_handler = MemoryBootstrap(memory_path, self)
            self.auto_capture_enabled = True

            # Set the integration reference in bootstrap handler
            if hasattr(self.bootstrap_handler, 'set_integration_reference'):
                self.bootstrap_handler.set_integration_reference(self)
        except ImportError:
            # Bootstrap functionality not available in package mode
            self.bootstrap_handler = None
            self.auto_capture_enabled = False

        print("🧠 Memory Integration System initialized")
        print("🔄 Automatic memory capture enabled")

    def update_conversation(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Update the conversation context with new message

        Args:
            role: Message role ('user', 'assistant', 'system')
            content: Message content
            metadata: Optional metadata about the message
        """
        # Update context integrator
        self.context_integrator.update_conversation_context(role, content)

        # Store conversation history
        message_entry = {
            "role": role,
            "content": content,
            "timestamp": datetime.now(),
            "metadata": metadata or {},
            "token_estimate": len(content) // 4  # Rough estimate
        }
        self.conversation_history.append(message_entry)

        # Keep only recent history to avoid memory bloat
        if len(self.conversation_history) > 50:  # Keep last 50 messages
            self.conversation_history = self.conversation_history[-50:]

        # Automatically capture important insights
        if self.auto_capture_enabled and len(content) > 20:  # Only check substantial messages
            self._check_and_capture_insight(content, role)

    def get_context_memories(self, query: Optional[str] = None,
                           max_memories: int = 5,
                           context_tokens_used: int = 0) -> Dict[str, Any]:
        """
        Get memories optimized for current context

        Args:
            query: Optional specific query to focus retrieval
            max_memories: Maximum number of memories to return
            context_tokens_used: Tokens already used in current context

        Returns:
            Dictionary with integrated memories and metadata
        """
        # Update context manager with current usage
        self.context_manager.set_context_usage(context_tokens_used)

        # Get relevant memories from context integrator
        relevant_memories = self.context_integrator.get_relevant_memories_for_context()

        # Limit to requested maximum
        relevant_memories = relevant_memories[:max_memories]

        # Optimize for context window
        optimized_memories = self.context_manager.optimize_memories_for_context(relevant_memories)

        # Format for integration
        formatted_context = self.context_manager.format_optimized_context(optimized_memories)

        # Update usage statistics
        for memory in optimized_memories:
            self.memory_system.update_memory_usage(memory.original_id)

        return {
            "memories": optimized_memories,
            "formatted_context": formatted_context,
            "memory_count": len(optimized_memories),
            "tokens_used": sum(memory.token_count for memory in optimized_memories),
            "context_stats": self.context_manager.get_context_stats(),
            "integration_stats": self.context_integrator.get_memory_stats()
        }

    def add_memory(self, content: str, importance: float = 0.5,
                  tags: List[str] = None, context: Dict[str, Any] = None) -> str:
        """
        Add a new memory to the system

        Args:
            content: Memory content
            importance: Importance score (0.0-1.0)
            tags: Categorization tags
            context: Additional context information

        Returns:
            Memory ID
        """
        memory_id = self.memory_system.store_memory(content, importance, tags, context)

        if self.auto_save:
            self.memory_system._save_persistent_data()

        return memory_id

    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory from the system

        Args:
            memory_id: ID of the memory to delete

        Returns:
            True if memory was deleted, False if not found
        """
        deleted = self.memory_system.delete_memory(memory_id)

        if deleted and self.auto_save:
            self.memory_system._save_persistent_data()

        return deleted

    def get_conversation_context(self, include_memories: bool = True) -> Dict[str, Any]:
        """
        Get complete conversation context including integrated memories

        Args:
            include_memories: Whether to include relevant memories

        Returns:
            Complete context dictionary
        """
        context_data = {
            "conversation_history": self.conversation_history[-10:],  # Last 10 messages
            "current_topic": getattr(self.context_integrator.conversation_context, 'current_topic', ''),
            "active_tags": self.context_integrator.conversation_context.active_tags,
            "timestamp": datetime.now()
        }

        if include_memories:
            memory_context = self.get_context_memories()
            context_data.update({
                "integrated_memories": memory_context["formatted_context"],
                "memory_stats": memory_context["integration_stats"]
            })

        return context_data

    def search_memories(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search memories with enhanced context awareness

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of memory dictionaries
        """
        memories = self.memory_system.retrieve_memory(query, limit)

        # Enhance with usage statistics
        enhanced_memories = []
        for memory in memories:
            enhanced = {
                "id": memory.id,
                "content": memory.content,
                "importance": memory.importance,
                "tags": memory.tags,
                "timestamp": memory.timestamp.isoformat(),
                "usage_count": self.context_integrator.memory_usage_stats.get(memory.id, 0),
                "last_used": memory.last_accessed.isoformat() if memory.last_accessed else None
            }
            enhanced_memories.append(enhanced)

        return enhanced_memories

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        memory_stats = self.memory_system.get_memory_stats()
        context_stats = self.context_manager.get_context_stats()
        integration_stats = self.context_integrator.get_memory_stats()

        return {
            "memory_system": memory_stats,
            "context_manager": context_stats,
            "integration_layer": integration_stats,
            "conversation_messages": len(self.conversation_history),
            "auto_save_enabled": self.auto_save,
            "last_activity": datetime.now()
        }

    def optimize_for_performance(self):
        """Optimize system for better performance"""
        # Consolidate memories if needed
        memory_count = len(self.memory_system.memories)
        if memory_count > 1000:  # Arbitrary threshold
            self.memory_system.consolidate_memories(max_memories=800)

        # Clear old conversation history
        if len(self.conversation_history) > 100:
            # Keep only recent messages
            self.conversation_history = self.conversation_history[-50:]

        print("⚡ System optimized for performance")

    def _check_autonomous_milestone(self, context: str = "operation_complete"):
        """Check if current state warrants an autonomous milestone commit"""
        if hasattr(self.bootstrap_handler, 'check_and_commit_milestone'):
            self.bootstrap_handler.check_and_commit_milestone(context)

    def _check_and_capture_insight(self, content: str, speaker: str):
        """Check if content contains important insights worth remembering"""
        try:
            captured = self.bootstrap_handler.capture_important_insight(
                text=content,
                context=f"conversation_{speaker}",
                importance=None  # Let the system determine importance
            )
            if captured:
                # Update memory usage statistics for the newly captured memory
                # This will help with future relevance scoring
                pass  # The capture method already handles this
        except Exception as e:
            # Don't let auto-capture failures break the conversation flow
            print(f"⚠️  Auto-capture failed: {e}")

    def get_auto_capture_statistics(self) -> Dict[str, Any]:
        """Get statistics about automatic memory capture"""
        return self.bootstrap_handler.get_capture_statistics()

    def toggle_auto_capture(self, enabled: bool = None) -> bool:
        """
        Toggle automatic memory capture on/off

        Args:
            enabled: If provided, set to this state. If None, toggle current state.

        Returns:
            New state of auto capture
        """
        if enabled is not None:
            self.auto_capture_enabled = enabled
        else:
            self.auto_capture_enabled = not self.auto_capture_enabled

        state_text = "enabled" if self.auto_capture_enabled else "disabled"
        print(f"🔄 Automatic memory capture {state_text}")

        return self.auto_capture_enabled

    def export_context_snapshot(self, filepath: str = None) -> str:
        """
        Export current context snapshot for debugging/analysis

        Args:
            filepath: Optional file path to save to

        Returns:
            JSON string of context snapshot
        """
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "conversation_context": self.get_conversation_context(),
            "system_status": self.get_system_status(),
            "recent_memories": self.search_memories("recent", limit=3)
        }

        snapshot_json = json.dumps(snapshot, indent=2, default=str)

        if filepath:
            with open(filepath, 'w') as f:
                f.write(snapshot_json)

        return snapshot_json


def create_memory_integration(memory_path: str = "vector_memory",
                            context_window: int = 8000) -> MemoryIntegration:
    """
    Factory function to create a memory integration system

    Args:
        memory_path: Path to memory storage
        context_window: Context window size in tokens

    Returns:
        Configured MemoryIntegration instance
    """
    return MemoryIntegration(
        memory_path=memory_path,
        context_window_tokens=context_window
    )


# Convenience functions for easy integration
def get_relevant_memories(query: str = "", max_tokens: int = 1000) -> str:
    """
    Quick function to get relevant memories for a query

    Args:
        query: Search query
        max_tokens: Maximum tokens to use

    Returns:
        Formatted memory context string
    """
    integrator = create_memory_integration()
    result = integrator.get_context_memories(query, context_tokens_used=0)
    return result["formatted_context"]


if __name__ == "__main__":
    # Test the integration system
    print("Testing Memory Integration System")
    print("=" * 40)

    # Create integration system
    integration = create_memory_integration()

    # Add some test memories
    integration.add_memory(
        "Vector databases use embeddings to enable semantic search",
        importance=0.8,
        tags=["ai", "database", "vector"]
    )

    integration.add_memory(
        "Context window management is crucial for LLM efficiency",
        importance=0.9,
        tags=["llm", "performance", "context"]
    )

    # Simulate conversation
    integration.update_conversation("user", "How do vector databases work?")
    integration.update_conversation("assistant", "Vector databases store data as high-dimensional vectors...")

    # Get context memories
    context_result = integration.get_context_memories()
    print("INTEGRATED CONTEXT:")
    print(context_result["formatted_context"])
    print(f"\nMEMORY COUNT: {context_result['memory_count']}")
    print(f"TOKENS USED: {context_result['tokens_used']}")

    # Get system status
    status = integration.get_system_status()
    print(f"\nSYSTEM STATUS: {status['memory_system']['total_memories']} memories stored")
