"""
Engram - Passive Memory Layer for AI Conversations

This package provides a transparent memory system that automatically enhances
AI conversations with relevant context from past interactions.

Main Components:
- MemoryIntegration: Main interface for memory operations
- VectorMemory: Core FAISS-based vector storage
- MemoryContextIntegrator: Context-aware memory selection
- PassiveMemoryProxy: Transparent LLM proxy with memory injection

Quick Start:
    from engram_pkg import MemoryIntegration

    # Initialize memory system
    memory = MemoryIntegration()

    # Add memories
    memory.add_memory("Important insight", importance=0.8)

    # Retrieve relevant memories
    results = memory.get_context_memories("related topic")
"""

from .integration import MemoryIntegration
from .core import VectorMemory
from .context import MemoryContextIntegrator, create_memory_context_integrator
from .adaptive import create_adaptive_memory_system
from .window import ContextWindowManager, create_default_context_manager

__version__ = "2.0.0"
__all__ = [
    "MemoryIntegration",
    "VectorMemory",
    "MemoryContextIntegrator",
    "create_memory_context_integrator",
    "create_adaptive_memory_system",
    "ContextWindowManager",
    "create_default_context_manager",
]
