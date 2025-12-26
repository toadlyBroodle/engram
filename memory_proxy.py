"""
Passive Memory Proxy

A transparent proxy that sits between you and the LLM API.
Automatically injects relevant memories into context and extracts new memories from responses.

The LLM never knows memory is happening - it just sees enhanced context.

Usage:
    proxy = PassiveMemoryProxy()
    response = proxy.chat("Your message here")
"""

import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from engram_pkg import VectorMemory
from memory_integration import MemoryIntegration
from memory_extractor import MemoryExtractor

# Check for Gemini SDK
_gemini_available = False
try:
    from google import genai
    from google.genai import types
    _gemini_available = True
except ImportError:
    pass


@dataclass
class ProxyConfig:
    """Configuration for the memory proxy"""
    # Memory settings
    memory_path: str = "vector_memory"
    max_memories_to_inject: int = 5
    min_memory_importance: float = 0.2
    
    # LLM settings
    model: str = "gemini-2.0-flash"
    extraction_model: str = "gemini-2.0-flash-lite"
    max_tokens: int = 4096
    
    # Context settings
    memory_token_budget: int = 1000  # Max tokens for memory context
    include_timestamps: bool = False
    
    # Behavior
    extraction_enabled: bool = True
    verbose: bool = False


class PassiveMemoryProxy:
    """
    Transparent memory-enhanced LLM proxy.
    
    Synchronous path (blocking):
    1. User message → Search memories → Inject context → Call LLM → Return response
    
    Asynchronous path (background):
    2. After response → Extract insights → Store new memories
    """
    
    def __init__(self, config: ProxyConfig = None, api_key: str = None):
        """
        Initialize the memory proxy.
        
        Args:
            config: Proxy configuration
            api_key: Gemini API key (or uses GEMINI_API_KEY env var)
        """
        self.config = config or ProxyConfig()
        
        # Initialize memory system
        self.memory = MemoryIntegration(
            memory_path=self.config.memory_path,
            auto_save=True
        )
        
        # Initialize extractor for async memory creation
        self.extractor = MemoryExtractor(
            memory_system=self.memory.memory_system,
            model=self.config.extraction_model
        )
        
        # Start extraction worker
        if self.config.extraction_enabled:
            self.extractor.start_worker()
        
        # Initialize Gemini client
        self.client = None
        if _gemini_available:
            try:
                key = api_key or os.environ.get("GEMINI_API_KEY")
                if key:
                    self.client = genai.Client(api_key=key)
                    print(f"✅ Gemini client initialized ({self.config.model})")
                else:
                    print("❌ GEMINI_API_KEY not found in environment")
            except Exception as e:
                print(f"❌ Failed to initialize Gemini client: {e}")
        else:
            print("❌ google-genai SDK not installed. Install with: pip install google-genai")
        
        # Conversation state
        self.conversation_history = []
        self.system_prompt = "You are a helpful assistant."
        
        # Statistics
        self.stats = {
            "messages_processed": 0,
            "memories_injected": 0,
            "tokens_used_for_memory": 0,
            "session_start": datetime.now()
        }
        
        print(f"🧠 Passive Memory Proxy initialized")
        print(f"   Memory store: {len(self.memory.memory_system.memories)} memories")
        print(f"   Extraction: {'enabled' if self.config.extraction_enabled else 'disabled'}")
    
    def set_system_prompt(self, prompt: str):
        """Set the base system prompt (memory context will be prepended)"""
        self.system_prompt = prompt
    
    def _get_memory_context(self, user_message: str) -> str:
        """
        Get relevant memories formatted for context injection.
        
        This is the SYNCHRONOUS retrieval step - must complete before LLM call.
        """
        # Search for relevant memories
        memories = self.memory.memory_system.retrieve_memory(
            query=user_message,
            limit=self.config.max_memories_to_inject,
            min_importance=self.config.min_memory_importance
        )
        
        if not memories:
            return ""
        
        # Format memories for injection
        memory_lines = []
        total_chars = 0
        char_budget = self.config.memory_token_budget * 4  # Rough chars per token
        
        for mem in memories:
            line = f"• {mem.content}"
            if self.config.include_timestamps:
                line += f" (from {mem.timestamp.strftime('%Y-%m-%d')})"
            
            if total_chars + len(line) > char_budget:
                break
            
            memory_lines.append(line)
            total_chars += len(line)
            self.stats["memories_injected"] += 1
        
        if not memory_lines:
            return ""
        
        self.stats["tokens_used_for_memory"] += total_chars // 4
        
        context = "## Context from previous conversations:\n"
        context += "\n".join(memory_lines)
        context += "\n\nUse this context naturally when relevant. Don't explicitly mention 'from our previous conversations'."
        
        return context
    
    def _build_system_with_memory(self, memory_context: str) -> str:
        """Build the system prompt with memory context"""
        if memory_context:
            return f"{memory_context}\n\n---\n\n{self.system_prompt}"
        return self.system_prompt
    
    def chat(self, user_message: str) -> str:
        """
        Send a message and get a response with transparent memory enhancement.
        
        Args:
            user_message: The user's message
            
        Returns:
            The assistant's response
        """
        if not self.client:
            return "Error: Gemini client not initialized. Check your API key."
        
        self.stats["messages_processed"] += 1
        
        # ═══════════════════════════════════════════════════════════
        # STEP 1: RETRIEVE MEMORIES (synchronous, ~10-50ms)
        # ═══════════════════════════════════════════════════════════
        memory_context = self._get_memory_context(user_message)
        
        if self.config.verbose and memory_context:
            print(f"📚 Injecting {memory_context.count('•')} memories into context")
        
        # ═══════════════════════════════════════════════════════════
        # STEP 2: BUILD SYSTEM PROMPT WITH MEMORY
        # ═══════════════════════════════════════════════════════════
        system_with_memory = self._build_system_with_memory(memory_context)
        
        # ═══════════════════════════════════════════════════════════
        # STEP 3: BUILD CONVERSATION CONTENTS
        # ═══════════════════════════════════════════════════════════
        contents = []
        for msg in self.conversation_history:
            contents.append(types.Content(
                role=msg["role"],
                parts=[types.Part(text=msg["content"])]
            ))
        
        # Add current user message
        contents.append(types.Content(
            role="user",
            parts=[types.Part(text=user_message)]
        ))
        
        # ═══════════════════════════════════════════════════════════
        # STEP 4: CALL LLM (synchronous, blocking)
        # ═══════════════════════════════════════════════════════════
        try:
            response = self.client.models.generate_content(
                model=self.config.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_with_memory,
                    max_output_tokens=self.config.max_tokens
                )
            )
            assistant_message = response.text
            
        except Exception as e:
            error_msg = f"Error calling LLM: {e}"
            print(f"❌ {error_msg}")
            return error_msg
        
        # ═══════════════════════════════════════════════════════════
        # STEP 5: TRACK CONVERSATION
        # ═══════════════════════════════════════════════════════════
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        self.conversation_history.append({
            "role": "model",  # Gemini uses "model" instead of "assistant"
            "content": assistant_message
        })
        
        # Keep history manageable
        if len(self.conversation_history) > 40:
            self.conversation_history = self.conversation_history[-40:]
        
        # ═══════════════════════════════════════════════════════════
        # STEP 6: EXTRACT MEMORIES (async, non-blocking)
        # ═══════════════════════════════════════════════════════════
        if self.config.extraction_enabled:
            self.extractor.extract_async(user_message, assistant_message)
        
        return assistant_message
    
    def add_memory(self, content: str, importance: float = 0.7, 
                   tags: List[str] = None) -> str:
        """Manually add a memory"""
        return self.memory.add_memory(content, importance, tags or [])
    
    def search_memories(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search memories"""
        return self.memory.search_memories(query, limit)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get proxy statistics"""
        session_duration = (datetime.now() - self.stats["session_start"]).total_seconds()
        
        return {
            **self.stats,
            "session_duration_seconds": session_duration,
            "memory_count": len(self.memory.memory_system.memories),
            "extraction_stats": self.extractor.get_stats(),
            "conversation_length": len(self.conversation_history)
        }
    
    def clear_conversation(self):
        """Clear conversation history (memories persist)"""
        self.conversation_history = []
        print("🗑️  Conversation cleared (memories preserved)")
    
    def shutdown(self):
        """Graceful shutdown"""
        self.extractor.stop_worker()
        self.memory.memory_system.force_save()
        print("👋 Memory proxy shut down")


def create_proxy(verbose: bool = False, **kwargs) -> PassiveMemoryProxy:
    """Factory function to create a memory proxy"""
    config = ProxyConfig(verbose=verbose, **kwargs)
    return PassiveMemoryProxy(config=config)


if __name__ == "__main__":
    # Quick test
    print("Testing Passive Memory Proxy")
    print("=" * 40)
    
    proxy = create_proxy(verbose=True)
    
    print(f"\nStats: {proxy.get_stats()}")
    print("\nProxy ready! Use proxy.chat('message') to send messages.")
