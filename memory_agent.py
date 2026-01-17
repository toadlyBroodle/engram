"""
Memory Agent - RLM-style iterative memory access

Instead of passive memory injection, this agent can actively query
memory multiple times during a single response, following links
and building understanding iteratively.

Key difference from PassiveMemoryProxy:
- Passive: search once → inject top-k → LLM responds
- Active: LLM decides what to search → calls tools → reasons → searches more → responds
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from engram_pkg import VectorMemory
from memory_tools import MemoryTools, MEMORY_TOOL_DECLARATIONS
from memory_extractor import MemoryExtractor
from token_tracker import tracker

# Check for Gemini SDK
_gemini_available = False
try:
    from google import genai
    from google.genai import types
    _gemini_available = True
except ImportError:
    pass


@dataclass
class AgentConfig:
    """Configuration for the memory agent"""
    # Memory settings
    memory_path: str = "vector_memory"
    
    # LLM settings
    model: str = "gemini-2.0-flash"
    extraction_model: str = "gemini-2.0-flash-lite"
    max_tokens: int = 4096
    
    # Agent settings
    max_tool_calls: int = 10  # Max tool calls per turn (prevent infinite loops)
    extraction_enabled: bool = True
    verbose: bool = False


class MemoryAgent:
    """
    RLM-style memory agent with active tool-based retrieval.
    
    The LLM can call memory tools iteratively:
    1. search_memory - semantic search
    2. get_related_memories - follow connections
    3. get_recent_memories - temporal context
    4. store_memory - save new information
    
    This allows the LLM to reason about what it needs to know,
    query memory, and follow links to build understanding.
    """
    
    def __init__(self, config: AgentConfig = None, api_key: str = None):
        """
        Initialize the memory agent.
        
        Args:
            config: Agent configuration
            api_key: Gemini API key (or uses GEMINI_API_KEY env var)
        """
        self.config = config or AgentConfig()
        
        # Initialize memory system
        self.memory_system = VectorMemory(storage_path=self.config.memory_path)
        
        # Initialize memory tools
        self.tools = MemoryTools(self.memory_system)
        
        # Initialize extractor for async memory creation
        self.extractor = MemoryExtractor(
            memory_system=self.memory_system,
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
        
        # Build tool declarations for Gemini
        self._tool_config = types.Tool(function_declarations=[
            types.FunctionDeclaration(**decl) for decl in MEMORY_TOOL_DECLARATIONS
        ])
        
        # Conversation state
        self.conversation_history = []
        self.system_prompt = """You are a helpful AI assistant with persistent memory.

You have memory tools to recall past conversations. USE THEM:

SEARCH STRATEGY:
1. ALWAYS search memory for factual questions - don't guess
2. If first search doesn't find the answer, try DIFFERENT search terms
3. Be SPECIFIC with search queries - include key nouns from the question
4. For questions about "when", search for the event + time-related terms

ANSWERING:
- Base answers ONLY on what you find in memory
- If you don't find it, say "I don't have that information"
- Match the specificity the question asks for"""
        
        # Statistics
        self.stats = {
            "messages_processed": 0,
            "tool_calls_made": 0,
            "memories_retrieved": 0,
            "memories_stored": 0,
            "session_start": datetime.now()
        }
        
        print(f"🧠 Memory Agent initialized (RLM-style)")
        print(f"   Memory store: {len(self.memory_system.memories)} memories")
        print(f"   Extraction: {'enabled' if self.config.extraction_enabled else 'disabled'}")
    
    def set_system_prompt(self, prompt: str):
        """Set the base system prompt"""
        self.system_prompt = prompt
    
    def _execute_tool_call(self, function_call) -> str:
        """Execute a tool call and return the result as a string"""
        name = function_call.name
        args = dict(function_call.args) if function_call.args else {}
        
        if self.config.verbose:
            print(f"   🔧 Tool: {name}({args})")
        
        try:
            if name == "search_memory":
                result = self.tools.search_memory(
                    query=args.get("query", ""),
                    limit=args.get("limit", 5)
                )
                self.stats["memories_retrieved"] += len(result)
                
            elif name == "get_related_memories":
                result = self.tools.get_related_memories(
                    memory_id=args.get("memory_id", ""),
                    limit=args.get("limit", 3)
                )
                self.stats["memories_retrieved"] += len(result)
                
            elif name == "get_recent_memories":
                result = self.tools.get_recent_memories(
                    hours=args.get("hours", 24),
                    limit=args.get("limit", 5)
                )
                self.stats["memories_retrieved"] += len(result)
                
            elif name == "store_memory":
                result = self.tools.store_memory(
                    content=args.get("content", ""),
                    importance=args.get("importance", 0.7),
                    tags=args.get("tags")
                )
                self.stats["memories_stored"] += 1
                
            else:
                result = {"error": f"Unknown tool: {name}"}
            
            self.stats["tool_calls_made"] += 1
            return json.dumps(result, default=str)
            
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    def chat(self, user_message: str) -> str:
        """
        Send a message and get a response with active memory retrieval.
        
        The LLM can call memory tools multiple times before responding.
        
        Args:
            user_message: The user's message
            
        Returns:
            The assistant's response
        """
        if not self.client:
            return "Error: Gemini client not initialized. Check your API key."
        
        self.stats["messages_processed"] += 1
        
        # Build conversation contents
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
        
        # Agent loop - keep calling until we get a text response
        tool_call_count = 0
        
        while tool_call_count < self.config.max_tool_calls:
            try:
                response = self.client.models.generate_content(
                    model=self.config.model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=self.system_prompt,
                        max_output_tokens=self.config.max_tokens,
                        tools=[self._tool_config]
                    )
                )
                
                # Track token usage
                tracker.record_brain_usage(response.usage_metadata, self.config.model)
                
            except Exception as e:
                error_msg = f"Error calling LLM: {e}"
                print(f"❌ {error_msg}")
                return error_msg
            
            # Check if we have function calls to process
            candidate = response.candidates[0] if response.candidates else None
            if not candidate or not candidate.content:
                return "Error: No response from model"
            
            parts = candidate.content.parts
            
            # Look for function calls
            function_calls = [p.function_call for p in parts if hasattr(p, 'function_call') and p.function_call]
            
            if not function_calls:
                # No more tool calls - extract text response
                text_parts = [p.text for p in parts if hasattr(p, 'text') and p.text]
                assistant_message = "\n".join(text_parts) if text_parts else ""
                break
            
            # Execute function calls and add results to conversation
            # First, add the model's function call to contents
            contents.append(types.Content(
                role="model",
                parts=parts
            ))
            
            # Execute each function call and build response parts
            function_response_parts = []
            for fc in function_calls:
                result = self._execute_tool_call(fc)
                function_response_parts.append(
                    types.Part(function_response=types.FunctionResponse(
                        name=fc.name,
                        response={"result": result}
                    ))
                )
                tool_call_count += 1
            
            # Add function responses to conversation
            contents.append(types.Content(
                role="user",
                parts=function_response_parts
            ))
            
            if self.config.verbose:
                print(f"   📊 Tool calls: {tool_call_count}/{self.config.max_tool_calls}")
        
        else:
            # Hit max tool calls
            assistant_message = "I apologize, but I'm having trouble processing this request. Could you try rephrasing?"
        
        # Update conversation history (just user and final assistant message)
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        self.conversation_history.append({
            "role": "model",
            "content": assistant_message
        })
        
        # Keep history manageable
        if len(self.conversation_history) > 40:
            self.conversation_history = self.conversation_history[-40:]
        
        # Async memory extraction from the exchange
        if self.config.extraction_enabled:
            self.extractor.extract_async(user_message, assistant_message)
        
        return assistant_message
    
    def add_memory(self, content: str, importance: float = 0.7, 
                   tags: List[str] = None) -> str:
        """Manually add a memory"""
        return self.tools.store_memory(content, importance, tags)["id"]
    
    def search_memories(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search memories"""
        return self.tools.search_memory(query, limit)
    
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID"""
        return self.memory_system.delete_memory(memory_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        session_duration = (datetime.now() - self.stats["session_start"]).total_seconds()
        
        return {
            **self.stats,
            "session_duration_seconds": session_duration,
            "memory_count": len(self.memory_system.memories),
            "extraction_stats": self.extractor.get_stats(),
            "conversation_length": len(self.conversation_history)
        }
    
    def clear_conversation(self):
        """Clear conversation history (memories persist)"""
        self.conversation_history = []
        print("🗑️  Conversation cleared (memories preserved)")
    
    def shutdown(self, show_stats: bool = False):
        """Graceful shutdown"""
        self.extractor.stop_worker()
        self.memory_system.force_save()
        print("👋 Memory agent shut down")
        
        if show_stats:
            print()
            print(tracker.format_stats())
    
    def get_token_stats(self) -> dict:
        """Get current token usage stats"""
        return tracker.get_stats()
    
    def print_token_stats(self):
        """Print token usage stats"""
        print(tracker.format_stats())


def create_agent(verbose: bool = False, **kwargs) -> MemoryAgent:
    """Factory function to create a memory agent"""
    config = AgentConfig(verbose=verbose, **kwargs)
    return MemoryAgent(config=config)


if __name__ == "__main__":
    # Quick test
    print("Testing Memory Agent")
    print("=" * 40)
    
    agent = create_agent(verbose=True)
    
    print(f"\nStats: {agent.get_stats()}")
    print("\nAgent ready! Use agent.chat('message') to send messages.")
