"""
Memory Tools - Tool definitions for RLM-style memory access

Instead of passive injection, these tools let the LLM actively query
and traverse memory as needed during a conversation.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from engram_pkg import VectorMemory


@dataclass
class MemoryToolResult:
    """Result from a memory tool call"""
    success: bool
    data: Any
    message: str


class MemoryTools:
    """
    Tools for active memory retrieval.
    
    The LLM can call these tools iteratively to:
    1. Search for relevant memories
    2. Follow related memory links
    3. Get specific memories by ID
    4. Store new memories
    """
    
    def __init__(self, memory_system: VectorMemory):
        self.memory = memory_system
    
    def search_memory(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search memories by semantic similarity.
        
        Args:
            query: Natural language search query
            limit: Maximum number of results (default 5)
            
        Returns:
            List of matching memories with content, importance, tags, and IDs
        """
        memories = self.memory.retrieve_memory(query, limit=limit)
        
        results = []
        for mem in memories:
            results.append({
                "id": mem.id,
                "content": mem.content,
                "importance": round(mem.importance, 2),
                "tags": mem.tags,
                "access_count": mem.access_count,
                "has_related": len(mem.related_memories) > 0,
                "related_count": len(mem.related_memories)
            })
        
        return results
    
    def get_related_memories(self, memory_id: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Get memories related to a specific memory.
        
        Args:
            memory_id: ID of the memory to find relations for (can be partial)
            limit: Maximum number of related memories to return
            
        Returns:
            List of related memories
        """
        # Support partial ID matching
        full_id = self._resolve_memory_id(memory_id)
        if not full_id:
            return []
        
        memory = self.memory.get_memory_by_id(full_id)
        if not memory or not memory.related_memories:
            return []
        
        results = []
        for related_id in memory.related_memories[:limit]:
            related = self.memory.get_memory_by_id(related_id)
            if related:
                results.append({
                    "id": related.id,
                    "content": related.content,
                    "importance": round(related.importance, 2),
                    "tags": related.tags
                })
        
        return results
    
    def get_memory_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific memory by its ID.
        
        Args:
            memory_id: Full or partial memory ID
            
        Returns:
            Memory details or None if not found
        """
        full_id = self._resolve_memory_id(memory_id)
        if not full_id:
            return None
        
        memory = self.memory.get_memory_by_id(full_id)
        if not memory:
            return None
        
        return {
            "id": memory.id,
            "content": memory.content,
            "importance": round(memory.importance, 2),
            "tags": memory.tags,
            "access_count": memory.access_count,
            "timestamp": memory.timestamp.isoformat(),
            "related_memories": memory.related_memories[:5]
        }
    
    def get_recent_memories(self, hours: int = 24, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get recently created memories.
        
        Args:
            hours: Look back this many hours (default 24)
            limit: Maximum number of memories to return
            
        Returns:
            List of recent memories
        """
        memories = self.memory.get_recent_memories(hours=hours, limit=limit)
        
        results = []
        for mem in memories:
            results.append({
                "id": mem.id,
                "content": mem.content,
                "importance": round(mem.importance, 2),
                "tags": mem.tags,
                "timestamp": mem.timestamp.isoformat()
            })
        
        return results
    
    def store_memory(self, content: str, importance: float = 0.7, 
                     tags: List[str] = None) -> Dict[str, Any]:
        """
        Store a new memory.
        
        Args:
            content: The memory content to store
            importance: Importance score 0.0-1.0 (default 0.7)
            tags: Optional list of tags for categorization
            
        Returns:
            Stored memory info with ID
        """
        memory_id = self.memory.store_memory(
            content=content,
            importance=importance,
            tags=tags or []
        )
        
        return {
            "id": memory_id,
            "content": content,
            "importance": importance,
            "stored": True
        }
    
    def _resolve_memory_id(self, partial_id: str) -> Optional[str]:
        """Resolve a partial memory ID to full ID"""
        if partial_id in self.memory.memories:
            return partial_id
        
        matches = [mid for mid in self.memory.memories if mid.startswith(partial_id)]
        if len(matches) == 1:
            return matches[0]
        
        return None
    
    def get_memory_count(self) -> int:
        """Get total number of stored memories"""
        return len(self.memory.memories)


# Tool schemas for Gemini function calling
MEMORY_TOOL_DECLARATIONS = [
    {
        "name": "search_memory",
        "description": "Search your memory for relevant information using semantic similarity. Use this to recall facts, preferences, past decisions, or anything you might have learned about the user.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query describing what you're looking for"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of memories to return (default 5)"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_related_memories",
        "description": "Get memories that are semantically related to a specific memory. Use this to follow connections and build deeper understanding.",
        "parameters": {
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "ID of the memory to find relations for (can be partial ID)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of related memories to return (default 3)"
                }
            },
            "required": ["memory_id"]
        }
    },
    {
        "name": "get_recent_memories",
        "description": "Get the most recently stored memories. Useful for understanding recent context or what was just discussed.",
        "parameters": {
            "type": "object",
            "properties": {
                "hours": {
                    "type": "integer",
                    "description": "Look back this many hours (default 24)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of memories to return (default 5)"
                }
            },
            "required": []
        }
    },
    {
        "name": "store_memory",
        "description": "Store an important piece of information in memory for future reference. Use sparingly - only for genuinely important facts, preferences, or decisions.",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The information to remember"
                },
                "importance": {
                    "type": "number",
                    "description": "How important is this? 0.0 (low) to 1.0 (critical). Default 0.7"
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional categorization tags"
                }
            },
            "required": ["content"]
        }
    }
]
