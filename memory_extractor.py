"""
Memory Extractor - Async Insight Extraction Agent

Extracts structured memories from conversation exchanges using a cheap/fast LLM.
Runs asynchronously to avoid blocking the main conversation flow.

This is the "memory agent" that intelligently decides what to remember.
"""

import json
import os
import threading
import queue
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Will be set when Gemini client is available
_gemini_available = False
_genai_client = None
try:
    from google import genai
    _gemini_available = True
except ImportError:
    pass


@dataclass
class ExtractedMemory:
    """A memory extracted from conversation"""
    content: str
    importance: float
    memory_type: str  # fact, preference, decision, insight
    tags: List[str]
    confidence: float
    source_context: str


EXTRACTION_PROMPT = """Analyze this conversation exchange and extract any information worth remembering for future conversations.

User message:
{user_message}

Assistant response:
{assistant_response}

Extract memories as JSON. Only include genuinely important/reusable information.
Skip: greetings, acknowledgments, transient details, obvious facts.

Focus on:
- User preferences (how they like things done)
- Important facts about them or their projects
- Decisions made
- Technical insights or solutions discovered
- Recurring patterns or context

Return JSON only, no markdown:
{{
  "should_remember": true/false,
  "memories": [
    {{
      "content": "concise statement of what to remember",
      "type": "preference|fact|decision|insight",
      "importance": 0.0-1.0,
      "tags": ["relevant", "tags"],
      "confidence": 0.0-1.0
    }}
  ],
  "reasoning": "why these are worth remembering (or why nothing is)"
}}

If nothing is worth remembering, return should_remember: false with empty memories array."""


class MemoryExtractor:
    """
    Extracts memories from conversation exchanges asynchronously.
    
    Uses a cheap LLM (Gemini Flash) to analyze conversations
    and extract structured information worth remembering.
    """
    
    def __init__(self, memory_system=None, model: str = "gemini-2.0-flash-lite"):
        """
        Initialize the memory extractor.
        
        Args:
            memory_system: VectorMemory instance to store extracted memories
            model: Model to use for extraction (default: Gemini Flash for cost efficiency)
        """
        self.memory_system = memory_system
        self.model_name = model
        self.client = None
        
        # Extraction queue for async processing
        self.extraction_queue = queue.Queue()
        self.worker_thread = None
        self.running = False
        
        # Statistics
        self.stats = {
            "exchanges_processed": 0,
            "memories_extracted": 0,
            "extraction_errors": 0,
            "last_extraction": None
        }
        
        # Initialize client if available
        if _gemini_available:
            try:
                # Try to get API key from environment
                api_key = os.environ.get("GEMINI_API_KEY")
                if api_key:
                    self.client = genai.Client(api_key=api_key)
                    print(f"🧠 Memory Extractor initialized with Gemini ({model})")
                else:
                    print("⚠️  GEMINI_API_KEY not found in environment")
                    print("   Memory extraction will use fallback heuristics")
            except Exception as e:
                print(f"⚠️  Could not initialize Gemini client: {e}")
                print("   Memory extraction will use fallback heuristics")
        else:
            print("⚠️  google-genai SDK not installed. Using fallback extraction.")
            print("   Install with: pip install google-genai")
    
    def start_worker(self):
        """Start the background extraction worker"""
        if self.worker_thread is not None and self.worker_thread.is_alive():
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._extraction_worker, daemon=True)
        self.worker_thread.start()
        print("🔄 Memory extraction worker started")
    
    def stop_worker(self):
        """Stop the background extraction worker"""
        self.running = False
        if self.worker_thread:
            # Put a sentinel to wake up the worker
            self.extraction_queue.put(None)
            self.worker_thread.join(timeout=2.0)
    
    def _extraction_worker(self):
        """Background worker that processes extraction queue"""
        while self.running:
            try:
                item = self.extraction_queue.get(timeout=1.0)
                
                if item is None:  # Sentinel to stop
                    break
                
                user_msg, assistant_msg, callback = item
                
                try:
                    memories = self._extract_memories_sync(user_msg, assistant_msg)
                    
                    # Store extracted memories
                    for memory in memories:
                        self._store_memory(memory)
                    
                    if callback:
                        callback(memories)
                        
                except Exception as e:
                    self.stats["extraction_errors"] += 1
                    print(f"⚠️  Extraction error: {e}")
                
                self.extraction_queue.task_done()
                
            except queue.Empty:
                continue
    
    def extract_async(self, user_message: str, assistant_response: str, 
                     callback: callable = None):
        """
        Queue an exchange for async memory extraction.
        
        Args:
            user_message: The user's message
            assistant_response: The assistant's response
            callback: Optional callback with extracted memories
        """
        # Start worker if not running
        if not self.running:
            self.start_worker()
        
        # Queue for extraction
        self.extraction_queue.put((user_message, assistant_response, callback))
    
    def _extract_memories_sync(self, user_message: str, 
                               assistant_response: str) -> List[ExtractedMemory]:
        """
        Synchronously extract memories from an exchange.
        
        Returns:
            List of ExtractedMemory objects
        """
        self.stats["exchanges_processed"] += 1
        self.stats["last_extraction"] = datetime.now()
        
        # Skip very short exchanges
        if len(user_message) < 20 and len(assistant_response) < 50:
            return []
        
        # Use LLM extraction if available
        if self.client:
            return self._extract_with_llm(user_message, assistant_response)
        else:
            return self._extract_with_heuristics(user_message, assistant_response)
    
    def _extract_with_llm(self, user_message: str, 
                          assistant_response: str) -> List[ExtractedMemory]:
        """Extract memories using Gemini LLM"""
        try:
            prompt = EXTRACTION_PROMPT.format(
                user_message=user_message[:2000],  # Limit length
                assistant_response=assistant_response[:2000]
            )
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            response_text = response.text
            
            # Parse response
            try:
                # Try to find JSON in the response
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    data = json.loads(json_str)
                else:
                    return []
            except json.JSONDecodeError:
                return []
            
            if not data.get("should_remember", False):
                return []
            
            memories = []
            for mem_data in data.get("memories", []):
                memory = ExtractedMemory(
                    content=mem_data.get("content", ""),
                    importance=float(mem_data.get("importance", 0.5)),
                    memory_type=mem_data.get("type", "insight"),
                    tags=mem_data.get("tags", []),
                    confidence=float(mem_data.get("confidence", 0.7)),
                    source_context="conversation"
                )
                if memory.content:
                    memories.append(memory)
            
            self.stats["memories_extracted"] += len(memories)
            return memories
            
        except Exception as e:
            print(f"⚠️  LLM extraction failed: {e}")
            return self._extract_with_heuristics(user_message, assistant_response)
    
    def _extract_with_heuristics(self, user_message: str, 
                                  assistant_response: str) -> List[ExtractedMemory]:
        """Fallback heuristic-based extraction when LLM unavailable"""
        memories = []
        
        combined_text = f"{user_message} {assistant_response}".lower()
        
        # Preference detection
        preference_indicators = [
            "i prefer", "i like", "i always", "i usually", "i want",
            "please always", "please don't", "i don't like"
        ]
        
        for indicator in preference_indicators:
            if indicator in user_message.lower():
                # Extract the sentence containing the preference
                sentences = user_message.split('.')
                for sentence in sentences:
                    if indicator in sentence.lower():
                        memory = ExtractedMemory(
                            content=sentence.strip(),
                            importance=0.7,
                            memory_type="preference",
                            tags=["preference", "user"],
                            confidence=0.6,
                            source_context="heuristic_extraction"
                        )
                        memories.append(memory)
                        break
        
        # Decision detection
        decision_indicators = [
            "decided to", "going to", "will use", "chose to",
            "let's go with", "we'll use"
        ]
        
        for indicator in decision_indicators:
            if indicator in combined_text:
                for sentence in (user_message + " " + assistant_response).split('.'):
                    if indicator in sentence.lower():
                        memory = ExtractedMemory(
                            content=f"Decision: {sentence.strip()}",
                            importance=0.6,
                            memory_type="decision",
                            tags=["decision"],
                            confidence=0.5,
                            source_context="heuristic_extraction"
                        )
                        memories.append(memory)
                        break
        
        self.stats["memories_extracted"] += len(memories)
        return memories
    
    def _store_memory(self, memory: ExtractedMemory):
        """Store an extracted memory in the memory system"""
        if not self.memory_system:
            return
        
        # Adjust importance by confidence
        adjusted_importance = memory.importance * memory.confidence
        
        # Add memory type to tags
        tags = memory.tags.copy()
        if memory.memory_type not in tags:
            tags.append(memory.memory_type)
        
        try:
            memory_id = self.memory_system.store_memory(
                content=memory.content,
                importance=adjusted_importance,
                tags=tags,
                context={
                    "source": memory.source_context,
                    "type": memory.memory_type,
                    "confidence": memory.confidence,
                    "extracted_at": datetime.now().isoformat()
                }
            )
            print(f"💾 Extracted memory stored: {memory.content[:50]}...")
        except Exception as e:
            print(f"⚠️  Failed to store memory: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get extraction statistics"""
        return {
            **self.stats,
            "queue_size": self.extraction_queue.qsize(),
            "worker_running": self.running
        }


def create_memory_extractor(memory_system=None) -> MemoryExtractor:
    """Factory function to create a memory extractor"""
    return MemoryExtractor(memory_system=memory_system)


if __name__ == "__main__":
    # Test the extractor
    print("Testing Memory Extractor")
    print("=" * 40)
    
    extractor = MemoryExtractor()
    
    # Test extraction
    test_user = "I prefer dark mode in all my applications and I'm working on a project called ProjectX"
    test_assistant = "I'll keep that preference in mind. For ProjectX, let me know what you need help with."
    
    memories = extractor._extract_memories_sync(test_user, test_assistant)
    
    print(f"\nExtracted {len(memories)} memories:")
    for mem in memories:
        print(f"  - [{mem.memory_type}] {mem.content}")
        print(f"    Importance: {mem.importance}, Confidence: {mem.confidence}")
    
    print(f"\nStats: {extractor.get_stats()}")
