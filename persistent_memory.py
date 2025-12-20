"""
Vector-Based Persistent Memory System for Self-Improving AI

This module provides a persistent memory architecture that uses vector embeddings
as the primary storage mechanism. Knowledge is stored as high-dimensional vectors
that enable semantic similarity search and true understanding of relationships.

VECTOR DATABASE ARCHITECTURE:
- FAISS vector database for fast similarity search
- Sentence transformers for semantic encoding
- Memory consolidation for quality maintenance
- Automatic knowledge network creation
"""

import json
import os
import hashlib
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
import numpy as np
import faiss
import sentence_transformers
from sentence_transformers import SentenceTransformer


@dataclass
class MemoryEntry:
    """A single memory entry stored in the vector database"""
    id: str
    content: str
    timestamp: datetime
    importance: float  # 0.0 to 1.0
    tags: List[str]
    context: Dict[str, Any]  # Additional context information
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    related_memories: List[str] = field(default_factory=list)  # IDs of related memories
    embedding: List[float] = field(default_factory=list)  # Vector embedding (required for vector DB)
    faiss_index: int = -1  # Position in FAISS index


class VectorMemory:
    """
    A vector-based persistent memory system using FAISS as primary storage.

    This system stores all knowledge as high-dimensional vectors, enabling
    semantic similarity search and true understanding of knowledge relationships.
    No traditional database required - everything lives in the vector space.
    """

    def __init__(self, storage_path: str = "vector_memory", hardware_config: str = "hardware_config.json"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Load hardware configuration for optimizations
        self.hardware_config = self._load_hardware_config(hardware_config)
        self._configure_for_hardware()

        # Initialize sentence transformer for embeddings (hardware-optimized)
        model_name = self.hardware_config.get('recommendations', {}).get('embedding_model', 'all-MiniLM-L6-v2')
        self.encoder = SentenceTransformer(model_name, device='cpu')  # Force CPU usage
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()

        # Initialize FAISS index for vector storage and search
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity

        # Memory storage and mappings
        self.memories = {}  # id -> MemoryEntry
        self.id_to_idx = {}  # memory_id -> faiss_index
        self.idx_to_id = {}  # faiss_index -> memory_id

        # Metadata storage
        self.metadata_file = self.storage_path / "metadata.pkl"
        self.faiss_file = self.storage_path / "faiss_index.bin"

        # Load existing data
        self._load_persistent_data()

        print(f"🧠 Vector Memory initialized with {len(self.memories)} memories (CPU-optimized)")

    def _load_hardware_config(self, config_file: str) -> dict:
        """Load hardware configuration for optimization"""
        config_path = Path(config_file)
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️  Could not load hardware config: {e}")

        # Default CPU-only configuration
        return {
            "cpu": {"cores": 4, "has_avx2": True},
            "memory": {"total_gb": 6.0},
            "gpu": {"nvidia_gpu": False},
            "recommendations": {
                "use_gpu": False,
                "embedding_model": "all-MiniLM-L6-v2",
                "batch_size": 16,
                "max_memories": 5000
            }
        }

    def _configure_for_hardware(self):
        """Configure memory system based on hardware capabilities"""
        # Set memory limits based on available RAM
        total_ram = self.hardware_config.get('memory', {}).get('total_gb', 6.0)
        self.max_memories = min(
            self.hardware_config.get('recommendations', {}).get('max_memories', 10000),
            int(total_ram * 1000)  # Rough estimate: 1 memory per MB of RAM
        )

        # Set batch processing size
        self.batch_size = self.hardware_config.get('recommendations', {}).get('batch_size', 32)

        # Configure FAISS for CPU optimization
        if self.hardware_config.get('cpu', {}).get('has_avx2', False):
            # Enable AVX2 optimizations in FAISS
            import os
            os.environ['FAISS_OPT_LEVEL'] = 'avx2'

    def _load_persistent_data(self):
        """Load memories and FAISS index from persistent storage"""
        try:
            # Load metadata
            if self.metadata_file.exists():
                with open(self.metadata_file, 'rb') as f:
                    metadata = pickle.load(f)
                    self.memories = metadata.get('memories', {})
                    self.id_to_idx = metadata.get('id_to_idx', {})
                    self.idx_to_id = metadata.get('idx_to_id', {})

            # Load FAISS index
            if self.faiss_file.exists():
                self.faiss_index = faiss.read_index(str(self.faiss_file))

        except Exception as e:
            print(f"⚠️  Could not load existing data: {e}")
            print("Starting with empty vector memory...")

    def _save_persistent_data(self):
        """Save memories and FAISS index to persistent storage"""
        try:
            # Save metadata
            metadata = {
                'memories': self.memories,
                'id_to_idx': self.id_to_idx,
                'idx_to_id': self.idx_to_id
            }
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(metadata, f)

            # Save FAISS index
            faiss.write_index(self.faiss_index, str(self.faiss_file))

        except Exception as e:
            print(f"❌ Error saving data: {e}")

    def store_memory(self, content: str, importance: float = 0.5,
                    tags: List[str] = None, context: Dict[str, Any] = None) -> str:
        """
        Store a new memory in the vector database (hardware-optimized)

        Args:
            content: The memory content to encode and store
            importance: Importance score (0.0 to 1.0)
            tags: List of tags for categorization
            context: Additional context information

        Returns:
            Memory ID
        """
        if tags is None:
            tags = []
        if context is None:
            context = {}

        # Generate unique ID
        memory_id = hashlib.md5(f"{content}{datetime.now().isoformat()}".encode()).hexdigest()[:16]
        timestamp = datetime.now()

        # Generate vector embedding (CPU-optimized)
        embedding = self.encoder.encode(content, batch_size=self.batch_size).tolist()

        # Create memory entry
        memory = MemoryEntry(
            id=memory_id,
            content=content,
            timestamp=timestamp,
            importance=importance,
            tags=tags,
            context=context,
            embedding=embedding
        )

        # Add to FAISS index
        embedding_array = np.array([embedding], dtype=np.float32)
        faiss.normalize_L2(embedding_array)  # Normalize for cosine similarity
        self.faiss_index.add(embedding_array)

        # Update mappings
        faiss_idx = self.faiss_index.ntotal - 1
        memory.faiss_index = faiss_idx
        self.id_to_idx[memory_id] = faiss_idx
        self.idx_to_id[faiss_idx] = memory_id

        # Store in memory dictionary
        self.memories[memory_id] = memory

        # Check if we need to consolidate (memory management)
        if len(self.memories) > self.max_memories:
            self.consolidate_memories(int(self.max_memories * 0.8))

        # Find and link related memories
        self._link_related_memories(memory)

        # Persist changes (batch saves for efficiency)
        self._save_persistent_data()

        return memory_id

    def _link_related_memories(self, new_memory: MemoryEntry):
        """Find and link semantically related memories using vector similarity"""
        if self.faiss_index.ntotal <= 1:  # Only the new memory exists
            return

        related_ids = []

        # Search for similar memories in vector space
        query_embedding = np.array([new_memory.embedding], dtype=np.float32)
        faiss.normalize_L2(query_embedding)

        # Search for top-k similar memories (excluding self)
        k = min(15, self.faiss_index.ntotal - 1)  # Don't include the just-added memory
        scores, indices = self.faiss_index.search(query_embedding, k + 1)  # +1 to account for self

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # No more results
                break

            existing_id = self.idx_to_id.get(idx)
            if existing_id and existing_id != new_memory.id:
                existing_memory = self.memories.get(existing_id)
                if existing_memory and score > 0.4:  # Higher threshold for relatedness
                    related_ids.append(existing_id)
                    # Bidirectional linking
                    if new_memory.id not in existing_memory.related_memories:
                        existing_memory.related_memories.append(new_memory.id)

        new_memory.related_memories = related_ids

    def retrieve_memory(self, query: str, limit: int = 5,
                       min_importance: float = 0.0) -> List[MemoryEntry]:
        """
        Retrieve relevant memories using semantic vector similarity search

        Args:
            query: Natural language search query
            limit: Maximum number of results to return
            min_importance: Minimum importance threshold (0.0-1.0)

        Returns:
            List of relevant memory entries sorted by semantic relevance
        """
        if self.faiss_index.ntotal == 0:
            return []

        # Generate embedding for query
        query_embedding = self.encoder.encode(query)
        query_embedding = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_embedding)

        # Search vector space for similar memories
        k = min(limit * 3, self.faiss_index.ntotal)  # Get more candidates for filtering
        scores, indices = self.faiss_index.search(query_embedding, k)

        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                break

            memory_id = self.idx_to_id.get(idx)
            if memory_id:
                memory = self.memories.get(memory_id)
                if memory and memory.importance >= min_importance:
                    # Calculate final relevance score combining multiple factors
                    final_score = self._calculate_relevance_score(score, memory, query)
                    candidates.append((final_score, memory))

                    # Update access statistics
                    memory.access_count += 1
                    memory.last_accessed = datetime.now()

        # Sort by relevance and return top results
        candidates.sort(key=lambda x: x[0], reverse=True)

        # Persist access statistics
        self._save_persistent_data()

        return [memory for _, memory in candidates[:limit]]

    def _calculate_relevance_score(self, semantic_score: float, memory: MemoryEntry, query: str) -> float:
        """Calculate final relevance score combining semantic similarity with other factors"""
        # Tag matching bonus (check if query terms appear in tags)
        query_words = set(query.lower().split())
        memory_tag_words = set(word.lower() for tag in memory.tags for word in tag.split())
        tag_match = len(memory_tag_words & query_words) / len(query_words) if query_words else 0

        # Importance boost (higher importance = higher relevance)
        importance_boost = memory.importance * 0.2

        # Recency boost (newer memories get slight preference)
        days_old = (datetime.now() - memory.timestamp).days
        recency_boost = max(0, 1 - days_old / 365) * 0.05

        # Access frequency boost (frequently accessed memories are likely more relevant)
        access_boost = min(0.1, memory.access_count / 50) * 0.05

        # Content relevance bonus (check if query terms appear in content)
        content_words = set(memory.content.lower().split())
        content_match = len(content_words & query_words) / len(query_words) if query_words else 0

        # Combine factors: semantic similarity (60%) + other factors (40%)
        return (semantic_score * 0.6 +
                tag_match * 0.15 +
                content_match * 0.1 +
                importance_boost +
                recency_boost +
                access_boost)

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the vector memory system"""
        total_memories = len(self.memories)
        if total_memories == 0:
            return {
                "total_memories": 0,
                "average_importance": 0.0,
                "total_accesses": 0,
                "tag_distribution": {},
                "oldest_memory": None,
                "newest_memory": None,
                "vector_dimensions": self.embedding_dim,
                "faiss_index_size": self.faiss_index.ntotal
            }

        avg_importance = sum(m.importance for m in self.memories.values()) / total_memories
        total_accesses = sum(m.access_count for m in self.memories.values())

        # Tag distribution
        tag_counts = defaultdict(int)
        for memory in self.memories.values():
            for tag in memory.tags:
                tag_counts[tag] += 1

        return {
            "total_memories": total_memories,
            "average_importance": avg_importance,
            "total_accesses": total_accesses,
            "tag_distribution": dict(tag_counts),
            "oldest_memory": min((m.timestamp for m in self.memories.values())),
            "newest_memory": max((m.timestamp for m in self.memories.values())),
            "vector_dimensions": self.embedding_dim,
            "faiss_index_size": self.faiss_index.ntotal
        }

    def consolidate_memories(self, max_memories: int = 1000):
        """
        Consolidate memories by importance, recency, and access frequency.
        Maintains quality by keeping the most valuable memories in vector space.
        """
        if len(self.memories) <= max_memories:
            return

        # Score memories by combined metrics
        scored_memories = []
        for memory in self.memories.values():
            days_old = (datetime.now() - memory.timestamp).days
            recency_score = max(0, 1 - days_old / 365)  # Recent memories score higher
            access_score = min(1.0, memory.access_count / 50)  # Frequently accessed memories

            # Combined score: importance (50%) + recency (30%) + access (20%)
            total_score = (memory.importance * 0.5 +
                          recency_score * 0.3 +
                          access_score * 0.2)
            scored_memories.append((total_score, memory))

        # Sort by score (highest first) and keep top memories
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        keep_memories = scored_memories[:max_memories]

        # Rebuild FAISS index and memory storage with only the best memories
        self._rebuild_index_with_memories([memory for _, memory in keep_memories])

        print(f"🧹 Consolidated memories: kept {len(keep_memories)} of {len(scored_memories)}")

    def _rebuild_index_with_memories(self, keep_memories: List[MemoryEntry]):
        """Rebuild FAISS index and memory storage with selected memories"""
        # Create new FAISS index
        new_faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        new_memories = {}
        new_id_to_idx = {}
        new_idx_to_id = {}

        # Add memories back to index
        embeddings = []
        for memory in keep_memories:
            embeddings.append(memory.embedding)
            new_memories[memory.id] = memory

        if embeddings:
            embeddings_array = np.array(embeddings, dtype=np.float32)
            faiss.normalize_L2(embeddings_array)
            new_faiss_index.add(embeddings_array)

            # Update mappings
            for i, memory in enumerate(keep_memories):
                memory.faiss_index = i
                new_id_to_idx[memory.id] = i
                new_idx_to_id[i] = memory.id

        # Replace current structures
        self.faiss_index = new_faiss_index
        self.memories = new_memories
        self.id_to_idx = new_id_to_idx
        self.idx_to_id = new_idx_to_id

        # Save changes
        self._save_persistent_data()


def main():
    """Demo of the vector-based memory system"""
    memory = VectorMemory()

    # Store some example memories
    print("🧠 Storing example memories in vector space...")
    memory.store_memory(
        "Python list comprehensions are more efficient than traditional loops",
        importance=0.8,
        tags=["python", "programming", "efficiency"]
    )

    memory.store_memory(
        "Neural networks learn better with proper data normalization",
        importance=0.9,
        tags=["machine_learning", "neural_networks", "data_processing"]
    )

    memory.store_memory(
        "Always validate user input to prevent security vulnerabilities",
        importance=0.95,
        tags=["security", "programming", "best_practices"]
    )

    memory.store_memory(
        "Machine learning models need feature scaling for optimal performance",
        importance=0.85,
        tags=["machine_learning", "preprocessing", "data_science"]
    )

    # Retrieve relevant memories using semantic search
    print("\n🔍 Retrieving memories about 'programming optimization'...")
    results = memory.retrieve_memory("programming optimization", limit=3)
    for result in results:
        print(f"  → {result.content[:70]}... (importance: {result.importance:.2f})")

    # Test semantic similarity (should find related concepts)
    print("\n🔍 Retrieving memories about 'data preparation'...")
    results = memory.retrieve_memory("data preparation for ML", limit=3)
    for result in results:
        print(f"  → {result.content[:70]}... (importance: {result.importance:.2f})")

    # Get statistics
    print("\n📊 Vector Memory Statistics:")
    stats = memory.get_memory_stats()
    print(f"  Total memories: {stats['total_memories']}")
    print(f"  Average importance: {stats['average_importance']:.2f}")
    print(".1f")
    print(f"  Top tags: {stats['tag_distribution']}")


if __name__ == "__main__":
    main()
