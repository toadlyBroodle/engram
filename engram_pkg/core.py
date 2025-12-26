"""
Engram - Vector-Based Memory System

This module provides the core vector memory architecture for Engram.
Knowledge is stored as high-dimensional vectors that enable semantic
similarity search and true understanding of relationships.

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
    Engram's vector-based memory system using FAISS as primary storage.

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

        # Batch saving system for improved performance
        self.pending_changes = False
        self.batch_save_interval = 10  # Save every 10 operations
        self.operation_count = 0

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

        # Reset pending changes flag after successful save
        self.pending_changes = False
        self.operation_count = 0

    def _check_memory_similarity(self, new_content: str, new_embedding: List[float],
                                similarity_threshold: float = 0.85) -> Optional[str]:
        """
        Check if a similar memory already exists

        Args:
            new_content: New memory content
            new_embedding: New memory embedding
            similarity_threshold: Threshold for considering memories similar

        Returns:
            Memory ID if similar memory found, None otherwise
        """
        if self.faiss_index.ntotal == 0:
            return None

        # Search for similar memories
        query_embedding = np.array([new_embedding], dtype=np.float32)
        faiss.normalize_L2(query_embedding)

        k = min(5, self.faiss_index.ntotal)  # Check top 5 most similar
        scores, indices = self.faiss_index.search(query_embedding, k)

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                break

            existing_id = self.idx_to_id.get(idx)
            if existing_id and score > similarity_threshold:
                existing_memory = self.memories.get(existing_id)
                if existing_memory:
                    # Additional content-based similarity check
                    content_similarity = self._calculate_content_similarity(new_content, existing_memory.content)
                    if content_similarity > 0.7:  # Content is also similar
                        return existing_id

        return None

    def _calculate_content_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)

    def store_memory(self, content: str, importance: float = 0.5,
                    tags: List[str] = None, context: Dict[str, Any] = None,
                    allow_duplicates: bool = False) -> str:
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

        # Check for duplicate memories unless explicitly allowed
        if not allow_duplicates:
            duplicate_id = self._check_memory_similarity(content, embedding)
            if duplicate_id:
                # Update existing memory instead of creating duplicate
                existing_memory = self.memories[duplicate_id]
                # Update importance if new memory is more important
                if importance > existing_memory.importance:
                    existing_memory.importance = importance
                # Merge tags
                existing_memory.tags = list(set(existing_memory.tags + tags))
                # Update context if provided
                if context:
                    existing_memory.context.update(context)
                # Update timestamp and access count
                existing_memory.last_accessed = timestamp
                existing_memory.access_count += 1

                # Mark changes as pending
                self.pending_changes = True
                self.operation_count += 1

                return duplicate_id

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

        # Mark changes as pending (batch saves for efficiency)
        self.pending_changes = True
        self.operation_count += 1

        # Auto-save periodically or on significant operations
        if self.operation_count >= self.batch_save_interval:
            self._save_persistent_data()

        return memory_id

    def get_memory_by_id(self, memory_id: str) -> Optional[MemoryEntry]:
        """Retrieve a specific memory by ID for context integration"""
        return self.memories.get(memory_id)

    def get_memories_by_tags(self, tags: List[str], limit: int = 10) -> List[MemoryEntry]:
        """Retrieve memories that match any of the given tags"""
        matching_memories = []
        tag_set = set(tags)

        for memory in self.memories.values():
            if tag_set & set(memory.tags):  # Intersection of tag sets
                matching_memories.append(memory)

        # Sort by importance and recency
        matching_memories.sort(key=lambda m: (m.importance, m.timestamp), reverse=True)
        return matching_memories[:limit]

    def get_recent_memories(self, hours: int = 24, limit: int = 10) -> List[MemoryEntry]:
        """Get recently created memories for context awareness"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        recent_memories = [
            memory for memory in self.memories.values()
            if memory.timestamp >= cutoff_time
        ]

        # Sort by timestamp (most recent first)
        recent_memories.sort(key=lambda m: m.timestamp, reverse=True)
        return recent_memories[:limit]

    def update_memory_usage(self, memory_id: str):
        """Update usage statistics for a memory (called by context integrator)"""
        if memory_id in self.memories:
            memory = self.memories[memory_id]
            memory.access_count += 1
            memory.last_accessed = datetime.now()
            # Mark as pending save (batch saves for efficiency)
            self.pending_changes = True
            self.operation_count += 1

            # Auto-save periodically
            if self.operation_count >= self.batch_save_interval:
                self._save_persistent_data()

    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory from the system

        Args:
            memory_id: ID of the memory to delete

        Returns:
            True if memory was deleted, False if not found
        """
        if memory_id not in self.memories:
            return False

        # Get the FAISS index position
        faiss_idx = self.memories[memory_id].faiss_index

        # Remove from memories dictionary
        del self.memories[memory_id]

        # Remove from ID mappings
        if faiss_idx in self.idx_to_id:
            del self.idx_to_id[faiss_idx]
        if memory_id in self.id_to_idx:
            del self.id_to_idx[memory_id]

        # Note: FAISS index doesn't support easy deletion, so we'll rebuild it
        # when saving. For now, mark that we have pending changes.
        self.pending_changes = True

        print(f"🗑️  Memory {memory_id} deleted")
        return True

    def get_topic_relevant_memories(self, topic_keywords: List[str], limit: int = 5) -> List[MemoryEntry]:
        """Get memories relevant to specific topic keywords"""
        relevant_memories = []
        keyword_set = set(word.lower() for word in topic_keywords)

        for memory in self.memories.values():
            # Check content for keywords
            content_words = set(memory.content.lower().split())
            content_matches = len(keyword_set & content_words)

            # Check tags for keywords
            tag_matches = len(keyword_set & set(tag.lower() for tag in memory.tags))

            # Score based on matches and importance
            if content_matches > 0 or tag_matches > 0:
                relevance_score = (content_matches * 2 + tag_matches) * memory.importance
                relevant_memories.append((relevance_score, memory))

        # Sort by relevance score
        relevant_memories.sort(key=lambda x: x[0], reverse=True)
        return [memory for _, memory in relevant_memories[:limit]]

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

        # Mark access statistics as pending save (batch saves for efficiency)
        if candidates:  # Only mark if there were actual retrievals
            self.pending_changes = True
            self.operation_count += 1

            # Auto-save periodically
            if self.operation_count >= self.batch_save_interval:
                self._save_persistent_data()

        return [memory for _, memory in candidates[:limit]]

    def _calculate_relevance_score(self, semantic_score: float, memory: MemoryEntry, query: str) -> float:
        """Calculate final relevance score combining semantic similarity with other factors using enhanced algorithm"""
        query_words = set(query.lower().split()) if query else set()

        # Enhanced tag matching with partial word matching
        tag_match_score = self._calculate_tag_relevance(query_words, memory.tags)

        # Enhanced content matching with TF-IDF style weighting
        content_match_score = self._calculate_content_relevance(query_words, memory.content)

        # Importance with diminishing returns (logarithmic scaling)
        importance_boost = min(0.3, memory.importance * 0.4 + (memory.importance ** 0.5) * 0.1)

        # Temporal decay with configurable half-life (30 days)
        temporal_score = self._calculate_temporal_relevance(memory.timestamp, half_life_days=30)

        # Access frequency with exponential decay (recent accesses matter more)
        access_score = self._calculate_access_relevance(memory.access_count, memory.last_accessed)

        # Context relevance (if memory has context matching query)
        context_score = self._calculate_context_relevance(query_words, memory.context)

        # Quality score based on memory completeness and usage patterns
        quality_score = self._calculate_memory_quality(memory)

        # Weighted combination with improved balance
        # Semantic similarity (40%) - most important for meaning
        # Content/Tag matching (25%) - direct textual relevance
        # Importance (15%) - user-assigned significance
        # Temporal/Access (15%) - usage patterns and freshness
        # Context/Quality (5%) - additional factors
        final_score = (
            semantic_score * 0.40 +
            (tag_match_score + content_match_score) * 0.25 +
            importance_boost * 0.15 +
            (temporal_score + access_score) * 0.15 +
            (context_score + quality_score) * 0.05
        )

        return final_score

    def _calculate_tag_relevance(self, query_words: set, tags: List[str]) -> float:
        """Calculate relevance based on tag matching with partial matches"""
        if not query_words or not tags:
            return 0.0

        total_score = 0.0
        tag_words = set()
        for tag in tags:
            tag_words.update(tag.lower().split())

        # Exact matches
        exact_matches = len(query_words & tag_words)
        # Partial matches (word contains query term or vice versa)
        partial_matches = 0
        for query_word in query_words:
            for tag_word in tag_words:
                if query_word in tag_word or tag_word in query_word:
                    partial_matches += 0.5

        total_matches = exact_matches + partial_matches
        return min(1.0, total_matches / len(query_words))

    def _calculate_content_relevance(self, query_words: set, content: str) -> float:
        """Calculate content relevance with position weighting"""
        if not query_words or not content:
            return 0.0

        content_lower = content.lower()
        content_words = content_lower.split()

        # Exact word matches
        exact_matches = len(query_words & set(content_words))

        # Proximity bonus (query terms appearing close together)
        proximity_score = 0.0
        if len(query_words) > 1:
            positions = []
            for word in query_words:
                if word in content_words:
                    positions.extend([i for i, w in enumerate(content_words) if w == word])

            if len(positions) > 1:
                # Calculate average distance between query terms
                positions.sort()
                avg_distance = sum(positions[i+1] - positions[i] for i in range(len(positions)-1)) / (len(positions)-1)
                proximity_score = max(0, 1 - avg_distance / 20) * 0.2  # Bonus for close terms

        # Position bonus (terms appearing early in content get higher weight)
        position_score = 0.0
        for word in query_words:
            try:
                first_pos = content_words.index(word)
                position_score += max(0, 1 - first_pos / len(content_words)) * 0.1
            except ValueError:
                continue

        total_score = exact_matches / len(query_words) + proximity_score + position_score
        return min(1.0, total_score)

    def _calculate_temporal_relevance(self, timestamp: datetime, half_life_days: int = 30) -> float:
        """Calculate temporal relevance with exponential decay"""
        days_old = (datetime.now() - timestamp).days
        if days_old <= 0:
            return 0.15  # Maximum boost for very recent memories

        # Exponential decay: score = 0.15 * (0.5)^(days_old/half_life)
        decay_factor = 0.5 ** (days_old / half_life_days)
        return 0.15 * decay_factor

    def _calculate_access_relevance(self, access_count: int, last_accessed: Optional[datetime]) -> float:
        """Calculate relevance based on access patterns"""
        if access_count == 0:
            return 0.0

        # Base access score (logarithmic to prevent runaway growth)
        access_base = min(0.1, (access_count ** 0.5) / 10)

        # Recency of last access (more recent = higher weight)
        recency_boost = 0.0
        if last_accessed:
            days_since_access = (datetime.now() - last_accessed).days
            recency_boost = max(0, 1 - days_since_access / 30) * 0.05

        return access_base + recency_boost

    def _calculate_context_relevance(self, query_words: set, context: Dict[str, Any]) -> float:
        """Calculate relevance based on context information"""
        if not context or not query_words:
            return 0.0

        context_text = json.dumps(context, default=str).lower()
        context_words = set(context_text.split())

        matches = len(query_words & context_words)
        return min(0.1, matches / len(query_words) * 0.1)

    def _calculate_memory_quality(self, memory: MemoryEntry) -> float:
        """Calculate memory quality score based on completeness and usage patterns"""
        quality_score = 0.0

        # Completeness bonus (memories with more information are valued higher)
        if memory.tags:
            quality_score += 0.02 * min(len(memory.tags), 5)
        if memory.context:
            quality_score += 0.02 * min(len(memory.context), 5)
        if len(memory.content) > 50:  # Substantial content
            quality_score += 0.01

        # Usage pattern bonus (well-used memories are likely higher quality)
        if memory.access_count > 10:
            quality_score += 0.02

        # Age bonus (surviving memories are likely more valuable)
        days_old = (datetime.now() - memory.timestamp).days
        if days_old > 30:  # Survived initial consolidation
            quality_score += 0.01

        return min(0.05, quality_score)

    def get_memory_health_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive memory health report with quality metrics and recommendations

        Returns:
            Dictionary containing health metrics and maintenance recommendations
        """
        if not self.memories:
            return {
                "status": "empty",
                "recommendations": ["Add initial memories to start building knowledge base"],
                "quality_score": 0.0
            }

        # Calculate overall quality metrics
        quality_scores = []
        access_patterns = []
        temporal_distribution = []
        tag_coverage = defaultdict(int)

        now = datetime.now()
        total_memories = len(self.memories)

        for memory in self.memories.values():
            # Individual memory quality
            quality = self._calculate_memory_quality(memory)
            quality_scores.append(quality)

            # Access patterns
            access_patterns.append(memory.access_count)

            # Temporal distribution (age in days)
            age_days = (now - memory.timestamp).days
            temporal_distribution.append(age_days)

            # Tag coverage
            for tag in memory.tags:
                tag_coverage[tag] += 1

        # Overall health metrics
        avg_quality = sum(quality_scores) / len(quality_scores)
        avg_access = sum(access_patterns) / len(access_patterns)
        avg_age = sum(temporal_distribution) / len(temporal_distribution)

        # Health indicators
        quality_health = avg_quality * 100  # 0-100 scale
        access_health = min(100, avg_access * 10)  # Scale access frequency
        freshness_health = max(0, 100 - (avg_age / 3))  # Prefer memories < 3 months old

        # Diversity score (unique tags vs total memories)
        unique_tags = len(tag_coverage)
        tag_diversity = min(100, (unique_tags / total_memories) * 200)

        # Overall health score (weighted average)
        overall_health = (
            quality_health * 0.3 +
            access_health * 0.3 +
            freshness_health * 0.25 +
            tag_diversity * 0.15
        )

        # Generate recommendations
        recommendations = self._generate_health_recommendations(
            overall_health, quality_health, access_health, freshness_health, tag_diversity,
            temporal_distribution, access_patterns
        )

        return {
            "status": self._get_health_status(overall_health),
            "overall_health_score": round(overall_health, 1),
            "metrics": {
                "quality_score": round(avg_quality, 3),
                "average_access_count": round(avg_access, 2),
                "average_age_days": round(avg_age, 1),
                "unique_tags": unique_tags,
                "total_memories": total_memories,
                "tag_diversity_ratio": round(unique_tags / total_memories, 2),
                "quality_health": round(quality_health, 1),
                "access_health": round(access_health, 1),
                "freshness_health": round(freshness_health, 1),
                "tag_diversity": round(tag_diversity, 1)
            },
            "distribution": {
                "oldest_memory_days": max(temporal_distribution),
                "newest_memory_days": min(temporal_distribution),
                "most_accessed_count": max(access_patterns),
                "least_accessed_count": min(access_patterns),
                "top_tags": sorted(tag_coverage.items(), key=lambda x: x[1], reverse=True)[:5]
            },
            "recommendations": recommendations,
            "maintenance_needed": overall_health < 70.0
        }

    def _get_health_status(self, health_score: float) -> str:
        """Convert health score to status string"""
        if health_score >= 85:
            return "excellent"
        elif health_score >= 70:
            return "good"
        elif health_score >= 50:
            return "fair"
        elif health_score >= 30:
            return "poor"
        else:
            return "critical"

    def _generate_health_recommendations(self, overall_health: float, quality_health: float,
                                       access_health: float, freshness_health: float,
                                       tag_diversity: float, age_distribution: List[int],
                                       access_distribution: List[int]) -> List[str]:
        """Generate specific recommendations based on health metrics"""
        recommendations = []

        # Overall health recommendations
        if overall_health < 50:
            recommendations.append("⚠️  Critical: Memory system health is poor - consider full system maintenance")
        elif overall_health < 70:
            recommendations.append("📉 Memory health needs attention - schedule maintenance soon")

        # Quality recommendations
        if quality_health < 60:
            recommendations.append("🔍 Add more detailed context and tags to improve memory quality")
        elif quality_health > 90:
            recommendations.append("✅ Memory quality is excellent - continue current practices")

        # Access pattern recommendations
        if access_health < 40:
            recommendations.append("🎯 Many memories are unused - review and consolidate low-value memories")
        elif access_health > 80:
            recommendations.append("📈 Excellent memory utilization - system is being actively used")

        # Freshness recommendations
        if freshness_health < 50:
            recommendations.append("🕒 Memory base is aging - add fresh memories to maintain relevance")
        elif freshness_health > 80:
            recommendations.append("🆕 Memory freshness is good - recent content is being added regularly")

        # Diversity recommendations
        if tag_diversity < 30:
            recommendations.append("🏷️  Improve tag diversity - use more varied categorization")
        elif tag_diversity > 70:
            recommendations.append("🏷️  Good tag diversity - maintain varied categorization practices")

        # Specific maintenance recommendations
        old_memories = sum(1 for age in age_distribution if age > 365)  # Over 1 year old
        if old_memories > len(age_distribution) * 0.3:
            recommendations.append(f"🧹 Consider consolidating {old_memories} memories older than 1 year")

        unused_memories = sum(1 for access in access_distribution if access == 0)
        if unused_memories > len(access_distribution) * 0.4:
            recommendations.append(f"🗑️  Review {unused_memories} never-accessed memories for removal")

        # Performance recommendations
        if len(self.memories) > self.max_memories * 0.8:
            recommendations.append("⚡ Approaching memory capacity - consider consolidation or expansion")

        return recommendations

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

    def force_save(self):
        """Force immediate save of all pending changes"""
        if self.pending_changes:
            self._save_persistent_data()
            self.pending_changes = False

    def __del__(self):
        """Ensure changes are saved when the object is destroyed"""
        try:
            if hasattr(self, 'pending_changes') and self.pending_changes:
                self._save_persistent_data()
        except:
            pass  # Ignore errors during cleanup


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


def test_memory_health():
    """Test memory health monitoring functionality"""
    memory = VectorMemory()

    print("🩺 **Memory Health Report**")
    print("=" * 50)

    health_report = memory.get_memory_health_report()

    print(f"Status: {health_report['status'].upper()}")
    print(f"Overall Health Score: {health_report['overall_health_score']}/100")
    print()

    print("📊 **Key Metrics:**")
    metrics = health_report['metrics']
    print(f"  Quality Score: {metrics['quality_score']:.3f}")
    print(f"  Average Access Count: {metrics['average_access_count']}")
    print(f"  Average Age: {metrics['average_age_days']} days")
    print(f"  Unique Tags: {metrics['unique_tags']}")
    print(f"  Tag Diversity Ratio: {metrics['tag_diversity_ratio']}")
    print()

    print("📈 **Health Breakdown:**")
    print(f"  Quality Health: {metrics['quality_health']}/100")
    print(f"  Access Health: {metrics['access_health']}/100")
    print(f"  Freshness Health: {metrics['freshness_health']}/100")
    print(f"  Tag Diversity: {metrics['tag_diversity']}/100")
    print()

    print("🏷️ **Top Tags:**")
    for tag, count in health_report['distribution']['top_tags']:
        print(f"  {tag}: {count}")
    print()

    print("💡 **Recommendations:**")
    for rec in health_report['recommendations']:
        print(f"  {rec}")
    print()

    maintenance_needed = "YES" if health_report['maintenance_needed'] else "NO"
    print(f"🔧 Maintenance Needed: {maintenance_needed}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "health":
        test_memory_health()
    else:
        main()
