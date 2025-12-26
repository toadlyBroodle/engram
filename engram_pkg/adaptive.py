"""
Adaptive Memory Learning System

Learns from usage patterns to improve memory relevance scoring over time.
Uses reinforcement learning principles to optimize memory integration.

Key Features:
- Usage pattern analysis and learning
- Adaptive relevance scoring based on success metrics
- Memory importance evolution
- Contextual pattern recognition
- Performance tracking and optimization
"""

import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
import numpy as np
from pathlib import Path


class MemoryPerformanceTracker:
    """Tracks how well memories perform in different contexts"""

    def __init__(self, tracking_file: str = "memory_performance.pkl"):
        self.tracking_file = Path(tracking_file)
        self.performance_data = self._load_performance_data()

        # Performance metrics
        self.usage_counts = defaultdict(int)  # memory_id -> total_uses
        self.success_rates = defaultdict(list)  # memory_id -> list of success scores
        self.context_patterns = defaultdict(Counter)  # memory_id -> context_type frequencies
        self.temporal_patterns = defaultdict(list)  # memory_id -> usage timestamps

    def _load_performance_data(self) -> Dict[str, Any]:
        """Load existing performance data"""
        if self.tracking_file.exists():
            try:
                with open(self.tracking_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"⚠️  Could not load performance data: {e}")

        return {
            "usage_counts": {},
            "success_rates": {},
            "context_patterns": {},
            "temporal_patterns": {},
            "last_updated": datetime.now()
        }

    def _save_performance_data(self):
        """Save performance data to disk"""
        data = {
            "usage_counts": dict(self.usage_counts),
            "success_rates": dict(self.success_rates),
            "context_patterns": {k: dict(v) for k, v in self.context_patterns.items()},
            "temporal_patterns": dict(self.temporal_patterns),
            "last_updated": datetime.now()
        }

        try:
            with open(self.tracking_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"❌ Error saving performance data: {e}")

    def record_memory_usage(self, memory_id: str, context_type: str = "general",
                          success_score: float = 0.5, query: str = ""):
        """
        Record memory usage for learning

        Args:
            memory_id: ID of the memory used
            context_type: Type of context (conversation, task, etc.)
            success_score: How successful the memory was (0.0-1.0)
            query: The query/context that triggered this memory usage
        """
        # Update usage counts
        self.usage_counts[memory_id] += 1

        # Record success rate
        self.success_rates[memory_id].append(success_score)

        # Keep only last 50 success scores for memory efficiency
        if len(self.success_rates[memory_id]) > 50:
            self.success_rates[memory_id] = self.success_rates[memory_id][-50:]

        # Record context patterns
        self.context_patterns[memory_id][context_type] += 1

        # Record temporal pattern
        self.temporal_patterns[memory_id].append(datetime.now())

        # Keep only recent temporal data (last 100 uses)
        if len(self.temporal_patterns[memory_id]) > 100:
            self.temporal_patterns[memory_id] = self.temporal_patterns[memory_id][-100:]

        # Periodic save
        if sum(self.usage_counts.values()) % 10 == 0:  # Save every 10 uses
            self._save_performance_data()

    def get_memory_performance_score(self, memory_id: str) -> float:
        """Calculate overall performance score for a memory"""
        if memory_id not in self.usage_counts:
            return 0.5  # Default neutral score

        # Success rate component
        success_scores = self.success_rates.get(memory_id, [])
        avg_success = np.mean(success_scores) if success_scores else 0.5

        # Usage frequency component (more uses = more proven)
        usage_count = self.usage_counts[memory_id]
        usage_score = min(1.0, usage_count / 20)  # Cap at 20 uses

        # Recency component (recently used memories score higher)
        if memory_id in self.temporal_patterns and self.temporal_patterns[memory_id]:
            last_used = max(self.temporal_patterns[memory_id])
            days_since_used = (datetime.now() - last_used).days
            recency_score = max(0.1, 1 - days_since_used / 30)  # 30-day decay
        else:
            recency_score = 0.5

        # Weighted combination
        performance_score = (
            avg_success * 0.5 +      # Success rate (50%)
            usage_score * 0.3 +      # Usage frequency (30%)
            recency_score * 0.2      # Recency (20%)
        )

        return performance_score

    def get_context_effectiveness(self, memory_id: str, context_type: str) -> float:
        """Get how effective a memory is in a specific context"""
        if memory_id not in self.context_patterns:
            return 0.5

        context_counts = self.context_patterns[memory_id]
        total_uses = sum(context_counts.values())

        if total_uses == 0:
            return 0.5

        context_specific_uses = context_counts.get(context_type, 0)
        return context_specific_uses / total_uses

    def get_top_performing_memories(self, limit: int = 10) -> List[Tuple[str, float]]:
        """Get highest performing memories by performance score"""
        memory_scores = [
            (memory_id, self.get_memory_performance_score(memory_id))
            for memory_id in self.usage_counts.keys()
        ]

        memory_scores.sort(key=lambda x: x[1], reverse=True)
        return memory_scores[:limit]

    def analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends and patterns"""
        total_memories = len(self.usage_counts)
        total_uses = sum(self.usage_counts.values())

        if total_memories == 0:
            return {"total_memories": 0, "total_uses": 0}

        # Calculate average performance
        avg_performance = np.mean([
            self.get_memory_performance_score(mid)
            for mid in self.usage_counts.keys()
        ])

        # Find most successful context types
        context_success = defaultdict(list)
        for memory_id in self.usage_counts.keys():
            for context_type, count in self.context_patterns[memory_id].items():
                success_scores = self.success_rates.get(memory_id, [0.5])
                avg_success = np.mean(success_scores)
                context_success[context_type].extend([avg_success] * count)

        context_avg_performance = {
            context: np.mean(scores)
            for context, scores in context_success.items()
        }

        return {
            "total_memories": total_memories,
            "total_uses": total_uses,
            "average_performance": avg_performance,
            "context_performance": context_avg_performance,
            "top_performing_memories": self.get_top_performing_memories(5)
        }


class AdaptiveMemoryScorer:
    """
    Adaptive scoring system that learns from memory usage patterns
    to improve relevance predictions over time.
    """

    def __init__(self, performance_tracker: MemoryPerformanceTracker = None):
        self.performance_tracker = performance_tracker or MemoryPerformanceTracker()

        # Learning parameters
        self.learning_rate = 0.1
        self.forget_factor = 0.95  # How much to forget old patterns

        # Adaptive weights for relevance scoring
        self.relevance_weights = {
            'semantic_similarity': 0.4,
            'importance': 0.25,
            'usage_frequency': 0.15,
            'recency': 0.1,
            'context_match': 0.1
        }

        # Pattern learning
        self.query_memory_patterns = defaultdict(Counter)  # query_type -> memory_success
        self.memory_context_success = defaultdict(lambda: defaultdict(float))

    def adapt_relevance_score(self, memory_id: str, base_relevance: float,
                            context_type: str, query_features: Dict[str, Any]) -> float:
        """
        Adapt relevance score based on learned patterns

        Args:
            memory_id: Memory identifier
            base_relevance: Base relevance score from vector similarity
            context_type: Type of context (conversation, task, etc.)
            query_features: Features extracted from the query

        Returns:
            Adapted relevance score
        """
        # Get performance-based adjustment
        performance_score = self.performance_tracker.get_memory_performance_score(memory_id)
        context_effectiveness = self.performance_tracker.get_context_effectiveness(memory_id, context_type)

        # Learn from query patterns
        query_type = self._classify_query(query_features)
        if query_type:
            self.query_memory_patterns[query_type][memory_id] += performance_score

        # Context-specific learning
        self.memory_context_success[memory_id][context_type] = (
            self.memory_context_success[memory_id][context_type] * (1 - self.learning_rate) +
            performance_score * self.learning_rate
        )

        # Calculate adapted score
        adapted_score = (
            base_relevance * self.relevance_weights['semantic_similarity'] +
            performance_score * self.relevance_weights['usage_frequency'] +
            context_effectiveness * self.relevance_weights['context_match']
        )

        # Normalize to 0-1 range
        return min(1.0, max(0.0, adapted_score))

    def _classify_query(self, query_features: Dict[str, Any]) -> Optional[str]:
        """Classify query type for pattern learning"""
        query_text = query_features.get('text', '').lower()

        # Simple classification based on keywords
        if any(word in query_text for word in ['how', 'what', 'why', 'explain']):
            return 'explanatory'
        elif any(word in query_text for word in ['optimize', 'improve', 'performance']):
            return 'optimization'
        elif any(word in query_text for word in ['error', 'problem', 'issue']):
            return 'troubleshooting'
        elif any(word in query_text for word in ['code', 'function', 'class']):
            return 'coding'
        else:
            return 'general'

    def update_weights_from_feedback(self, feedback_data: Dict[str, Any]):
        """
        Update scoring weights based on feedback about memory effectiveness

        Args:
            feedback_data: Dictionary containing feedback metrics
        """
        # Simple online learning to adjust weights
        if 'memory_effectiveness' in feedback_data:
            effectiveness = feedback_data['memory_effectiveness']

            # Adjust weights based on overall effectiveness
            if effectiveness > 0.7:  # Good performance
                self.relevance_weights['usage_frequency'] *= 1.05
                self.relevance_weights['context_match'] *= 1.05
            elif effectiveness < 0.3:  # Poor performance
                self.relevance_weights['semantic_similarity'] *= 1.05
                self.relevance_weights['importance'] *= 1.05

            # Normalize weights
            total_weight = sum(self.relevance_weights.values())
            self.relevance_weights = {
                k: v / total_weight for k, v in self.relevance_weights.items()
            }

    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights from the learning process"""
        insights = {
            "relevance_weights": self.relevance_weights,
            "performance_trends": self.performance_tracker.analyze_performance_trends(),
            "query_patterns": dict(self.query_memory_patterns),
            "learning_rate": self.learning_rate
        }

        # Add top patterns
        if self.query_memory_patterns:
            top_patterns = sorted(
                [(query_type, dict(memory_counts.most_common(3)))
                 for query_type, memory_counts in self.query_memory_patterns.items()],
                key=lambda x: sum(x[1].values()),
                reverse=True
            )[:3]
            insights["top_query_patterns"] = top_patterns

        return insights

    def reset_learning(self):
        """Reset learned patterns (useful for testing or after major changes)"""
        self.query_memory_patterns.clear()
        self.memory_context_success.clear()
        self.relevance_weights = {
            'semantic_similarity': 0.4,
            'importance': 0.25,
            'usage_frequency': 0.15,
            'recency': 0.1,
            'context_match': 0.1
        }
        print("🔄 Adaptive learning reset to defaults")


def create_adaptive_memory_system() -> Tuple[MemoryPerformanceTracker, AdaptiveMemoryScorer]:
    """
    Create a complete adaptive memory learning system

    Returns:
        Tuple of (performance_tracker, adaptive_scorer)
    """
    tracker = MemoryPerformanceTracker()
    scorer = AdaptiveMemoryScorer(tracker)

    return tracker, scorer


if __name__ == "__main__":
    # Test the adaptive memory system
    print("Testing Adaptive Memory Learning System")
    print("=" * 50)

    tracker, scorer = create_adaptive_memory_system()

    # Simulate some memory usage
    test_memories = ["mem_001", "mem_002", "mem_003"]

    for i in range(10):
        for mem_id in test_memories:
            # Simulate different performance levels
            success_score = 0.5 + (hash(mem_id + str(i)) % 50) / 100  # 0.5-1.0 range
            tracker.record_memory_usage(
                mem_id,
                context_type="conversation" if i % 2 == 0 else "task",
                success_score=success_score
            )

            # Test adaptive scoring
            adapted_score = scorer.adapt_relevance_score(
                mem_id,
                base_relevance=0.7,
                context_type="conversation",
                query_features={"text": "how to optimize performance"}
            )

    # Get insights
    insights = scorer.get_learning_insights()
    print("LEARNING INSIGHTS:")
    print(json.dumps({
        "relevance_weights": insights["relevance_weights"],
        "performance_trends": insights["performance_trends"],
        "top_patterns": insights.get("top_query_patterns", [])
    }, indent=2))

    print("\nTOP PERFORMING MEMORIES:")
    top_memories = tracker.get_top_performing_memories(3)
    for mem_id, score in top_memories:
        print(f"  {mem_id}: {score:.3f}")
