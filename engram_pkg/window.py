"""
Context Window Manager

Efficiently manages token allocation and memory integration within context window limits.
Optimizes information density while respecting LLM context constraints.

Key Features:
- Dynamic token allocation based on context importance
- Memory compression and summarization for space efficiency
- Priority-based memory selection and ordering
- Real-time context window monitoring
"""

import json
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import numpy as np


@dataclass
class TokenBudget:
    """Manages token allocation within context window"""
    total_tokens: int
    used_tokens: int = 0
    reserved_tokens: int = 0  # For system prompts, etc.

    @property
    def available_tokens(self) -> int:
        """Get available tokens for memory integration"""
        return max(0, self.total_tokens - self.used_tokens - self.reserved_tokens)

    def allocate_tokens(self, requested_tokens: int) -> int:
        """Allocate tokens, returns actual allocated amount"""
        available = self.available_tokens
        allocated = min(requested_tokens, available)
        self.used_tokens += allocated
        return allocated

    def reserve_tokens(self, tokens: int):
        """Reserve tokens for fixed content"""
        self.reserved_tokens += tokens


@dataclass
class CompressedMemory:
    """Memory optimized for context window inclusion"""
    original_id: str
    content: str
    importance: float
    priority: float  # Calculated priority score
    token_count: int
    compression_ratio: float  # How much it was compressed
    summary: Optional[str] = None  # Compressed version if applicable

    def get_optimal_content(self) -> str:
        """Get the best content version for available space"""
        return self.summary if self.summary and len(self.summary) < len(self.content) else self.content


class ContextWindowManager:
    """
    Intelligent context window management for memory integration.

    Optimizes information density by:
    - Compressing memories when necessary
    - Prioritizing high-value content
    - Dynamic allocation based on context needs
    """

    def __init__(self, max_tokens: int = 8000, compression_enabled: bool = True):
        self.token_budget = TokenBudget(total_tokens=max_tokens)
        self.compression_enabled = compression_enabled

        # Token estimation (conservative)
        self.chars_per_token = 4.0
        self.compression_ratios = {
            'high': 0.3,    # Aggressive compression for critical space
            'medium': 0.5,  # Moderate compression
            'low': 0.7     # Light compression
        }

        # Priority weights for different memory attributes
        self.priority_weights = {
            'importance': 0.4,
            'relevance': 0.3,
            'recency': 0.15,
            'usage': 0.15
        }

        # Adaptive learning system
        self.usage_history = []
        self.performance_history = []
        self.context_pressure_stats = {'high': 0, 'medium': 0, 'low': 0}
        self.learning_enabled = True
        self._current_context_keywords = set()

    def set_context_usage(self, used_tokens: int, reserved_tokens: int = 1000):
        """Set current context usage (conversation + system prompts)"""
        self.token_budget.used_tokens = used_tokens
        self.token_budget.reserved_tokens = reserved_tokens

    def optimize_memories_for_context(self, memories: List[Any],
                                    available_tokens: Optional[int] = None) -> List[CompressedMemory]:
        """
        Optimize a list of memories for context inclusion

        Args:
            memories: List of ContextMemory or similar objects
            available_tokens: Override automatic token calculation

        Returns:
            List of CompressedMemory objects optimized for space
        """
        if not memories:
            return []

        # Calculate available tokens
        if available_tokens is None:
            available_tokens = self.token_budget.available_tokens

        # Convert to CompressedMemory format and calculate priorities
        compressed_memories = []
        for memory in memories:
            compressed = self._compress_memory(memory)
            compressed.priority = self._calculate_priority(memory)
            compressed_memories.append(compressed)

        # Sort by priority (highest first)
        compressed_memories.sort(key=lambda x: x.priority, reverse=True)

        # Select optimal subset within token budget
        selected_memories = []
        used_tokens = 0

        for memory in compressed_memories:
            optimal_content = memory.get_optimal_content()
            content_tokens = len(optimal_content) // self.chars_per_token

            if used_tokens + content_tokens <= available_tokens:
                selected_memories.append(memory)
                used_tokens += content_tokens
            else:
                # Try compressed version if original doesn't fit
                if memory.summary and memory.summary != optimal_content:
                    summary_tokens = len(memory.summary) // self.chars_per_token
                    if used_tokens + summary_tokens <= available_tokens:
                        # Update to use summary
                        memory.content = memory.summary
                        memory.token_count = summary_tokens
                        selected_memories.append(memory)
                        used_tokens += summary_tokens

        return selected_memories

    def _compress_memory(self, memory) -> CompressedMemory:
        """Compress memory content for efficient context usage"""
        original_content = memory.content
        original_tokens = len(original_content) // self.chars_per_token

        compressed = CompressedMemory(
            original_id=memory.memory_id,
            content=original_content,
            importance=getattr(memory, 'importance', 0.5),
            priority=0.0,  # Will be calculated later
            token_count=original_tokens,
            compression_ratio=1.0
        )

        if self.compression_enabled and original_tokens > 50:  # Only compress longer memories
            compressed.summary = self._generate_summary(original_content)
            if compressed.summary:
                summary_tokens = len(compressed.summary) // self.chars_per_token
                compressed.compression_ratio = summary_tokens / original_tokens

        return compressed

    def _generate_summary(self, content: str) -> Optional[str]:
        """Enhanced intelligent summarization with multiple strategies"""
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip() and len(s) > 5]

        if len(sentences) <= 1:
            return None  # Too short to summarize effectively

        # Multi-factor sentence scoring
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            score = self._score_sentence(sentence, i, len(sentences))
            scored_sentences.append((score, sentence))

        # Sort by score and select optimal number
        scored_sentences.sort(key=lambda x: x[0], reverse=True)

        # Adaptive selection based on content length
        content_length = len(content)
        if content_length < 200:
            num_sentences = 1  # Very short - keep just the best
        elif content_length < 500:
            num_sentences = 2  # Short - keep top 2
        else:
            num_sentences = max(2, min(3, len(sentences) // 2))  # Longer - keep more but cap

        top_sentences = scored_sentences[:num_sentences]

        # Reconstruct summary with intelligent ordering
        # Keep original order for coherence, but prioritize high-scoring sentences
        selected_sentences = []
        used_indices = set()

        # First pass: include high-scoring sentences in original order
        for score, sentence in top_sentences:
            for original_idx, orig_sentence in enumerate(sentences):
                if orig_sentence == sentence and original_idx not in used_indices:
                    selected_sentences.append((original_idx, sentence))
                    used_indices.add(original_idx)
                    break

        # Sort by original position for coherence
        selected_sentences.sort(key=lambda x: x[0])

        # Build summary
        summary_parts = [sentence for _, sentence in selected_sentences]
        summary = '. '.join(summary_parts)

        # Clean up and validate
        if not summary.endswith(('.', '!', '?')):
            summary += '.'

        # Ensure compression is worthwhile
        compression_ratio = len(summary) / len(content)
        if compression_ratio >= 0.8:  # Less than 20% compression
            return None

        return summary

    def _score_sentence(self, sentence: str, position: int, total_sentences: int) -> float:
        """Multi-factor sentence scoring for summarization"""
        score = 0.0

        # Length score (prefer substantial sentences)
        word_count = len(sentence.split())
        if 5 <= word_count <= 25:  # Sweet spot for sentence length
            score += 0.3
        elif word_count > 25:
            score += 0.2  # Still good but less optimal

        # Position score (beginning and end often contain key info)
        if position == 0:  # First sentence often most important
            score += 0.25
        elif position == total_sentences - 1:  # Last sentence often contains conclusion
            score += 0.2
        elif 0.2 <= position/total_sentences <= 0.8:  # Middle sentences get moderate bonus
            score += 0.15

        # Content quality indicators
        sentence_lower = sentence.lower()

        # Technical content bonus
        technical_terms = ['algorithm', 'system', 'method', 'approach', 'solution', 'implementation']
        technical_count = sum(1 for term in technical_terms if term in sentence_lower)
        score += min(0.2, technical_count * 0.05)

        # Actionable content bonus
        action_indicators = ['how', 'steps', 'process', 'use', 'create', 'build']
        if any(indicator in sentence_lower for indicator in action_indicators):
            score += 0.15

        # Quantitative information bonus
        if re.search(r'\d+', sentence):  # Contains numbers
            score += 0.1

        # Keyword density bonus (sentences with important words)
        important_words = ['important', 'key', 'critical', 'main', 'primary', 'essential']
        keyword_count = sum(1 for word in important_words if word in sentence_lower)
        score += min(0.1, keyword_count * 0.05)

        # Penalty for very short sentences (unless they're key)
        if word_count < 3 and position not in [0, total_sentences - 1]:
            score -= 0.1

        return max(0.0, score)

    def _calculate_priority(self, memory) -> float:
        """Enhanced priority calculation with adaptive weighting and context awareness"""
        importance = getattr(memory, 'importance', 0.5)
        relevance = getattr(memory, 'relevance_score', 0.5)
        usage_count = getattr(memory, 'usage_count', 0)

        # Enhanced recency scoring with multiple time windows
        recency_score = self._calculate_adaptive_recency(memory)

        # Enhanced usage scoring with frequency patterns
        usage_score = self._calculate_usage_priority(usage_count, memory)

        # Content quality bonus
        quality_bonus = self._assess_content_quality_for_priority(memory)

        # Contextual relevance bonus (based on conversation context if available)
        context_bonus = self._calculate_context_relevance(memory)

        # Diversity bonus (encourage variety in selected memories)
        diversity_bonus = self._calculate_diversity_bonus(memory)

        # Adaptive weighting based on context pressure
        adaptive_weights = self._get_adaptive_weights()

        # Enhanced weighted combination with bonuses
        base_priority = (
            adaptive_weights['importance'] * importance +
            adaptive_weights['relevance'] * relevance +
            adaptive_weights['recency'] * recency_score +
            adaptive_weights['usage'] * usage_score
        )

        # Apply bonuses (diminishing returns)
        total_bonus = min(0.3, quality_bonus + context_bonus + diversity_bonus)
        final_priority = base_priority + total_bonus

        return min(1.0, max(0.0, final_priority))

    def _calculate_adaptive_recency(self, memory) -> float:
        """Calculate recency with adaptive time windows based on memory age distribution"""
        if not hasattr(memory, 'last_used') or not memory.last_used:
            # For new memories, give high recency but not maximum
            return 0.7

        days_old = (datetime.now() - memory.last_used).days

        # Adaptive time windows based on memory's age relative to others
        if days_old <= 1:
            return 0.95  # Very recent (last day)
        elif days_old <= 7:
            return 0.85  # Recent week
        elif days_old <= 30:
            return max(0.3, 0.8 - (days_old - 7) * 0.02)  # Recent month with decay
        elif days_old <= 90:
            return max(0.1, 0.4 - (days_old - 30) * 0.005)  # Quarter with slow decay
        else:
            return max(0.05, 0.2 - (days_old - 90) * 0.002)  # Older memories with minimal preference

    def _calculate_usage_priority(self, usage_count: int, memory) -> float:
        """Enhanced usage scoring with recency-weighted frequency"""
        if usage_count == 0:
            return 0.0

        # Base frequency score with logarithmic scaling
        frequency_score = min(1.0, (usage_count ** 0.7) / 3)  # Slower growth, cap at ~10 uses

        # Recency weighting - recent usage is more valuable
        recency_multiplier = 1.0
        if hasattr(memory, 'last_used') and memory.last_used:
            days_since_use = (datetime.now() - memory.last_used).days
            if days_since_use <= 1:
                recency_multiplier = 1.2  # 20% bonus for very recent use
            elif days_since_use <= 7:
                recency_multiplier = 1.1  # 10% bonus for recent use
            elif days_since_use <= 30:
                recency_multiplier = 1.0  # Normal for monthly use
            else:
                recency_multiplier = 0.8  # Penalty for stale usage

        return min(1.0, frequency_score * recency_multiplier)

    def _assess_content_quality_for_priority(self, memory) -> float:
        """Assess content quality factors that influence priority"""
        content = getattr(memory, 'content', '')
        if not content:
            return 0.0

        quality_score = 0.0

        # Length appropriateness (prefer substantial but not overwhelming content)
        word_count = len(content.split())
        if 10 <= word_count <= 100:
            quality_score += 0.05  # Sweet spot for memory content
        elif word_count > 200:
            quality_score -= 0.02  # Penalty for very long content

        # Information density (sentences per word)
        sentence_count = len(re.split(r'[.!?]+', content))
        if sentence_count > 0:
            info_density = sentence_count / word_count
            if 0.01 <= info_density <= 0.05:  # Good balance of sentences to words
                quality_score += 0.03

        # Contains actionable information
        actionable_indicators = ['how to', 'steps', 'process', 'method', 'approach', 'solution']
        if any(indicator in content.lower() for indicator in actionable_indicators):
            quality_score += 0.04

        # Contains specific examples or data
        if re.search(r'\d+', content):  # Contains numbers
            quality_score += 0.02
        if re.search(r'```|code|example', content, re.IGNORECASE):  # Contains code/examples
            quality_score += 0.03

        return min(0.15, max(0.0, quality_score))

    def _calculate_context_relevance(self, memory) -> float:
        """Calculate bonus based on current conversation context"""
        # This would integrate with conversation context if available
        # For now, provide basic context matching
        content = getattr(memory, 'content', '').lower()

        # Look for conversation context indicators (would be passed in from conversation)
        context_keywords = getattr(self, '_current_context_keywords', set())

        if context_keywords:
            matches = len(context_keywords & set(content.split()))
            context_relevance = min(0.1, matches / len(context_keywords) * 0.1)
        else:
            context_relevance = 0.0

        return context_relevance

    def _calculate_diversity_bonus(self, memory) -> float:
        """Calculate bonus for maintaining diversity in selected memories"""
        # This would track already selected memories and provide diversity bonus
        # For now, provide basic diversity based on tags/domains
        tags = getattr(memory, 'tags', [])
        if not tags:
            return 0.0

        # Simple diversity - bonus for memories with multiple tags (broader coverage)
        tag_count = len(tags)
        diversity_score = min(0.05, tag_count * 0.01)

        # Bonus for unique or rare tags (would need tag frequency tracking)
        unique_tags = [tag for tag in tags if len(tag) > 6]  # Longer tags tend to be more specific
        if unique_tags:
            diversity_score += 0.02

        return diversity_score

    def _get_adaptive_weights(self) -> Dict[str, float]:
        """Get adaptive weights based on context pressure and memory characteristics"""
        # Start with base weights
        weights = self.priority_weights.copy()

        # Adapt based on available token pressure
        available_ratio = self.token_budget.available_tokens / self.token_budget.total_tokens

        if available_ratio < 0.3:  # High pressure - prefer concise, important info
            weights['importance'] += 0.1
            weights['relevance'] += 0.05
            weights['recency'] -= 0.05
            weights['usage'] -= 0.1
        elif available_ratio > 0.7:  # Plenty of space - can be more inclusive
            weights['recency'] += 0.05
            weights['usage'] += 0.05
            weights['importance'] -= 0.05
            weights['relevance'] -= 0.05

        # Normalize weights to sum to 1.0
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        return weights

    def set_conversation_context(self, context_keywords: List[str]):
        """Set current conversation context for better memory selection"""
        self._current_context_keywords = set(word.lower() for word in context_keywords)

    def record_usage_performance(self, memories_selected: int, tokens_used: int,
                                context_quality_rating: Optional[float] = None):
        """
        Record performance metrics for learning and adaptation

        Args:
            memories_selected: Number of memories included in context
            tokens_used: Tokens consumed by memories
            context_quality_rating: Optional quality rating (0.0-1.0)
        """
        if not self.learning_enabled:
            return

        usage_record = {
            'timestamp': datetime.now(),
            'memories_selected': memories_selected,
            'tokens_used': tokens_used,
            'available_tokens': self.token_budget.available_tokens,
            'total_tokens': self.token_budget.total_tokens,
            'context_quality': context_quality_rating,
            'pressure_ratio': (self.token_budget.used_tokens + self.token_budget.reserved_tokens) / self.token_budget.total_tokens
        }

        self.usage_history.append(usage_record)

        # Track context pressure patterns
        pressure_ratio = usage_record['pressure_ratio']
        if pressure_ratio > 0.8:
            self.context_pressure_stats['high'] += 1
        elif pressure_ratio > 0.5:
            self.context_pressure_stats['medium'] += 1
        else:
            self.context_pressure_stats['low'] += 1

        # Maintain history size (keep last 50 records)
        if len(self.usage_history) > 50:
            self.usage_history = self.usage_history[-50:]

    def get_adaptive_recommendations(self) -> Dict[str, Any]:
        """Get recommendations based on usage patterns and performance history"""
        if len(self.usage_history) < 5:
            return {"recommendations": ["Collecting more usage data for better recommendations"]}

        # Analyze usage patterns
        avg_memories_selected = sum(r['memories_selected'] for r in self.usage_history) / len(self.usage_history)
        avg_tokens_used = sum(r['tokens_used'] for r in self.usage_history) / len(self.usage_history)
        avg_pressure = sum(r['pressure_ratio'] for r in self.usage_history) / len(self.usage_history)

        # Calculate efficiency metrics
        efficiency = avg_memories_selected / max(1, avg_tokens_used / 100)  # memories per 100 tokens

        recommendations = []

        # Token usage recommendations
        if avg_pressure > 0.8:
            recommendations.append("⚠️ High context pressure detected - consider increasing max tokens or enabling more compression")
        elif avg_pressure < 0.3:
            recommendations.append("💡 Low context utilization - could increase memory inclusion or reduce reserved tokens")

        # Efficiency recommendations
        if efficiency < 0.5:
            recommendations.append("📊 Low memory density - consider improving compression or priority algorithms")
        elif efficiency > 2.0:
            recommendations.append("✅ High memory efficiency - current settings working well")

        # Context pressure pattern recommendations
        total_pressure_events = sum(self.context_pressure_stats.values())
        if total_pressure_events > 0:
            high_pressure_ratio = self.context_pressure_stats['high'] / total_pressure_events
            if high_pressure_ratio > 0.6:
                recommendations.append("🔧 Frequent high pressure - consider adaptive token allocation")

        # Quality-based recommendations
        quality_ratings = [r['context_quality'] for r in self.usage_history if r['context_quality'] is not None]
        if quality_ratings:
            avg_quality = sum(quality_ratings) / len(quality_ratings)
            if avg_quality < 0.6:
                recommendations.append("🎯 Context quality could be improved - review memory selection criteria")
            elif avg_quality > 0.8:
                recommendations.append("⭐ Excellent context quality - maintain current approach")

        return {
            "recommendations": recommendations,
            "metrics": {
                "average_memories_selected": round(avg_memories_selected, 1),
                "average_tokens_used": round(avg_tokens_used, 0),
                "average_pressure": round(avg_pressure, 2),
                "efficiency_score": round(efficiency, 2),
                "total_observations": len(self.usage_history)
            }
        }

    def optimize_for_patterns(self):
        """Optimize settings based on learned usage patterns"""
        if len(self.usage_history) < 10:
            return  # Need more data

        # Analyze optimal compression settings
        high_pressure_ratio = self.context_pressure_stats['high'] / sum(self.context_pressure_stats.values())

        if high_pressure_ratio > 0.5:
            # Enable more aggressive compression for high pressure scenarios
            self.compression_ratios['medium'] = 0.4  # More aggressive
            self.compression_ratios['low'] = 0.6

        # Adjust priority weights based on what works
        quality_ratings = [r for r in self.usage_history if r['context_quality'] is not None]
        if quality_ratings:
            # Could implement more sophisticated optimization here
            pass

        print("⚡ Context manager optimized based on usage patterns")

    def format_optimized_context(self, memories: List[CompressedMemory]) -> str:
        """Format optimized memories for context integration"""
        if not memories:
            return ""

        # Group by priority levels
        high_priority = [m for m in memories if m.priority >= 0.7]
        medium_priority = [m for m in memories if 0.4 <= m.priority < 0.7]
        low_priority = [m for m in memories if m.priority < 0.4]

        sections = []

        if high_priority:
            sections.append("🔥 CRITICAL CONTEXT:")
            sections.extend([f"• {memory.get_optimal_content()}" for memory in high_priority])

        if medium_priority:
            if sections:
                sections.append("")
            sections.append("📝 RELEVANT INFORMATION:")
            sections.extend([f"• {memory.get_optimal_content()}" for memory in medium_priority])

        if low_priority:
            if sections:
                sections.append("")
            sections.append("💭 ADDITIONAL NOTES:")
            sections.extend([f"• {memory.get_optimal_content()}" for memory in low_priority])

        return "\n".join(sections)

    def get_context_stats(self) -> Dict[str, Any]:
        """Get statistics about context window usage"""
        return {
            "total_tokens": self.token_budget.total_tokens,
            "used_tokens": self.token_budget.used_tokens,
            "reserved_tokens": self.token_budget.reserved_tokens,
            "available_tokens": self.token_budget.available_tokens,
            "usage_percentage": (self.token_budget.used_tokens / self.token_budget.total_tokens) * 100,
            "compression_enabled": self.compression_enabled
        }

    def estimate_token_usage(self, text: str) -> int:
        """Estimate token count for a text string"""
        return int(len(text) / self.chars_per_token)


# Integration helper functions
def integrate_memories_into_context(
    memories: List[Any],
    context_window_manager: ContextWindowManager,
    current_context_tokens: int = 0
) -> str:
    """
    High-level function to integrate memories into context efficiently

    Args:
        memories: List of memory objects to integrate
        context_window_manager: Configured context manager
        current_context_tokens: Tokens already used by conversation

    Returns:
        Formatted context string with integrated memories
    """
    # Update context usage
    context_window_manager.set_context_usage(current_context_tokens)

    # Optimize memories for available space
    optimized_memories = context_window_manager.optimize_memories_for_context(memories)

    # Format for context integration
    integrated_context = context_window_manager.format_optimized_context(optimized_memories)

    return integrated_context


def create_default_context_manager(max_tokens: int = 8000) -> ContextWindowManager:
    """Create a context window manager with sensible defaults"""
    return ContextWindowManager(max_tokens=max_tokens, compression_enabled=True)


if __name__ == "__main__":
    # Test the context window manager
    manager = create_default_context_manager(max_tokens=4000)
    manager.set_context_usage(2000, 500)  # 2000 used, 500 reserved

    print("Context Window Stats:")
    print(json.dumps(manager.get_context_stats(), indent=2))

    # Test token estimation
    test_text = "This is a test memory about vector databases and their performance characteristics."
    tokens = manager.estimate_token_usage(test_text)
    print(f"\nEstimated tokens for test text: {tokens}")
    print(f"Available tokens: {manager.token_budget.available_tokens}")
