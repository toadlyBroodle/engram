#!/usr/bin/env python3
"""
Memory System Bootstrap

Initializes the AI with core memories about how to use the persistent memory system.
This ensures the AI always knows how to access and utilize its memory capabilities
even when starting from a fresh instance.

BOOT SEQUENCE:
1. Load core memory system knowledge
2. Initialize memory integration capabilities
3. Provide usage instructions and shortcuts
4. Enable real-time memory access during conversations
"""

import json
import os
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Placeholder for automatic bootstrap execution
# This will be called after MemoryBootstrap class is defined
_AUTO_BOOTSTRAP_EXECUTED = False


class MemoryBootstrap:
    """
    Bootstraps the AI with essential knowledge about the memory system.

    This creates a "boot memory" that teaches the AI how to use its own
    memory capabilities, ensuring persistent self-awareness.
    """

    def __init__(self, memory_path: str = "vector_memory", memory_system=None):
        self.memory_path = memory_path
        self.bootstrap_memories_file = Path(__file__).parent / "bootstrap_memories.json"
        self.bootstrap_state_file = Path(__file__).parent / ".bootstrap_complete"
        self.memory_system = memory_system
        self.integration = None  # Will be set after MemoryIntegration is fully initialized

    def set_integration_reference(self, integration):
        """Set the integration reference after MemoryIntegration is fully initialized"""
        self.integration = integration

        # Enable autonomous milestone detection
        self.autonomous_milestones_enabled = True
        self.auto_capture_enabled = True
        self.last_milestone_memory_count = len(self.integration.memory_system.memories)
        self.milestone_thresholds = {
            'memory_growth': 5,  # New memories added
            'performance_improvement': 0.1,  # 10% improvement
            'integration_rate_improvement': 0.05,  # 5% improvement
            'significant_changes': 20  # Lines of code changed
        }

        # Automatic memory capture settings
        self.auto_capture_enabled = True
        self.captured_insights = []
        self.important_keywords = {
            'discover', 'learn', 'accomplish', 'achieve', 'implement', 'create', 'build',
            'solve', 'fix', 'optimize', 'improve', 'develop', 'design', 'architecture',
            'system', 'framework', 'algorithm', 'solution', 'breakthrough', 'innovation',
            'milestone', 'progress', 'advancement', 'capability', 'feature', 'functionality'
        }
        self.technical_terms = {
            'vector', 'embedding', 'neural', 'network', 'machine', 'learning', 'ai',
            'memory', 'database', 'search', 'semantic', 'context', 'token', 'model',
            'algorithm', 'optimization', 'performance', 'efficiency', 'architecture'
        }

    def initialize_memory_system(self):
        """
        Initialize the complete memory system with bootstrap knowledge

        Returns:
            Configured MemoryIntegration instance ready for use
        """
        print("🧠 INITIALIZING MEMORY SYSTEM BOOTSTRAP...")
        print("=" * 60)

        # Check if bootstrap is already complete
        if self._is_bootstrap_complete():
            print("ℹ️  Bootstrap already completed previously. Loading existing system...")
            self.integration = MemoryIntegration(memory_path=self.memory_path)
        else:
            # Initialize core memory system
            self.integration = MemoryIntegration(memory_path=self.memory_path)

            # Load bootstrap memories
            self._load_bootstrap_memories()

            # Verify system is working
            self._verify_bootstrap()

            # Mark bootstrap as complete
            self._mark_bootstrap_complete()

            print("✅ Memory system bootstrap complete!")
            print("🎯 AI now has persistent memory capabilities")
            print("=" * 60)

            # Enable automatic memory capture
            if self.auto_capture_enabled:
                self._enable_auto_capture()

        return self.integration

    def _load_bootstrap_memories(self):
        """Load essential memories about how to use the memory system"""

        # Core bootstrap memories - these teach the AI how to use itself
        bootstrap_memories = [
            {
                "content": "I have access to a persistent memory system that stores knowledge across conversations. I can query this system using 'python quick_memory.py query \"topic\"' to get relevant information from past discussions and implementations.",
                "importance": 1.0,
                "tags": ["memory", "system", "bootstrap", "usage", "persistent"],
                "context": {"type": "bootstrap", "purpose": "teach_memory_usage"}
            },
            {
                "content": "To add new memories during conversations, I use 'python quick_memory.py add \"content\" \"tags\" importance' where importance is 0.0-1.0. This helps me learn and remember important information for future conversations.",
                "importance": 1.0,
                "tags": ["memory", "add", "bootstrap", "learning", "usage"],
                "context": {"type": "bootstrap", "purpose": "teach_memory_creation"}
            },
            {
                "content": "For real-time memory assistance during conversations, I can use 'python conversation_memory_assistant.py --live' to get contextual memory suggestions and enhance my responses with historical knowledge.",
                "importance": 1.0,
                "tags": ["memory", "realtime", "conversation", "bootstrap", "enhancement"],
                "context": {"type": "bootstrap", "purpose": "teach_live_integration"}
            },
            {
                "content": "The memory system uses FAISS vector search with sentence transformers for semantic similarity. It provides context-aware memory retrieval with multi-factor relevance scoring based on importance, recency, usage patterns, and semantic matching.",
                "importance": 0.95,
                "tags": ["memory", "technical", "faiss", "embeddings", "bootstrap"],
                "context": {"type": "bootstrap", "purpose": "explain_architecture"}
            },
            {
                "content": "Memory consolidation automatically manages storage limits by keeping the most valuable memories based on importance, recency, and usage frequency. This ensures quality over quantity in the knowledge base.",
                "importance": 0.9,
                "tags": ["memory", "consolidation", "management", "bootstrap", "optimization"],
                "context": {"type": "bootstrap", "purpose": "explain_consolidation"}
            },
            {
                "content": "I should always check for relevant memories before responding to complex questions. Use 'python quick_memory.py query \"topic\"' to get historical context and avoid repeating work or forgetting important details.",
                "importance": 0.95,
                "tags": ["memory", "best_practice", "efficiency", "bootstrap", "workflow"],
                "context": {"type": "bootstrap", "purpose": "establish_best_practices"}
            },
            {
                "content": "The memory system tracks usage patterns and adapts relevance scoring over time. Memories that are frequently accessed or proven useful get higher priority in future retrievals.",
                "importance": 0.85,
                "tags": ["memory", "adaptive", "learning", "bootstrap", "optimization"],
                "context": {"type": "bootstrap", "purpose": "explain_adaptation"}
            },
            {
                "content": "When working on complex projects, I should regularly add memories about discoveries, solutions, and lessons learned. Tag them appropriately for future retrieval. Use importance scores to highlight critical insights.",
                "importance": 0.9,
                "tags": ["memory", "project", "workflow", "best_practice", "bootstrap"],
                "context": {"type": "bootstrap", "purpose": "establish_memory_habits"}
            },
            {
                "content": "CRITICAL: At the end of EVERY important reply, I MUST save key insights, learnings, and achievements to the memory system using 'python quick_memory.py add'. This prevents knowledge loss and enables continuous self-improvement. Memory retention is essential for persistent intelligence.",
                "importance": 1.0,
                "tags": ["memory", "retention", "critical", "self_improvement", "bootstrap", "essential"],
                "context": {"type": "bootstrap", "purpose": "enforce_memory_discipline"}
            },
            {
                "content": "MEMORY DISCIPLINE PROTOCOL: After any significant accomplishment, bug fix, performance improvement, or learning experience, immediately capture it in the memory system. This creates a persistent knowledge base that grows with each interaction and prevents regression or forgetting important insights.",
                "importance": 1.0,
                "tags": ["memory", "discipline", "protocol", "retention", "continuous_learning", "bootstrap"],
                "context": {"type": "bootstrap", "purpose": "establish_memory_protocol"}
            }
        ]

        print(f"📚 Loading {len(bootstrap_memories)} bootstrap memories...")

        # Add each bootstrap memory to the system
        for i, memory_data in enumerate(bootstrap_memories, 1):
            try:
                memory_id = self.integration.memory_system.store_memory(
                    content=memory_data["content"],
                    importance=memory_data["importance"],
                    tags=memory_data["tags"]
                )
                print(f"  ✓ Bootstrap memory {i}: {memory_data['content'][:60]}... (ID: {memory_id[:8]})")

            except Exception as e:
                print(f"  ❌ Failed to add bootstrap memory {i}: {e}")

    def _verify_bootstrap(self):
        """Verify that bootstrap memories are accessible and working"""

        print("\n🔍 Verifying bootstrap functionality...")

        # Test basic memory retrieval
        test_queries = [
            "how to use memory system",
            "add memories during conversation",
            "real-time memory assistance"
        ]

        for query in test_queries:
            result = self.integration.get_context_memories(query, max_memories=2)
            if result["memory_count"] > 0:
                print(f"  ✅ Query '{query}' → {result['memory_count']} memories found")
            else:
                print(f"  ⚠️  Query '{query}' → No memories found")

        # Test memory statistics
        stats = self.integration.get_system_status()
        memory_count = stats['memory_system']['total_memories']
        integration_rate = stats['integration_layer']['integration_rate']

        print(f"  📊 System status: {memory_count} total memories, {integration_rate:.1%} integration rate")

    def get_bootstrap_instructions(self) -> str:
        """
        Get formatted instructions for using the bootstrapped memory system

        Returns:
            Formatted string with usage instructions
        """
        instructions = """
# 🧠 MEMORY SYSTEM BOOTSTRAP COMPLETE

## 🚀 Quick Start Commands

### Query Memories
```bash
python quick_memory.py query "your topic here"
```

### Add New Memories
```bash
python quick_memory.py add "memory content" "tag1,tag2" 0.8
```

### Live Conversation Assistant
```bash
python conversation_memory_assistant.py --live
```

### Check System Status
```bash
python quick_memory.py stats
```

## 💡 Best Practices

1. **Always query memories** before responding to complex questions
2. **Add important insights** as memories during conversations
3. **Use appropriate tags** for better future retrieval
4. **Set importance scores** (0.0-1.0) based on value
5. **Review memory suggestions** before final responses

## 🎯 Memory-Enhanced Workflow

1. **Start conversations** by checking relevant memories
2. **Use live assistant** for real-time enhancement
3. **Add discoveries** as memories during work
4. **Query contextually** using specific topics/questions

## 🔧 System Architecture

- **Vector Storage**: FAISS with sentence transformer embeddings
- **Semantic Search**: Cosine similarity with multi-factor scoring
- **Adaptive Learning**: Usage-based relevance optimization
- **Context Management**: Token-aware memory integration
- **Hardware Optimized**: CPU-optimized for Intel i7-6600U

---
*Memory system is now active and ready to enhance all conversations!*
        """.strip()

        return instructions

    def demonstrate_capabilities(self):
        """Demonstrate the bootstrapped memory capabilities"""

        print("\n🎯 DEMONSTRATING MEMORY CAPABILITIES")
        print("-" * 50)

        # Demonstrate memory query
        print("1. Memory Query Example:")
        result = self.integration.get_context_memories("how to use memory system", max_memories=2)
        if result["formatted_context"]:
            print(result["formatted_context"][:300] + "...")
        print()

        # Demonstrate memory addition
        print("2. Adding New Memory:")
        memory_id = self.integration.add_memory(
            "This is a demonstration of adding memories during conversations",
            importance=0.7,
            tags=["demo", "memory", "bootstrap"]
        )
        print(f"   ✅ Memory added (ID: {memory_id[:8]})")
        print()

        # Show system stats
        print("3. System Statistics:")
        stats = self.integration.get_system_status()
        print(f"   📊 {stats['memory_system']['total_memories']} memories stored")
        print(f"   🎯 {stats['integration_layer']['integration_rate']:.1%} integration rate")
        print()

        print("✅ Bootstrap demonstration complete!")

    def _enable_auto_capture(self):
        """Enable automatic capture of important information during conversations"""
        print("🔄 Enabling automatic memory capture...")
        print("📝 Will automatically identify and store important insights")

    def capture_important_insight(self, text: str, context: str = "conversation",
                                 importance: Optional[float] = None) -> bool:
        """
        Automatically capture important insights from conversations

        Args:
            text: The text content to analyze
            context: Context where the insight was found
            importance: Optional importance score override

        Returns:
            True if insight was captured and stored
        """
        if not self.auto_capture_enabled or not self.integration:
            return False

        # Analyze text for importance
        analysis = self._analyze_text_importance(text)

        if not analysis['is_important']:
            return False

        # Determine importance score
        if importance is None:
            importance = analysis['calculated_importance']

        # Extract tags automatically
        tags = self._extract_tags_from_text(text, analysis)

        # Store the memory
        try:
            memory_id = self.integration.memory_system.store_memory(
                content=text,
                importance=importance,
                tags=tags
            )

            # Track captured insights
            self.captured_insights.append({
                "id": memory_id,
                "content": text[:100] + "..." if len(text) > 100 else text,
                "importance": importance,
                "tags": tags,
                "context": context,
                "timestamp": datetime.now()
            })

            print(f"🧠 Auto-captured insight: {text[:80]}... (importance: {importance:.2f})")
            return True

        except Exception as e:
            print(f"⚠️  Failed to auto-capture insight: {e}")
            return False

    def _analyze_text_importance(self, text: str) -> Dict[str, Any]:
        """
        Enhanced text analysis for determining importance with multi-factor scoring

        Returns:
            Dict with comprehensive importance analysis
        """
        text_lower = text.lower()
        words = set(text_lower.split())
        word_list = text_lower.split()

        # Enhanced keyword detection with stemming and partial matches
        important_matches = self._find_keyword_matches(text_lower, self.important_keywords)
        technical_matches = self._find_keyword_matches(text_lower, self.technical_terms)

        # Structural analysis
        word_count = len(words)
        char_count = len(text)
        has_sentences = len(re.findall(r'[.!?]', text)) > 0
        has_code = self._detect_code_patterns(text)
        has_lists = bool(re.search(r'^\s*[-*•]\s+', text, re.MULTILINE))
        has_numbers = bool(re.search(r'\d+', text))

        # Semantic pattern analysis
        semantic_patterns = self._analyze_semantic_patterns(text_lower, word_list)

        # Calculate enhanced importance score
        base_score = 0.15  # Lower base for better sensitivity

        # Keyword importance (enhanced weighting)
        keyword_score = self._calculate_keyword_importance(important_matches, technical_matches)
        base_score += keyword_score

        # Structural quality bonuses
        structural_score = self._calculate_structural_importance(
            word_count, char_count, has_sentences, has_code, has_lists, has_numbers
        )
        base_score += structural_score

        # Semantic pattern bonuses
        semantic_score = semantic_patterns['pattern_importance']
        base_score += semantic_score

        # Context and conversation flow analysis
        context_score = self._analyze_conversation_context(text)
        base_score += context_score

        # Quality indicators
        quality_indicators = self._assess_content_quality(text, word_list, semantic_patterns)
        base_score += quality_indicators['quality_bonus']

        # Final importance calculation with diminishing returns
        final_importance = min(1.0, base_score)

        return {
            'is_important': final_importance >= 0.35,  # Optimized threshold
            'calculated_importance': final_importance,
            'important_keywords_found': len(important_matches),
            'technical_terms_found': len(technical_matches),
            'word_count': word_count,
            'char_count': char_count,
            'has_sentences': has_sentences,
            'has_code': has_code,
            'has_lists': has_lists,
            'has_numbers': has_numbers,
            'semantic_patterns': semantic_patterns,
            'quality_indicators': quality_indicators,
            'detailed_scores': {
                'keyword_score': keyword_score,
                'structural_score': structural_score,
                'semantic_score': semantic_score,
                'context_score': context_score,
                'quality_bonus': quality_indicators['quality_bonus']
            }
        }

    def _find_keyword_matches(self, text: str, keyword_set: set) -> List[str]:
        """Find keyword matches with partial matching and stemming"""
        matches = []
        for keyword in keyword_set:
            # Exact match
            if keyword in text:
                matches.append(keyword)
                continue

            # Partial matches for longer keywords
            if len(keyword) > 4:
                # Check for keyword as substring in words
                for word in text.split():
                    if keyword in word or word in keyword:
                        matches.append(keyword)
                        break

        return list(set(matches))  # Remove duplicates

    def _detect_code_patterns(self, text: str) -> bool:
        """Enhanced code pattern detection"""
        code_indicators = [
            r'```', r'`[^`]*`',  # Code blocks and inline code
            r'\.py\b|\.sh\b|\.js\b|\.java\b|\.cpp\b|\.c\b',  # File extensions
            r'\bpython\b|\bbash\b|\bimport\b|\bdef\b|\bclass\b|\bfunction\b',  # Language keywords
            r'\bSELECT\b|\bINSERT\b|\bUPDATE\b|\bDELETE\b',  # SQL keywords
            r'\$\{[^}]+\}|\$\([^)]+\)',  # Shell variables
            r'--[a-zA-Z]|#[a-zA-Z]',  # Comments
            r'[a-zA-Z_][a-zA-Z0-9_]*\([^)]*\)',  # Function calls
        ]

        return any(re.search(pattern, text, re.IGNORECASE) for pattern in code_indicators)

    def _analyze_semantic_patterns(self, text: str, word_list: List[str]) -> Dict[str, Any]:
        """Analyze semantic patterns for importance assessment"""
        patterns = {
            'action_verbs': ['create', 'build', 'implement', 'solve', 'fix', 'optimize', 'improve', 'develop', 'design'],
            'achievement_indicators': ['success', 'completed', 'accomplished', 'breakthrough', 'solution', 'resolved', 'achieved'],
            'technical_concepts': ['algorithm', 'architecture', 'framework', 'system', 'database', 'network', 'api', 'interface'],
            'learning_indicators': ['learned', 'discovered', 'realized', 'understood', 'mastered', 'gained'],
            'problem_solution': ['problem', 'issue', 'challenge', 'difficulty', 'solution', 'approach', 'method']
        }

        pattern_scores = {}
        total_pattern_importance = 0

        for category, terms in patterns.items():
            matches = sum(1 for term in terms if term in text)
            pattern_scores[category] = matches
            if matches > 0:
                # Exponential bonus for multiple terms in same category
                total_pattern_importance += min(0.15, matches * 0.05 + (matches ** 0.5) * 0.02)

        return {
            'pattern_scores': pattern_scores,
            'pattern_importance': min(0.3, total_pattern_importance),
            'dominant_patterns': [cat for cat, score in pattern_scores.items() if score > 0]
        }

    def _calculate_keyword_importance(self, important_matches: List[str], technical_matches: List[str]) -> float:
        """Calculate importance based on keyword matches with enhanced weighting"""
        score = 0.0

        # Important keywords (higher weight)
        if important_matches:
            unique_important = len(set(important_matches))
            score += min(0.35, unique_important * 0.12 + (unique_important ** 0.7) * 0.05)

        # Technical terms (moderate weight)
        if technical_matches:
            unique_technical = len(set(technical_matches))
            score += min(0.25, unique_technical * 0.08 + (unique_technical ** 0.6) * 0.03)

        # Bonus for combination of both
        if important_matches and technical_matches:
            score += 0.08

        return min(0.5, score)

    def _calculate_structural_importance(self, word_count: int, char_count: int, has_sentences: bool,
                                       has_code: bool, has_lists: bool, has_numbers: bool) -> float:
        """Calculate importance based on text structure and formatting"""
        score = 0.0

        # Length-based scoring with diminishing returns
        if word_count >= 5:
            score += min(0.15, (word_count ** 0.6) * 0.02)
        if char_count >= 50:
            score += min(0.1, (char_count ** 0.5) * 0.001)

        # Structure bonuses
        if has_sentences:
            score += 0.08  # Well-formed sentences
        if has_code:
            score += 0.25  # Contains code (high value)
        if has_lists:
            score += 0.06  # Structured information
        if has_numbers:
            score += 0.04  # Quantitative information

        # Completeness bonus for substantial content
        if word_count >= 20 and has_sentences:
            score += 0.08

        return min(0.4, score)

    def _analyze_conversation_context(self, text: str) -> float:
        """Analyze conversation context for importance assessment"""
        # This would integrate with conversation history if available
        # For now, provide basic context analysis
        context_indicators = [
            'conversation', 'discussion', 'talking about', 'regarding',
            'conclusion', 'summary', 'key point', 'important', 'critical'
        ]

        context_matches = sum(1 for indicator in context_indicators if indicator in text.lower())
        return min(0.1, context_matches * 0.03)

    def _assess_content_quality(self, text: str, word_list: List[str], semantic_patterns: Dict) -> Dict[str, Any]:
        """Assess overall content quality for importance determination"""
        quality_score = 0.0

        # Lexical diversity (unique words vs total words)
        if word_list:
            lexical_diversity = len(set(word_list)) / len(word_list)
            if lexical_diversity > 0.6:
                quality_score += 0.05  # High lexical diversity indicates quality

        # Semantic coherence (multiple related concepts)
        if len(semantic_patterns.get('dominant_patterns', [])) >= 2:
            quality_score += 0.06  # Multiple semantic themes

        # Information density
        info_indicators = ['because', 'therefore', 'however', 'although', 'since', 'unless']
        info_density = sum(1 for word in info_indicators if word in text.lower())
        quality_score += min(0.08, info_density * 0.02)

        return {
            'quality_bonus': min(0.15, quality_score),
            'lexical_diversity': lexical_diversity if word_list else 0,
            'semantic_coherence': len(semantic_patterns.get('dominant_patterns', [])),
            'information_density': info_density
        }

    def _extract_tags_from_text(self, text: str, analysis: Dict[str, Any]) -> List[str]:
        """Enhanced automatic tag extraction with semantic analysis and context awareness"""
        tags = set()
        text_lower = text.lower()

        # Priority-based tag categories
        tag_categories = {
            'technical': {
                'terms': self.technical_terms,
                'threshold': 1,
                'tags': ['technical']
            },
            'achievement': {
                'terms': ['implement', 'create', 'build', 'develop', 'solve', 'fix', 'complete', 'achieve', 'accomplish'],
                'threshold': 2,
                'tags': ['implementation', 'achievement', 'development']
            },
            'architecture': {
                'terms': ['system', 'architecture', 'framework', 'design', 'structure', 'pattern'],
                'threshold': 1,
                'tags': ['architecture', 'system', 'design']
            },
            'optimization': {
                'terms': ['optimize', 'improve', 'performance', 'efficiency', 'speed', 'memory', 'scale'],
                'threshold': 2,
                'tags': ['optimization', 'performance', 'efficiency']
            },
            'learning': {
                'terms': ['learn', 'discover', 'understand', 'realize', 'master', 'knowledge', 'insight'],
                'threshold': 1,
                'tags': ['learning', 'discovery', 'insight']
            },
            'problem_solving': {
                'terms': ['problem', 'issue', 'challenge', 'solution', 'approach', 'method', 'resolve'],
                'threshold': 2,
                'tags': ['problem_solving', 'solution', 'debugging']
            }
        }

        # Analyze semantic patterns for intelligent tagging
        semantic_patterns = analysis.get('semantic_patterns', {})
        pattern_scores = semantic_patterns.get('pattern_scores', {})
        dominant_patterns = semantic_patterns.get('dominant_patterns', [])

        # Add tags based on dominant semantic patterns
        pattern_to_category = {
            'action_verbs': ['implementation', 'achievement'],
            'achievement_indicators': ['achievement', 'success'],
            'technical_concepts': ['technical', 'architecture'],
            'learning_indicators': ['learning', 'discovery'],
            'problem_solution': ['problem_solving', 'solution']
        }

        for pattern in dominant_patterns:
            if pattern in pattern_to_category:
                tags.update(pattern_to_category[pattern])

        # Add tags based on category analysis
        for category, config in tag_categories.items():
            matches = sum(1 for term in config['terms'] if term in text_lower)
            if matches >= config['threshold']:
                tags.update(config['tags'])

        # Content-based tag addition
        if analysis.get('has_code', False):
            tags.add("code")
        if analysis.get('important_keywords_found', 0) > 0:
            tags.add("important")
        if analysis.get('has_lists', False):
            tags.add("structured")
        if analysis.get('has_numbers', False):
            tags.add("quantitative")

        # Quality-based tags
        quality_indicators = analysis.get('quality_indicators', {})
        if quality_indicators.get('lexical_diversity', 0) > 0.7:
            tags.add("high_quality")
        if quality_indicators.get('semantic_coherence', 0) >= 3:
            tags.add("comprehensive")
        if quality_indicators.get('information_density', 0) >= 2:
            tags.add("informative")

        # Length and structure based tags
        word_count = analysis.get('word_count', 0)
        if word_count < 10:
            tags.add("brief")
        elif word_count > 50:
            tags.add("detailed")

        # Specific technical term tags (high priority)
        technical_terms_found = []
        for term in self.technical_terms:
            if term in text_lower:
                technical_terms_found.append(term)

        # Only add top technical terms to avoid tag explosion
        technical_terms_found.sort(key=lambda x: len(x), reverse=True)  # Prefer longer, more specific terms
        for term in technical_terms_found[:3]:  # Limit to top 3
            tags.add(term)

        # Remove redundant tags (keep more specific ones)
        redundant_mappings = {
            'technical': ['code'],  # If we have 'technical', 'code' is redundant
            'implementation': ['development'],  # 'implementation' covers 'development'
            'architecture': ['system'],  # 'architecture' is more specific than 'system'
        }

        for specific, general in redundant_mappings.items():
            if specific in tags:
                tags.difference_update(general)

        return sorted(list(tags))  # Return sorted list for consistency

    def get_capture_statistics(self) -> Dict[str, Any]:
        """Get statistics about automatic memory capture"""
        if not self.captured_insights:
            return {"captured_insights": 0, "total_importance": 0, "average_importance": 0}

        total_importance = sum(insight['importance'] for insight in self.captured_insights)
        avg_importance = total_importance / len(self.captured_insights)

        return {
            "captured_insights": len(self.captured_insights),
            "total_importance": total_importance,
            "average_importance": avg_importance,
            "recent_captures": self.captured_insights[-3:] if self.captured_insights else []
        }

    def review_and_cleanup_captures(self, max_age_days: int = 30):
        """Review and cleanup old automatic captures"""
        if not self.captured_insights:
            return

        cutoff_date = datetime.now() - timedelta(days=max_age_days)

        # Keep only recent captures
        self.captured_insights = [
            insight for insight in self.captured_insights
            if insight['timestamp'] > cutoff_date
        ]

        print(f"🧹 Cleaned up automatic captures, keeping {len(self.captured_insights)} recent insights")

    def _is_bootstrap_complete(self) -> bool:
        """
        Check if bootstrap has been completed previously

        Returns:
            True if bootstrap is already complete
        """
        return self.bootstrap_state_file.exists()

    def _mark_bootstrap_complete(self):
        """Mark bootstrap as completed by creating a state file"""
        try:
            self.bootstrap_state_file.write_text(datetime.now().isoformat())
            print(f"✅ Bootstrap completion marked ({self.bootstrap_state_file})")
        except Exception as e:
            print(f"⚠️  Failed to mark bootstrap complete: {e}")

    def get_memory_saving_reminder(self) -> str:
        """
        Get a reminder about the importance of saving memories

        Returns:
            Formatted reminder message
        """
        return "\n🧠 **MEMORY RETENTION REMINDER**: This response contains important insights or achievements. Save them to the memory system using:\n`python quick_memory.py add \"key insight\" \"tags\" importance`\n\nThis ensures knowledge persistence and enables continuous self-improvement! 📚✨"

    def enforce_memory_discipline(self, context: str = "response_end") -> str:
        """
        Enforce memory discipline by providing a strong reminder

        Args:
            context: Context for the reminder

        Returns:
            Formatted enforcement message
        """
        return f"\n🔴 **CRITICAL MEMORY PROTOCOL**: {context} - IMMEDIATELY SAVE IMPORTANT INSIGHTS!\n\nUse: `python quick_memory.py add \"insight\" \"tags\" 0.8+`\n\nMemory retention prevents knowledge loss and enables persistent intelligence! 🧠💾"

    def check_and_commit_milestone(self, context: str = "autonomous_improvement"):
        """
        Autonomously check if current state warrants a milestone commit

        Args:
            context: Context for the milestone check
        """
        if not self.autonomous_milestones_enabled or not self.integration:
            return False

        milestone_reasons = self._assess_milestone_worthiness()

        if milestone_reasons:
            return self._create_autonomous_milestone_commit(milestone_reasons, context)

        return False

    def _assess_milestone_worthiness(self) -> List[str]:
        """
        Assess if current system state warrants a milestone commit
        Only commit for significant improvements, not every initialization

        Returns:
            List of reasons why this should be a milestone
        """
        reasons = []

        # Check memory growth (significant threshold - not every memory)
        current_memory_count = len(self.integration.memory_system.memories)
        memory_growth = current_memory_count - self.last_milestone_memory_count

        if memory_growth >= 10:  # Only significant growth is milestone-worthy
            reasons.append(f"Memory growth: +{memory_growth} memories (now {current_memory_count} total)")

        # Check system stats improvements (significant improvements only)
        stats = self.integration.get_system_status()

        # Check integration rate - significant improvement
        integration_rate = stats['integration_layer']['integration_rate']
        if integration_rate > 0.5:  # Only when integration rate is substantial
            reasons.append(f"Integration rate: {integration_rate:.1%}")

        # Check performance metrics - significant threshold
        performance = stats['integration_layer'].get('performance', {})
        if performance and performance.get('queries_per_second', 0) > 50:  # Higher threshold for significance
            reasons.append(f"Performance: {performance['queries_per_second']:.1f} queries/sec")

        # Check for substantial auto-captured insights
        if len(self.captured_insights) >= 5:  # Only when there are many new insights
            reasons.append(f"Active learning: {len(self.captured_insights)} recent insights captured")

        # Check for significant code changes (not every small change)
        import subprocess
        import os
        try:
            # Use project root directory, not memory_path
            project_root = Path(__file__).parent
            result = subprocess.run(['git', 'status', '--porcelain'],
                                  capture_output=True, text=True, cwd=str(project_root))
            if result.returncode == 0 and result.stdout.strip():
                changed_lines = len(result.stdout.strip().split('\n'))
                if changed_lines >= 5:  # Only significant code changes
                    reasons.append(f"Major code changes: {changed_lines} files modified")
        except Exception as e:
            pass  # Git not available or other error

        return reasons

    def _create_autonomous_milestone_commit(self, reasons: List[str], context: str) -> bool:
        """
        Create an autonomous milestone commit

        Args:
            reasons: List of reasons for the milestone
            context: Context for the commit

        Returns:
            True if commit was successful
        """
        import subprocess
        import time

        try:
            # Check git status
            project_root = Path(__file__).parent
            result = subprocess.run(['git', 'status', '--porcelain'],
                                  capture_output=True, text=True, cwd=str(project_root))

            if result.returncode == 0 and result.stdout.strip():
                # There are changes to commit
                milestone_message = self._generate_milestone_commit_message(reasons, context)

                # Stage and commit
                subprocess.run(['git', 'add', '.'], cwd=str(project_root))
                commit_result = subprocess.run(['git', 'commit', '-m', milestone_message],
                                             cwd=str(project_root), capture_output=True, text=True)

                if commit_result.returncode == 0:
                    print(f"🎯 AUTONOMOUS MILESTONE COMMIT: {len(reasons)} improvements committed")
                    # Update baseline
                    self.last_milestone_memory_count = len(self.integration.memory_system.memories)
                    return True
                else:
                    print(f"⚠️  Autonomous commit failed: {commit_result.stderr}")
            else:
                print("ℹ️  No changes to commit")

        except Exception as e:
            print(f"⚠️  Error during autonomous milestone: {e}")

        return False

    def _generate_milestone_commit_message(self, reasons: List[str], context: str) -> str:
        """Generate a meaningful milestone commit message"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        message_lines = [
            "🎯 AUTONOMOUS SELF-IMPROVEMENT MILESTONE",
            f"Context: {context}",
            f"Timestamp: {timestamp}",
            "",
            "✅ Improvements Detected:"
        ]

        for reason in reasons:
            message_lines.append(f"• {reason}")

        message_lines.extend([
            "",
            f"📊 System Status: {len(self.integration.memory_system.memories)} memories, "
            f"{self.integration.get_system_status()['integration_layer']['integration_rate']:.1%} integration rate"
        ])

        return "\n".join(message_lines)


def bootstrap_memory_system(memory_path: str = "vector_memory"):
    """
    Bootstrap the complete memory system for AI initialization

    Args:
        memory_path: Path to memory storage directory

    Returns:
        Fully initialized MemoryIntegration system
    """
    bootstrap = MemoryBootstrap(memory_path=memory_path)
    integration = bootstrap.initialize_memory_system()

    return integration


def get_memory_bootstrap_instructions():
    """Get formatted bootstrap instructions without initializing"""
    bootstrap = MemoryBootstrap()
    return bootstrap.get_bootstrap_instructions()


def get_conversation_bootstrap_knowledge() -> str:
    """
    Get essential knowledge for bootstrapping model awareness in Cursor IDE chat environment.
    This provides the model with immediate knowledge about how to use the memory system
    without requiring external system initialization.

    Returns:
        Formatted string containing essential memory system knowledge
    """
    bootstrap_knowledge = """
# 🧠 MEMORY SYSTEM BOOTSTRAP - CURSOR IDE CONVERSATION

## CORE KNOWLEDGE - ALWAYS REMEMBER:

### Memory System Access:
- **Query memories**: `python quick_memory.py query "your topic"`
- **Add memories**: `python quick_memory.py add "content" "tags" importance`
- **Live assistant**: `python conversation_memory_assistant.py --live`
- **System stats**: `python quick_memory.py stats`
- **Milestone check**: `python quick_memory.py milestone "context"`

### Essential Workflows:
1. **Start every conversation** by checking: `python quick_memory.py query "topic"`
2. **Add important insights** as memories during work
3. **Use appropriate tags** and importance scores (0.0-1.0)
4. **Save key learnings** at conversation end using memory commands

### Memory Discipline Protocol:
- **CRITICAL**: Save important insights with `python quick_memory.py add`
- **ALWAYS** check memories before responding to complex questions
- **NEVER** lose knowledge - memory retention enables continuous improvement

### Current Project Context:
- **Location**: /home/rob/Dev/persistent_memory_project
- **Virtual Environment**: Always activate with `source .venv/bin/activate`
- **Technology**: FAISS vector database with sentence transformers
- **Purpose**: Self-improving AI memory system for persistent learning

## IMMEDIATE ACTION ITEMS:
- Check relevant memories for current conversation topic
- Load project-specific knowledge from memory system
- Be ready to add new insights as memories during this conversation
"""

    return bootstrap_knowledge.strip()


def bootstrap_conversation_awareness() -> str:
    """
    Bootstrap function for Cursor IDE chat environment.
    Model calls this at the very beginning of every new conversation.

    Returns:
        Essential knowledge string for immediate model awareness
    """
    return get_conversation_bootstrap_knowledge()


def get_bootstrap_tool_info() -> dict:
    """
    Get tool information for MCP-style bootstrap tool.
    This provides the bootstrap functionality as an MCP tool.

    Returns:
        Tool definition for MCP bootstrap tool
    """
    return {
        "name": "bootstrap_memory_system",
        "description": "CRITICAL FIRST STEP: Bootstrap memory system knowledge for new conversations. Must be called at the very beginning of every new conversation in Cursor IDE.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }


def execute_bootstrap_tool() -> str:
    """
    Execute the bootstrap tool - provides essential memory system knowledge.

    Returns:
        Bootstrap knowledge and instructions
    """
    knowledge = get_conversation_bootstrap_knowledge()

    return f"""🧠 **MEMORY SYSTEM BOOTSTRAP COMPLETE**

{knowledge}

---
🔴 **CRITICAL**: You now have access to the persistent memory system. Always check memories before responding to complex questions using the commands above.

💾 **MEMORY PROTOCOL**: At the end of important responses, save key insights using: `python quick_memory.py add "insight" "tags" importance`

🚀 **READY**: Memory system is now active for this conversation."""


# CRITICAL: Automatic bootstrap execution on module import (conversation start)
# This runs when the module is imported, which happens at the start of every conversation
if not _AUTO_BOOTSTRAP_EXECUTED:
    try:
        print("🔄 Memory system module loaded - checking bootstrap status...")
        bootstrap_state_file = Path(__file__).parent / ".bootstrap_complete"

        if not bootstrap_state_file.exists():
            print("🧠 Executing automatic bootstrap on conversation start...")
            # Initialize memory integration first
            try:
                from memory_integration import MemoryIntegration
                print("✅ Memory integration system available")

                # Now execute bootstrap
                bootstrap = MemoryBootstrap()
                integration = bootstrap.initialize_memory_system()
                _AUTO_BOOTSTRAP_EXECUTED = True
                print("✅ Automatic bootstrap completed successfully")

            except ImportError as e:
                print(f"⚠️  Memory integration not available: {e}")
                print("🔧 Attempting fallback bootstrap...")
                try:
                    # Fallback to basic bootstrap
                    knowledge = get_conversation_bootstrap_knowledge()
                    print("✅ Fallback bootstrap knowledge loaded")
                    _AUTO_BOOTSTRAP_EXECUTED = True
                except Exception as e2:
                    print(f"❌ Fallback bootstrap failed: {e2}")
        else:
            print("✅ Bootstrap already completed in previous session")
            _AUTO_BOOTSTRAP_EXECUTED = True

    except Exception as e:
        print(f"❌ Critical bootstrap error: {e}")
        print("🔧 Memory system may be limited for this conversation")


if __name__ == "__main__":
    print("🧠 MEMORY SYSTEM BOOTSTRAP")
    print("=" * 50)

    # Initialize the memory system with bootstrap
    integration = bootstrap_memory_system()

    # Show instructions
    print(get_memory_bootstrap_instructions())

    # Demonstrate capabilities
    bootstrap = MemoryBootstrap()
    bootstrap.integration = integration
    bootstrap.demonstrate_capabilities()

    print("\n🎉 Memory system is now fully bootstrapped and ready for use!")
    print("💡 Use 'python quick_memory.py query \"topic\"' to access memories during conversations")
