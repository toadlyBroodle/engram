"""
Token and Cost Tracker for Engram

Tracks token usage and estimated costs for:
- Brain: Main LLM (chat responses)
- MemMan: Memory extraction LLM

Usage:
    from token_tracker import tracker
    
    # After LLM call:
    tracker.record_brain_usage(response.usage_metadata)
    tracker.record_memman_usage(response.usage_metadata)
    
    # Get stats:
    stats = tracker.get_stats()
    print(tracker.format_stats())
"""

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


# Model pricing per 1M tokens (as of Dec 2024)
# https://ai.google.dev/pricing
MODEL_PRICING = {
    # Gemini 2.0
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30},
    "gemini-2.0-flash-thinking": {"input": 0.70, "output": 2.80},
    # Gemini 1.5
    "gemini-1.5-pro": {"input": 2.50, "output": 10.00},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-flash-8b": {"input": 0.0375, "output": 0.15},
    # Default fallback
    "default": {"input": 0.10, "output": 0.40},
}


@dataclass
class ComponentStats:
    """Token stats for a single component (Brain or MemMan)"""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    call_count: int = 0
    estimated_cost: float = 0.0
    model: str = ""
    
    def add(self, input_tokens: int, output_tokens: int, model: str = ""):
        """Add token counts from a single call"""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_tokens += input_tokens + output_tokens
        self.call_count += 1
        if model:
            self.model = model
        
        # Calculate cost
        pricing = MODEL_PRICING.get(model, MODEL_PRICING["default"])
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        self.estimated_cost += input_cost + output_cost
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "call_count": self.call_count,
            "estimated_cost_usd": round(self.estimated_cost, 6),
            "model": self.model,
        }


class TokenTracker:
    """
    Thread-safe token and cost tracker for Engram components.
    
    Tracks:
    - Brain: Main chat/response LLM
    - MemMan: Memory extraction LLM
    """
    
    def __init__(self):
        self._lock = threading.Lock()
        self._brain = ComponentStats()
        self._memman = ComponentStats()
        self._session_start = datetime.now()
        self._log_path: Optional[Path] = None
    
    def set_log_path(self, path: str):
        """Set path for persistent logging"""
        self._log_path = Path(path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def record_brain_usage(self, usage_metadata, model: str = ""):
        """Record token usage from a Brain (chat) LLM call"""
        if not usage_metadata:
            return
        
        input_tokens = getattr(usage_metadata, 'prompt_token_count', 0) or 0
        output_tokens = getattr(usage_metadata, 'candidates_token_count', 0) or 0
        
        with self._lock:
            self._brain.add(input_tokens, output_tokens, model)
        
        self._log_usage("brain", input_tokens, output_tokens, model)
    
    def record_memman_usage(self, usage_metadata, model: str = ""):
        """Record token usage from a MemMan (extraction) LLM call"""
        if not usage_metadata:
            return
        
        input_tokens = getattr(usage_metadata, 'prompt_token_count', 0) or 0
        output_tokens = getattr(usage_metadata, 'candidates_token_count', 0) or 0
        
        with self._lock:
            self._memman.add(input_tokens, output_tokens, model)
        
        self._log_usage("memman", input_tokens, output_tokens, model)
    
    def _log_usage(self, component: str, input_tokens: int, output_tokens: int, model: str):
        """Log usage to file if path is set"""
        if not self._log_path:
            return
        
        try:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "component": component,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
            with open(self._log_path, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception:
            pass  # Don't fail on logging errors
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current stats as dictionary"""
        with self._lock:
            return {
                "session_start": self._session_start.isoformat(),
                "session_duration_seconds": (datetime.now() - self._session_start).total_seconds(),
                "brain": self._brain.to_dict(),
                "memman": self._memman.to_dict(),
                "total": {
                    "input_tokens": self._brain.input_tokens + self._memman.input_tokens,
                    "output_tokens": self._brain.output_tokens + self._memman.output_tokens,
                    "total_tokens": self._brain.total_tokens + self._memman.total_tokens,
                    "call_count": self._brain.call_count + self._memman.call_count,
                    "estimated_cost_usd": round(
                        self._brain.estimated_cost + self._memman.estimated_cost, 6
                    ),
                },
            }
    
    def format_stats(self, verbose: bool = True) -> str:
        """Format stats as human-readable string"""
        stats = self.get_stats()
        
        lines = [
            "═" * 50,
            "📊 TOKEN USAGE STATS",
            "═" * 50,
        ]
        
        # Brain stats
        b = stats["brain"]
        lines.append(f"🧠 Brain ({b['model'] or 'unknown'}):")
        lines.append(f"   Calls: {b['call_count']:,}")
        lines.append(f"   Tokens: {b['input_tokens']:,} in / {b['output_tokens']:,} out = {b['total_tokens']:,}")
        lines.append(f"   Cost: ${b['estimated_cost_usd']:.4f}")
        
        # MemMan stats
        m = stats["memman"]
        lines.append(f"💾 MemMan ({m['model'] or 'unknown'}):")
        lines.append(f"   Calls: {m['call_count']:,}")
        lines.append(f"   Tokens: {m['input_tokens']:,} in / {m['output_tokens']:,} out = {m['total_tokens']:,}")
        lines.append(f"   Cost: ${m['estimated_cost_usd']:.4f}")
        
        # Total
        t = stats["total"]
        lines.append("─" * 50)
        lines.append(f"📈 TOTAL:")
        lines.append(f"   Calls: {t['call_count']:,}")
        lines.append(f"   Tokens: {t['total_tokens']:,}")
        lines.append(f"   Cost: ${t['estimated_cost_usd']:.4f}")
        
        if verbose:
            duration = stats["session_duration_seconds"]
            if duration > 0:
                tokens_per_sec = t["total_tokens"] / duration
                lines.append(f"   Rate: {tokens_per_sec:.1f} tokens/sec")
        
        lines.append("═" * 50)
        
        return "\n".join(lines)
    
    def reset(self):
        """Reset all stats"""
        with self._lock:
            self._brain = ComponentStats()
            self._memman = ComponentStats()
            self._session_start = datetime.now()


# Global tracker instance
tracker = TokenTracker()


if __name__ == "__main__":
    # Demo
    print("Token Tracker Demo")
    print()
    
    # Simulate some usage
    class MockUsage:
        def __init__(self, input_t, output_t):
            self.prompt_token_count = input_t
            self.candidates_token_count = output_t
    
    tracker.record_brain_usage(MockUsage(100, 50), "gemini-2.0-flash-lite")
    tracker.record_brain_usage(MockUsage(200, 100), "gemini-2.0-flash-lite")
    tracker.record_memman_usage(MockUsage(500, 200), "gemini-2.0-flash-lite")
    tracker.record_memman_usage(MockUsage(400, 150), "gemini-2.0-flash-lite")
    
    print(tracker.format_stats())
    print()
    print("JSON stats:")
    print(json.dumps(tracker.get_stats(), indent=2))
