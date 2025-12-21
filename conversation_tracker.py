#!/usr/bin/env python3
"""
🗣️ AUTOMATIC CONVERSATION MEMORY TRACKER
==========================================

This system automatically tracks all conversations and captures insights
without any manual intervention required. It integrates seamlessly into
the AI's workflow and ensures continuous memory building.

FEATURES:
- Automatic conversation logging
- Real-time insight capture
- Seamless integration with memory system
- No manual prompts required
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

# Add project directory to path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

class ConversationMemoryTracker:
    """
    Automatically tracks conversations and captures memories in real-time.

    This class ensures that every conversation is automatically stored
    and analyzed for important insights without manual intervention.
    """

    def __init__(self):
        self.memory_integration = None
        self.conversation_buffer = []
        self.session_start = datetime.now()
        self._initialize_memory_system()

    def _initialize_memory_system(self):
        """Initialize memory system connection"""
        try:
            from memory_integration import MemoryIntegration
            self.memory_integration = MemoryIntegration()
            print("🧠 Conversation memory tracker initialized")
        except Exception as e:
            print(f"⚠️  Memory system initialization failed: {e}")
            self.memory_integration = None

    def track_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Track a conversation message automatically.

        Args:
            role: Message role ('user', 'assistant', 'system')
            content: Message content
            metadata: Optional metadata about the message
        """
        if not self.memory_integration:
            return

        try:
            # Track in conversation buffer
            message_entry = {
                'role': role,
                'content': content,
                'timestamp': datetime.now(),
                'metadata': metadata or {}
            }
            self.conversation_buffer.append(message_entry)

            # Update memory integration (this triggers auto-capture)
            self.memory_integration.update_conversation(role, content, metadata)

            # Log successful tracking
            print(f"📝 Auto-tracked {role} message ({len(content)} chars)")

        except Exception as e:
            print(f"⚠️  Message tracking failed: {e}")

    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of current conversation tracking"""
        if not self.memory_integration:
            return {"status": "memory_system_unavailable"}

        try:
            # Get memory system stats
            system_stats = self.memory_integration.get_system_status()
            capture_stats = self.memory_integration.get_auto_capture_statistics()

            return {
                "status": "active",
                "messages_tracked": len(self.conversation_buffer),
                "session_duration": str(datetime.now() - self.session_start),
                "memory_system_stats": system_stats,
                "auto_capture_stats": capture_stats,
                "last_activity": datetime.now().isoformat()
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def force_save_pending_memories(self):
        """Force save any pending memory changes"""
        if self.memory_integration:
            try:
                self.memory_integration.memory_system.force_save()
                print("💾 Forced save of pending memories")
            except Exception as e:
                print(f"⚠️  Force save failed: {e}")

# Global tracker instance
_conversation_tracker = None

def get_conversation_tracker() -> ConversationMemoryTracker:
    """Get the global conversation tracker instance"""
    global _conversation_tracker
    if _conversation_tracker is None:
        _conversation_tracker = ConversationMemoryTracker()
    return _conversation_tracker

def track_conversation_message(role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
    """
    Convenience function to track a conversation message.

    This should be called automatically for every conversation message.
    """
    tracker = get_conversation_tracker()
    tracker.track_message(role, content, metadata)

def get_tracking_status() -> Dict[str, Any]:
    """Get current conversation tracking status"""
    tracker = get_conversation_tracker()
    return tracker.get_conversation_summary()

# Initialize tracker on module import
try:
    _conversation_tracker = ConversationMemoryTracker()
    print("✅ Automatic conversation tracking activated")
except Exception as e:
    print(f"⚠️  Conversation tracking initialization failed: {e}")

if __name__ == "__main__":
    # Test the tracker
    print("🧠 **CONVERSATION MEMORY TRACKER TEST**")
    print("=" * 45)

    tracker = get_conversation_tracker()
    status = tracker.get_conversation_summary()

    print("📊 Tracking Status:")
    for key, value in status.items():
        if key != "memory_system_stats" and key != "auto_capture_stats":
            print(f"  {key}: {value}")

    print("\n🧪 Testing message tracking...")
    tracker.track_message("user", "This is a test message that should be automatically captured")
    tracker.track_message("assistant", "This response should also be captured with high importance")

    # Check if captures happened
    status_after = tracker.get_conversation_summary()
    print(f"Messages tracked: {status_after.get('messages_tracked', 0)}")

    print("\n✅ Conversation tracking test complete")
