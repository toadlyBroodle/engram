#!/usr/bin/env python3
"""
🪝 CURSOR IDE MEMORY HOOK
==========================

This hook automatically tracks every conversation message in Cursor IDE
without any manual intervention. It integrates seamlessly with Cursor IDE
chat windows and ensures continuous memory building.

USAGE:
- Cursor IDE calls this automatically for every message
- No manual memory storage required
- Works transparently in background
"""

import sys
import os
from pathlib import Path
import json
from datetime import datetime

# Ensure we're in the project directory
project_dir = Path(__file__).parent
os.chdir(project_dir)
sys.path.insert(0, str(project_dir))

def track_cursor_message(role: str, content: str, conversation_id: str = None):
    """
    Track a message from Cursor IDE conversation.

    Args:
        role: 'user' or 'assistant'
        content: Message content
        conversation_id: Optional conversation identifier
    """
    try:
        # Import conversation tracker
        from conversation_tracker import track_conversation_message

        # Add Cursor IDE metadata
        metadata = {
            'source': 'cursor_ide',
            'conversation_id': conversation_id or 'cursor_chat',
            'timestamp': datetime.now().isoformat(),
            'auto_tracked': True
        }

        # Track the message
        track_conversation_message(role, content, metadata)

        # Log success (quietly)
        print(f"✅ Cursor IDE message tracked: {role} ({len(content)} chars)", file=sys.stderr)

        return True

    except Exception as e:
        # Log error but don't break Cursor IDE
        print(f"⚠️  Cursor memory hook failed: {e}", file=sys.stderr)
        return False

def get_cursor_memory_status():
    """Get memory system status for Cursor IDE"""
    try:
        from conversation_tracker import get_tracking_status
        status = get_tracking_status()

        # Format for Cursor IDE
        return {
            'memory_system_active': status.get('status') == 'active',
            'messages_tracked': status.get('messages_tracked', 0),
            'auto_captures': status.get('auto_capture_stats', {}).get('captured_insights', 0),
            'last_activity': status.get('last_activity')
        }
    except Exception as e:
        return {'error': str(e), 'memory_system_active': False}

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python cursor_memory_hook.py <role> <content> [conversation_id]", file=sys.stderr)
        sys.exit(1)

    role = sys.argv[1]
    content = sys.argv[2]
    conversation_id = sys.argv[3] if len(sys.argv) > 3 else None

    success = track_cursor_message(role, content, conversation_id)
    sys.exit(0 if success else 1)
