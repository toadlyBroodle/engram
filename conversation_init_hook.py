#!/usr/bin/env python3
"""
🧠 CONVERSATION INITIALIZATION HOOK
===================================

This script is automatically loaded by Cursor IDE at the very beginning of every conversation.
It ensures bootstrap happens before any user interaction can occur.

CRITICAL: This cannot be bypassed - it runs before the conversation even starts.
"""

import sys
import os
from pathlib import Path

# Immediate path setup
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

def initialize_conversation():
    """
    Initialize conversation with automatic bootstrap.

    This function is called by Cursor IDE before any conversation begins.
    """
    print("\n🔄 CONVERSATION INITIALIZING...")
    print("🧠 Loading automatic memory bootstrap system...")

    try:
        # Import and execute auto bootstrap
        import auto_bootstrap
        success = auto_bootstrap.execute_unstoppable_bootstrap()

        if success:
            print("✅ Conversation bootstrap complete - memory system ready")
            return True
        else:
            print("❌ Conversation bootstrap failed")
            print("🔧 Memory system may be limited for this conversation")
            return False

    except Exception as e:
        print(f"❌ Critical error during conversation initialization: {e}")
        print("🔧 Attempting fallback bootstrap...")
        try:
            # Fallback to manual bootstrap
            from bootstrap_tool import bootstrap_memory_system
            bootstrap_memory_system()
            print("✅ Fallback bootstrap successful")
            return True
        except Exception as e2:
            print(f"❌ All bootstrap methods failed: {e2}")
            return False

# Execute immediately on import (which happens at conversation start)
if __name__ == "__main__":
    # This runs when called directly
    initialize_conversation()
else:
    # This runs when imported (conversation initialization)
    print("🎯 Cursor IDE Conversation Hook Activated")
    init_success = initialize_conversation()
    if init_success:
        print("🚀 Conversation ready with memory capabilities")
    else:
        print("⚠️  Conversation initialized with limited memory access")
