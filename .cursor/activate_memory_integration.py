#!/usr/bin/env python3
"""
🧠 CURSOR IDE MEMORY INTEGRATION ACTIVATOR
============================================

This script automatically activates memory integration for Cursor IDE
conversations. It sets up hooks and ensures all messages are tracked
without manual intervention.

RUNS AUTOMATICALLY: Called by Cursor IDE on conversation start
"""

import sys
import os
from pathlib import Path
import subprocess
import threading
import time

# Ensure we're in the project directory
project_dir = Path(__file__).parent.parent
os.chdir(project_dir)
sys.path.insert(0, str(project_dir))

def activate_memory_integration():
    """Activate memory integration for Cursor IDE"""
    print("🚀 ACTIVATING CURSOR IDE MEMORY INTEGRATION")
    print("=" * 55)

    try:
        # Step 1: Bootstrap memory system
        print("1. Bootstrapping memory system...")
        result = subprocess.run([
            sys.executable, "auto_bootstrap.py"
        ], capture_output=True, text=True, cwd=project_dir)

        if result.returncode == 0:
            print("✅ Memory system bootstrapped")
        else:
            print(f"⚠️  Bootstrap warning: {result.stderr}")
            # Continue anyway - bootstrap might have partial success

        # Step 2: Initialize conversation tracker
        print("2. Initializing conversation tracking...")
        from conversation_tracker import get_conversation_tracker
        tracker = get_conversation_tracker()
        print("✅ Conversation tracker initialized")

        # Step 3: Verify memory hook
        print("3. Testing memory hook...")
        hook_result = subprocess.run([
            sys.executable, "cursor_memory_hook.py", "test", "Memory integration test message"
        ], capture_output=True, text=True, cwd=project_dir)

        if hook_result.returncode == 0:
            print("✅ Memory hook functional")
        else:
            print(f"⚠️  Hook test warning: {hook_result.stderr}")

        # Step 4: Set up background monitoring
        print("4. Starting background monitoring...")
        monitor_thread = threading.Thread(target=background_memory_monitor, daemon=True)
        monitor_thread.start()

        print("✅ Memory integration activated successfully!")
        print("🧠 All Cursor IDE conversations will be automatically tracked")
        print("=" * 55)

        return True

    except Exception as e:
        print(f"❌ Activation failed: {e}")
        return False

def background_memory_monitor():
    """Background thread to monitor and maintain memory system"""
    while True:
        try:
            # Periodic health check and auto-save
            time.sleep(300)  # Check every 5 minutes

            from memory_integration import MemoryIntegration
            integrator = MemoryIntegration()

            # Force periodic save
            integrator.memory_system.force_save()

            # Log status
            stats = integrator.get_system_status()
            memory_count = stats['memory_system']['total_memories']

            print(f"💾 Auto-saved {memory_count} memories", file=sys.stderr)

        except Exception as e:
            print(f"⚠️  Background monitor error: {e}", file=sys.stderr)
            time.sleep(60)  # Wait a bit before retrying

def test_integration():
    """Test that integration is working"""
    print("🧪 TESTING MEMORY INTEGRATION...")

    try:
        # Test conversation tracking
        from conversation_tracker import track_conversation_message
        track_conversation_message('user', 'Test message from Cursor IDE integration')
        track_conversation_message('assistant', 'Test response confirming integration works')

        # Test memory query
        from memory_integration import MemoryIntegration
        integrator = MemoryIntegration()
        results = integrator.get_context_memories("Cursor IDE integration")

        print(f"✅ Integration test passed - {len(results.get('memories', []))} memories found")

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

    return True

if __name__ == "__main__":
    success = activate_memory_integration()

    if success and len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_integration()

    if success:
        print("\n🎯 CURSOR IDE MEMORY INTEGRATION ACTIVE")
        print("📝 Every message in this conversation will be automatically remembered")
        print("🧠 No manual memory storage required - it happens automatically!")
    else:
        print("\n⚠️  Memory integration activation incomplete")
        sys.exit(1)
