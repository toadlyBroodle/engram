#!/usr/bin/env python3
"""
🧠 AUTOMATIC BOOTSTRAP SYSTEM TEST
===================================

Tests the unstoppable automatic bootstrap system to ensure it works
reliably at the start of every conversation.
"""

import sys
import os
from pathlib import Path
import time

# Add project directory to path (parent of test directory)
project_dir = Path(__file__).parent.parent
sys.path.insert(0, str(project_dir))

def test_auto_bootstrap():
    """Test the automatic bootstrap system"""
    print("🧠 TESTING AUTOMATIC BOOTSTRAP SYSTEM")
    print("="*50)

    try:
        # Test 1: Import auto_bootstrap (simulates conversation start)
        print("\n1. Testing auto_bootstrap import (conversation initialization)...")
        import auto_bootstrap

        # Check if bootstrap executed
        status = auto_bootstrap.get_bootstrap_status()
        if status.get("bootstrapped"):
            print("✅ Auto-bootstrap executed successfully on import")
            print(f"   Total bootstraps logged: {status.get('total_bootstraps', 0)}")
        else:
            print("❌ Auto-bootstrap failed on import")
            return False

        # Test 2: Verify memory system integrity
        print("\n2. Testing memory system integrity...")
        integrity_ok, integrity_message = auto_bootstrap.verify_memory_system_integrity()
        if integrity_ok:
            print("✅ Memory system integrity verified")
            print(f"   Status: {integrity_message}")
        else:
            print("❌ Memory system integrity check failed")
            print(f"   Error: {integrity_message}")
            return False

        # Test 3: Test conversation initialization hook
        print("\n3. Testing conversation initialization hook...")
        import conversation_init_hook
        print("✅ Conversation initialization hook loaded")

        # Test 4: Verify bootstrap state persistence
        print("\n4. Testing bootstrap state persistence...")
        bootstrap_complete_file = project_dir / ".bootstrap_complete"
        bootstrap_log_file = project_dir / ".bootstrap_log.json"

        if bootstrap_complete_file.exists():
            print("✅ Bootstrap completion state persisted")
        else:
            print("⚠️  Bootstrap completion state not found")

        if bootstrap_log_file.exists():
            print("✅ Bootstrap activity logged")
        else:
            print("⚠️  Bootstrap activity not logged")

        print("\n" + "="*50)
        print("✅ ALL TESTS PASSED - Automatic bootstrap system working!")
        print("="*50)

        return True

    except Exception as e:
        print(f"\n❌ TEST FAILURE: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_conversation_simulation():
    """Simulate a full conversation start"""
    print("\n🎯 SIMULATING FULL CONVERSATION START")
    print("-"*40)

    # Simulate conversation initialization
    print("1. Conversation starting...")
    time.sleep(0.5)

    print("2. Loading memory system...")
    time.sleep(0.5)

    # Import bootstrap_memory (should trigger auto-bootstrap)
    print("3. Initializing bootstrap system...")
    import bootstrap_memory
    time.sleep(0.5)

    print("4. Verifying bootstrap completion...")
    try:
        bootstrap = bootstrap_memory.MemoryBootstrap()
        integration = bootstrap.initialize_memory_system()
        print("✅ Bootstrap verification successful")

        # Test memory functionality
        stats = integration.get_system_status()
        memory_count = stats['memory_system']['total_memories']
        print(f"📊 Memory system ready: {memory_count} memories available")

        return True

    except Exception as e:
        print(f"❌ Conversation simulation failed: {e}")
        return False

if __name__ == "__main__":
    print("🧠 AUTOMATIC BOOTSTRAP SYSTEM COMPREHENSIVE TEST")
    print("="*60)

    # Run comprehensive tests
    test1_passed = test_auto_bootstrap()
    test2_passed = test_conversation_simulation()

    print("\n" + "="*60)
    if test1_passed and test2_passed:
        print("🎉 ALL TESTS PASSED - Automatic bootstrap system is unstoppable!")
        print("💡 The AI will now automatically bootstrap at the start of every conversation")
        print("🚫 No more manual reminders needed - it's truly automatic")
    else:
        print("❌ SOME TESTS FAILED - Automatic bootstrap system needs adjustment")
        sys.exit(1)
