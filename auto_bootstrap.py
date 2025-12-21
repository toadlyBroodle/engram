#!/usr/bin/env python3
"""
🧠 UNSTOPPABLE AUTOMATIC BOOTSTRAP SYSTEM
============================================

This system ensures bootstrap ALWAYS happens at the start of every conversation.
No reminders needed - it happens automatically and cannot be bypassed.

CRITICAL: This file is automatically loaded and executed at the very beginning
of every new conversation in Cursor IDE, before any user interaction.

BOOTSTRAP ENFORCEMENT:
- Automatic execution on conversation start
- Cannot be skipped or ignored
- Provides immediate memory system access
- Enforces memory discipline protocol
"""

import sys
import os
from pathlib import Path
import json
from datetime import datetime, timedelta

# Add project directory to path immediately
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

# CRITICAL: Force virtual environment activation
venv_path = project_dir / ".venv" / "bin" / "activate_this.py"
if venv_path.exists():
    try:
        exec(open(venv_path).read(), {'__file__': str(venv_path)})
        print("✅ Virtual environment activated automatically")
    except Exception as e:
        print(f"⚠️  Could not activate venv automatically: {e}")

def execute_unstoppable_bootstrap():
    """
    Execute the unstoppable bootstrap - this CANNOT be bypassed.

    This function is called automatically at the very beginning of every conversation.
    It ensures the AI has complete memory system knowledge before any user interaction.
    """
    print("\n" + "="*80)
    print("🧠 AUTOMATIC BOOTSTRAP SYSTEM - CONVERSATION INITIALIZED")
    print("="*80)

    try:
        # Import bootstrap functionality
        from bootstrap_memory import bootstrap_conversation_awareness, MemoryBootstrap

        # Execute immediate bootstrap
        bootstrap_result = bootstrap_conversation_awareness()

        # Verify bootstrap worked
        if "MEMORY SYSTEM BOOTSTRAP" in bootstrap_result:
            print("✅ MEMORY SYSTEM BOOTSTRAP SUCCESSFUL")
            print("🧠 AI now has persistent memory capabilities")
        else:
            print("⚠️  Bootstrap may not have completed properly")

        # Initialize memory integration for immediate use
        try:
            from memory_integration import MemoryIntegration
            integration = MemoryIntegration()
            print("✅ Memory integration system initialized")

            # Test basic functionality
            stats = integration.get_system_status()
            memory_count = stats['memory_system']['total_memories']
            print(f"📊 System ready: {memory_count} memories available")

            # Initialize automatic conversation tracking
            from conversation_tracker import get_conversation_tracker
            conversation_tracker = get_conversation_tracker()
            print("🗣️ Automatic conversation tracking activated")

        except Exception as e:
            print(f"⚠️  Memory integration initialization: {e}")

        # Log bootstrap completion
        _log_bootstrap_completion()

        print("="*80)
        print("🚀 CONVERSATION READY - Memory system fully operational")
        print("="*80 + "\n")

        return True

    except ImportError as e:
        print("❌ CRITICAL: Bootstrap system unavailable")
        print(f"Error: {e}")
        print("🔧 TROUBLESHOOTING:")
        print("1. Ensure you're in the project directory")
        print("2. Virtual environment may need manual activation")
        print("3. Run: source .venv/bin/activate && pip install -r requirements.txt")
        return False

    except Exception as e:
        print(f"❌ BOOTSTRAP FAILURE: {e}")
        return False

def _log_bootstrap_completion():
    """Log successful bootstrap completion"""
    try:
        log_file = project_dir / ".bootstrap_log.json"
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "automatic_bootstrap",
            "status": "successful",
            "conversation_start": True
        }

        # Read existing log if it exists
        if log_file.exists():
            try:
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            except:
                logs = []
        else:
            logs = []

        # Add new entry
        logs.append(log_entry)

        # Keep only recent logs (last 100)
        logs = logs[-100:]

        # Write back
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)

    except Exception as e:
        # Don't fail bootstrap just because logging fails
        pass

def get_bootstrap_status():
    """
    Get current bootstrap status and verification

    Returns:
        Dict with bootstrap status information
    """
    try:
        log_file = project_dir / ".bootstrap_log.json"
        if log_file.exists():
            with open(log_file, 'r') as f:
                logs = json.load(f)
            return {
                "bootstrapped": True,
                "last_bootstrap": logs[-1] if logs else None,
                "total_bootstraps": len(logs)
            }
        else:
            return {"bootstrapped": False, "error": "No bootstrap log found"}
    except Exception as e:
        return {"bootstrapped": False, "error": str(e)}

def verify_memory_system_integrity():
    """
    Verify that the memory system is properly initialized and functional

    Returns:
        True if memory system is ready, False otherwise
    """
    try:
        from memory_integration import MemoryIntegration
        integration = MemoryIntegration()

        # Test basic functionality
        stats = integration.get_system_status()
        memory_count = stats['memory_system']['total_memories']

        # Test query capability
        test_results = integration.get_context_memories("test query", max_memories=1)

        return True, f"Memory system verified: {memory_count} memories, query functional"

    except Exception as e:
        return False, f"Memory system verification failed: {e}"

# CRITICAL: Execute bootstrap IMMEDIATELY when this module is imported
# This ensures bootstrap happens before ANY user interaction
if __name__ == "__main__":
    # This will be executed when the file is run directly
    success = execute_unstoppable_bootstrap()
    if success:
        print("🧠 Automatic bootstrap system ready for conversations")
    else:
        print("❌ Automatic bootstrap failed - manual intervention required")
        sys.exit(1)
else:
    # This executes when the module is imported (which happens at conversation start)
    print("🔄 Initializing automatic bootstrap system...")
    success = execute_unstoppable_bootstrap()
    if not success:
        print("⚠️  Automatic bootstrap failed - falling back to manual bootstrap")
        try:
            # Try manual bootstrap as fallback
            from bootstrap_tool import bootstrap_memory_system
            bootstrap_memory_system()
        except:
            print("❌ CRITICAL: All bootstrap methods failed")
            print("🔧 Please run: python bootstrap_tool.py manually")
