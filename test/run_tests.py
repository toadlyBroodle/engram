#!/usr/bin/env python3
"""
Test Runner for Persistent Memory System

Runs all test suites in the correct order with proper environment setup.
"""

import sys
import subprocess
import os
from pathlib import Path

def run_test(test_file, description):
    """Run a single test file"""
    print(f"\n🧪 Running {description}...")
    print("-" * 50)

    project_dir = Path(__file__).parent.parent
    cmd = [sys.executable, str(test_file)]

    result = subprocess.run(
        cmd,
        cwd=str(project_dir),
        env={**os.environ, "PYTHONPATH": str(project_dir)}
    )

    if result.returncode == 0:
        print(f"✅ {description} PASSED")
        return True
    else:
        print(f"❌ {description} FAILED (exit code: {result.returncode})")
        return False

def main():
    """Run all tests"""
    print("🧠 PERSISTENT MEMORY SYSTEM - COMPREHENSIVE TEST SUITE")
    print("=" * 60)

    test_dir = Path(__file__).parent
    tests = [
        (test_dir / "test_auto_bootstrap.py", "Auto Bootstrap Tests"),
        (test_dir / "test_quick_memory_cli.py", "CLI Interface Tests"),
        (test_dir / "test_memory_integration.py", "Memory Integration Tests"),
    ]

    passed = 0
    total = len(tests)

    for test_file, description in tests:
        if run_test(test_file, description):
            passed += 1

    print("\n" + "=" * 60)
    print(f"🧠 TEST RESULTS: {passed}/{total} test suites passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED! Memory system is healthy.")
        return 0
    else:
        print("❌ SOME TESTS FAILED. Please review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
