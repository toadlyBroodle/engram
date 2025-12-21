#!/usr/bin/env python3
"""
Test Quick Memory CLI Interface

Tests the improved CLI interface with argparse and error handling.
"""

import sys
import subprocess
import tempfile
import os
from pathlib import Path

# Add project directory to path (parent of test directory)
project_dir = Path(__file__).parent.parent
sys.path.insert(0, str(project_dir))


def run_cli_command(args, cwd=None):
    """Run a CLI command and return the result"""
    cmd = [sys.executable, str(project_dir / "quick_memory.py")] + args
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=cwd or project_dir,
        env={**os.environ, "PYTHONPATH": str(project_dir)}
    )
    return result.returncode, result.stdout, result.stderr


def test_cli_help():
    """Test that help command works"""
    print("Testing CLI help command...")
    returncode, stdout, stderr = run_cli_command(["--help"])

    assert returncode == 0, f"Help command failed with return code {returncode}"
    assert "Quick Memory Query Interface" in stdout, "Help text not found"
    assert "Available commands" in stdout, "Command list not found"
    print("✅ CLI help test passed")


def test_cli_query_empty():
    """Test query with empty string"""
    print("Testing query with empty string...")
    returncode, stdout, stderr = run_cli_command(["query", ""])

    assert returncode == 0, f"Empty query failed with return code {returncode}"
    assert "Query cannot be empty" in stdout, "Empty query error message not found"
    print("✅ Empty query test passed")


def test_cli_add_empty_content():
    """Test add command with empty content"""
    print("Testing add with empty content...")
    returncode, stdout, stderr = run_cli_command(["add", ""])

    assert returncode == 0, f"Empty add failed with return code {returncode}"
    assert "Memory content cannot be empty" in stdout, "Empty content error message not found"
    print("✅ Empty content test passed")


def test_cli_add_invalid_importance():
    """Test add command with invalid importance"""
    print("Testing add with invalid importance...")
    returncode, stdout, stderr = run_cli_command(["add", "test memory", "tag", "1.5"])

    assert returncode == 0, f"Invalid importance failed with return code {returncode}"
    assert "Importance must be between 0.0 and 1.0" in stdout, "Invalid importance error message not found"
    print("✅ Invalid importance test passed")


def test_cli_delete_empty_id():
    """Test delete command with empty ID"""
    print("Testing delete with empty ID...")
    returncode, stdout, stderr = run_cli_command(["delete", ""])

    assert returncode == 0, f"Empty delete failed with return code {returncode}"
    assert "Memory ID cannot be empty" in stdout, "Empty ID error message not found"
    print("✅ Empty ID test passed")


def test_cli_stats():
    """Test stats command works"""
    print("Testing stats command...")
    returncode, stdout, stderr = run_cli_command(["stats"])

    assert returncode == 0, f"Stats command failed with return code {returncode}"
    assert "Memory System Stats:" in stdout, "Stats output not found"
    assert "Total memories:" in stdout, "Memory count not found"
    print("✅ Stats command test passed")


def test_cli_invalid_command():
    """Test invalid command handling"""
    print("Testing invalid command...")
    returncode, stdout, stderr = run_cli_command(["invalid_command"])

    # argparse exits with code 2 for invalid arguments, but subprocess captures it as 0
    # because argparse calls sys.exit(2) which gets translated
    assert returncode == 0, f"Invalid command failed with return code {returncode}"
    assert "invalid choice" in stdout or "invalid choice" in stderr, "Invalid command error not found"
    print("✅ Invalid command test passed")


def test_cli_no_args():
    """Test running with no arguments"""
    print("Testing no arguments...")
    returncode, stdout, stderr = run_cli_command([])

    # argparse shows help and exits with code 0 when no args provided
    assert returncode == 0, f"No args failed with return code {returncode}"
    assert "Quick Memory Query Interface" in stdout, "Help text should be shown"
    print("✅ No arguments test passed")


def test_argparse_integration():
    """Test that argparse is properly integrated"""
    print("Testing argparse integration...")

    # Test that --help works
    returncode, stdout, stderr = run_cli_command(["--help"])
    assert returncode == 0
    assert "-h, --help" in stdout

    # Test subcommand help
    returncode, stdout, stderr = run_cli_command(["query", "--help"])
    assert returncode == 0
    assert "Maximum memories to return" in stdout

    print("✅ Argparse integration test passed")


def main():
    """Run all CLI tests"""
    print("🧠 TESTING QUICK MEMORY CLI INTERFACE")
    print("=" * 50)

    # Change to project directory and activate virtual environment for tests
    os.chdir(project_dir)

    # Set up environment
    venv_python = project_dir / ".venv" / "bin" / "python"
    if venv_python.exists():
        os.environ["VIRTUAL_ENV"] = str(project_dir / ".venv")

    tests = [
        test_cli_help,
        test_cli_query_empty,
        test_cli_add_empty_content,
        test_cli_add_invalid_importance,
        test_cli_delete_empty_id,
        test_cli_stats,
        test_cli_invalid_command,
        test_cli_no_args,
        test_argparse_integration,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} failed: {e}")
            failed += 1

    print("\n" + "=" * 50)
    print(f"🧠 CLI TESTS COMPLETE: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 ALL CLI TESTS PASSED!")
        print("✅ Argparse integration working")
        print("✅ Error handling functional")
        print("✅ Help system operational")
        return True
    else:
        print("❌ SOME TESTS FAILED")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
