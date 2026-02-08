#!/usr/bin/env python3
"""
DL-From-Scratch Test Runner

Usage:
    python utils/test_runner.py --day 01
    python utils/test_runner.py --day 05 --level 02
    python utils/test_runner.py --verify-setup
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def verify_setup():
    """Verify environment is correctly configured."""
    print("🔍 Verifying setup...\n")

    # Check Python version
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor} detected")
    else:
        print(f"❌ Python 3.8+ required, found {version.major}.{version.minor}")
        return False

    # Check NumPy
    try:
        import numpy as np

        print(f"✅ NumPy {np.__version__} installed")
    except ImportError:
        print("❌ NumPy not installed. Run: pip install numpy")
        return False

    # Check Matplotlib
    try:
        import matplotlib

        print(f"✅ Matplotlib {matplotlib.__version__} installed")
    except ImportError:
        print("❌ Matplotlib not installed. Run: pip install matplotlib")
        return False

    # Check pytest
    try:
        import pytest

        print(f"✅ Pytest {pytest.__version__} installed")
    except ImportError:
        print(
            "⚠️  Pytest not installed. Run: pip install pytest (optional but recommended)"
        )

    print("\n✅ All systems go! Start with Topic 01.")
    return True


def find_topic_dir(day: int) -> Path:
    """Find the directory for a given topic number."""
    root = Path(__file__).parent.parent

    # Map day to module
    if day <= 3:
        module = "Module 00-Foundations"
    elif day <= 10:
        module = "Module 01-Neural-Network-Core"
    elif day <= 17:
        module = "Module 02-CNNs"
    elif day <= 24:
        module = "Module 03-RNNs-Sequences"
    else:
        module = "Module 04-Transformers-Production"

    # Find matching topic folder
    module_path = root / module
    if not module_path.exists():
        return None

    for folder in module_path.iterdir():
        if folder.is_dir() and folder.name.startswith(f"Topic {day:02d}"):
            return folder

    return None


def run_tests(day: int, level: int = None):
    """Run tests for a specific topic."""
    try:
        import pytest  # noqa: F401
    except ImportError:
        print("❌ Pytest is not installed. Run: pip install -r requirements.txt")
        return False

    topic_dir = find_topic_dir(day)

    if not topic_dir:
        print(f"❌ Topic {day:02d} not found")
        return False

    tests_dir = topic_dir / "tests"
    if not tests_dir.exists():
        print(f"❌ No tests found for Topic {day:02d}")
        return False

    print(f"🧪 Running tests for Topic {day:02d}...\n")
    print(f"📂 Location: {topic_dir}\n")

    test_files = ["test_basic.py", "test_edge.py", "test_stress.py"]
    results = {"passed": 0, "failed": 0, "skipped": 0}

    for test_file in test_files:
        test_path = tests_dir / test_file
        if not test_path.exists():
            print(f"⏭️  {test_file}: SKIPPED (not found)")
            results["skipped"] += 1
            continue

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", str(test_path), "-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )
            if result.returncode == 0:
                print(f"✅ {test_file}: PASSED")
                results["passed"] += 1
            else:
                print(f"❌ {test_file}: FAILED")
                if result.stdout:
                    print(
                        result.stdout[-500:]
                        if len(result.stdout) > 500
                        else result.stdout
                    )
                results["failed"] += 1
        except subprocess.TimeoutExpired:
            print(f"⏰ {test_file}: TIMEOUT (exceeded 5 minutes)")
            results["failed"] += 1
        except Exception as e:
            print(f"❌ {test_file}: ERROR - {e}")
            results["failed"] += 1

    print(f"\n{'=' * 50}")
    print(
        f"Results: {results['passed']} passed, {results['failed']} failed, {results['skipped']} skipped"
    )

    if results["failed"] == 0 and results["passed"] > 0:
        print("\n🎉 All tests passed! Great work!")

    return results["failed"] == 0


def list_topics():
    """List all available topics."""
    root = Path(__file__).parent.parent

    modules = [
        "Module 00-Foundations",
        "Module 01-Neural-Network-Core",
        "Module 02-CNNs",
        "Module 03-RNNs-Sequences",
        "Module 04-Transformers-Production",
    ]

    print("📚 Available Topics:\n")

    for module_name in modules:
        module_path = root / module_name
        if not module_path.exists():
            continue

        print(f"\n{module_name}")
        print("-" * len(module_name))

        for topic_folder in sorted(module_path.iterdir()):
            if topic_folder.is_dir() and topic_folder.name.startswith("Topic"):
                has_tests = (topic_folder / "tests").exists()
                status = "✅" if has_tests else "📝"
                print(f"  {status} {topic_folder.name}")


def main():
    parser = argparse.ArgumentParser(
        description="DL-From-Scratch Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python utils/test_runner.py --verify-setup     Verify environment
  python utils/test_runner.py --day 01           Run Topic 01 tests
  python utils/test_runner.py --day 05 --level 2 Run Topic 05 Level 2 tests
  python utils/test_runner.py --list             List all topics
        """,
    )
    parser.add_argument("--day", type=int, help="Topic number to test (1-34)")
    parser.add_argument(
        "--level", type=int, choices=[1, 2, 3, 4], help="Solution level to test"
    )
    parser.add_argument(
        "--verify-setup", action="store_true", help="Verify environment setup"
    )
    parser.add_argument("--list", action="store_true", help="List all available topics")

    args = parser.parse_args()

    if args.verify_setup:
        success = verify_setup()
        sys.exit(0 if success else 1)

    if args.list:
        list_topics()
        sys.exit(0)

    if args.day:
        success = run_tests(args.day, args.level)
        sys.exit(0 if success else 1)

    parser.print_help()


if __name__ == "__main__":
    main()
