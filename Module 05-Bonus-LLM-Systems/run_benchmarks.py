#!/usr/bin/env python3
"""Run benchmark scripts for Module 05 topics."""

import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent
    topics = sorted(root.glob("Topic */benchmark.py"))
    if not topics:
        print("No benchmark scripts found.")
        return 1

    failed = 0
    for bench in topics:
        print(f"\n=== Benchmark: {bench.parent.name} ===")
        proc = subprocess.run([sys.executable, str(bench)], cwd=str(root.parent))
        if proc.returncode != 0:
            failed = 1
            print(f"FAILED: {bench}")

    return failed


if __name__ == "__main__":
    raise SystemExit(main())
