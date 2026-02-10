#!/usr/bin/env python3
"""
Full-curriculum validator for DL-From-Scratch.

Runs Topic 01-34 tests through utils/test_runner.py, writes:
- human-readable log
- machine-readable JSON report
- compatibility failure list file
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DEFAULT_START_DAY = 1
DEFAULT_END_DAY = 34


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def run_command(cmd: List[str], cwd: Path) -> Tuple[int, str, str, float]:
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    return proc.returncode, proc.stdout, proc.stderr, elapsed


def sanitize_days(start_day: int, end_day: int) -> List[int]:
    if start_day < DEFAULT_START_DAY or end_day > DEFAULT_END_DAY:
        raise ValueError("Day range must be within 1..34")
    if start_day > end_day:
        raise ValueError("start_day cannot be greater than end_day")
    return list(range(start_day, end_day + 1))


def run_setup_check(root: Path) -> Dict:
    cmd = [sys.executable, str(root / "utils" / "test_runner.py"), "--verify-setup"]
    code, stdout, stderr, elapsed = run_command(cmd, root)
    return {
        "passed": code == 0,
        "exit_code": code,
        "duration_seconds": round(elapsed, 3),
        "stdout": stdout,
        "stderr": stderr,
    }


def run_day_check(root: Path, day: int) -> Dict:
    cmd = [sys.executable, str(root / "utils" / "test_runner.py"), "--day", str(day)]
    code, stdout, stderr, elapsed = run_command(cmd, root)
    return {
        "day": day,
        "passed": code == 0,
        "exit_code": code,
        "duration_seconds": round(elapsed, 3),
        "stdout": stdout,
        "stderr": stderr,
    }


def render_log(setup_result: Optional[Dict], results: List[Dict]) -> str:
    lines: List[str] = []
    lines.append(
        f"Validation timestamp: {dt.datetime.now(dt.timezone.utc).isoformat()}"
    )
    lines.append(f"Python executable: {sys.executable}")
    lines.append("")

    if setup_result is not None:
        lines.append("===== SETUP CHECK =====")
        lines.append(setup_result["stdout"].rstrip())
        if setup_result["stderr"].strip():
            lines.append(setup_result["stderr"].rstrip())
        lines.append(
            f"SETUP: {'PASS' if setup_result['passed'] else 'FAIL'} "
            f"({setup_result['duration_seconds']:.3f}s)"
        )
        lines.append("")

    for result in results:
        lines.append(f"===== DAY {result['day']:02d} =====")
        lines.append(result["stdout"].rstrip())
        if result["stderr"].strip():
            lines.append(result["stderr"].rstrip())
        lines.append(
            f"DAY {result['day']:02d}: {'PASS' if result['passed'] else 'FAIL'} "
            f"({result['duration_seconds']:.3f}s)"
        )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def summarize_results(setup_result: Optional[Dict], results: List[Dict]) -> Dict:
    failed_days = [entry["day"] for entry in results if not entry["passed"]]
    passed_days = [entry["day"] for entry in results if entry["passed"]]
    setup_ok = True if setup_result is None else setup_result["passed"]
    success = setup_ok and not failed_days

    return {
        "total_days": len(results),
        "passed_days": len(passed_days),
        "failed_days_count": len(failed_days),
        "failing_days": failed_days,
        "success": success,
    }


def write_outputs(
    root: Path,
    log_text: str,
    report: Dict,
    log_out: Path,
    json_out: Path,
    failures_out: Path,
) -> None:
    (root / log_out).write_text(log_text, encoding="utf-8")
    (root / json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    failures = report["summary"]["failing_days"]
    failure_text = "\n".join(str(day) for day in failures)
    if failure_text:
        failure_text += "\n"
    (root / failures_out).write_text(failure_text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate all DL-From-Scratch topics")
    parser.add_argument("--start-day", type=int, default=DEFAULT_START_DAY)
    parser.add_argument("--end-day", type=int, default=DEFAULT_END_DAY)
    parser.add_argument("--skip-setup-check", action="store_true")
    parser.add_argument("--stop-on-fail", action="store_true")
    parser.add_argument("--log-out", default=".validation.log")
    parser.add_argument("--json-out", default=".validation_report.json")
    parser.add_argument("--failures-out", default=".validation_failures.log")
    args = parser.parse_args()

    root = repo_root()
    days = sanitize_days(args.start_day, args.end_day)

    setup_result = None
    if not args.skip_setup_check:
        setup_result = run_setup_check(root)
        status = "PASS" if setup_result["passed"] else "FAIL"
        print(
            f"[setup] {status} in {setup_result['duration_seconds']:.2f}s",
            flush=True,
        )

    results: List[Dict] = []
    for day in days:
        result = run_day_check(root, day)
        results.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        print(
            f"[day {day:02d}] {status} in {result['duration_seconds']:.2f}s",
            flush=True,
        )
        if args.stop_on_fail and not result["passed"]:
            break

    log_text = render_log(setup_result, results)
    summary = summarize_results(setup_result, results)
    report = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "root_dir": str(root),
        "range": {"start_day": args.start_day, "end_day": args.end_day},
        "setup_check": None
        if setup_result is None
        else {
            "passed": setup_result["passed"],
            "exit_code": setup_result["exit_code"],
            "duration_seconds": setup_result["duration_seconds"],
        },
        "results": [
            {
                "day": row["day"],
                "passed": row["passed"],
                "exit_code": row["exit_code"],
                "duration_seconds": row["duration_seconds"],
            }
            for row in results
        ],
        "summary": summary,
    }

    write_outputs(
        root=root,
        log_text=log_text,
        report=report,
        log_out=Path(args.log_out),
        json_out=Path(args.json_out),
        failures_out=Path(args.failures_out),
    )

    print("")
    print("Validation summary")
    print("------------------")
    print(f"Days checked : {summary['total_days']}")
    print(f"Passed       : {summary['passed_days']}")
    print(f"Failed       : {summary['failed_days_count']}")
    if summary["failing_days"]:
        print(
            "Failing days : "
            + ", ".join(f"{day:02d}" for day in summary["failing_days"])
        )
    print(f"Log file     : {args.log_out}")
    print(f"JSON report  : {args.json_out}")
    print(f"Failures file: {args.failures_out}")

    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
