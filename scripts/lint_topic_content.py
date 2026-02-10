#!/usr/bin/env python3
"""Lint topic content structure and quality signals."""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

MILESTONE_TOPICS = {10, 17, 24, 30, 34}
BONUS_TOPIC_MIN = 35

README_SIGNALS = {
    "objective": [
        "learning objectives",
        "objective",
        "why this topic",
    ],
    "problem_or_scope": [
        "problem statement",
        "the problem",
        "core apis",
        "deliverables",
        "key concepts to master",
        "file structure",
    ],
    "verification": [
        "success criteria",
        "verification workflow",
        "commands",
        "how to use this topic",
    ],
}

REQUIRED_FILES = [
    "README.md",
    "questions.md",
    "intuition.md",
    "math-refresh.md",
    "solutions/level01_naive.py",
    "solutions/level02_vectorized.py",
    "solutions/level03_memory_efficient.py",
    "solutions/level04_pytorch_reference.py",
    "tests/test_basic.py",
    "tests/test_edge.py",
    "tests/test_stress.py",
]


@dataclass
class TopicLintResult:
    topic_dir: str
    topic_number: Optional[int]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0


def parse_topic_number(topic_dir: Path) -> Optional[int]:
    match = re.match(r"Topic (\d+)", topic_dir.name)
    if not match:
        return None
    return int(match.group(1))


def contains_any(text: str, phrases: Sequence[str]) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in phrases)


def count_question_prompts(text: str) -> int:
    lines = text.splitlines()
    section_questions = sum(
        1 for line in lines if re.match(r"^\s*##\s+Q\d+", line, flags=re.IGNORECASE)
    )
    inline_questions = sum(1 for line in lines if line.strip().endswith("?"))
    return max(section_questions, inline_questions)


def ensure_hint_set(topic_dir: Path, result: TopicLintResult) -> None:
    for level in (1, 2, 3):
        matches = sorted(topic_dir.glob("hints/hint-{}*.md".format(level)))
        if not matches:
            result.errors.append("missing hints/hint-{}*.md".format(level))
            continue

        hint_text = matches[0].read_text(encoding="utf-8", errors="replace").strip()
        if len(hint_text) < 80:
            result.warnings.append(
                "hint-{} appears too short (<80 chars): {}".format(
                    level, matches[0].name
                )
            )


def lint_topic(topic_dir: Path) -> TopicLintResult:
    topic_number = parse_topic_number(topic_dir)
    result = TopicLintResult(topic_dir=str(topic_dir), topic_number=topic_number)

    for rel_path in REQUIRED_FILES:
        if not (topic_dir / rel_path).exists():
            result.errors.append("missing {}".format(rel_path))

    ensure_hint_set(topic_dir, result)

    readme_path = topic_dir / "README.md"
    if readme_path.exists():
        readme_text = readme_path.read_text(encoding="utf-8", errors="replace")
        for signal_name, phrases in README_SIGNALS.items():
            if not contains_any(readme_text, phrases):
                result.errors.append(
                    "README missing {} signal ({})".format(
                        signal_name, ", ".join(phrases[:2])
                    )
                )

    questions_path = topic_dir / "questions.md"
    if questions_path.exists():
        questions_text = questions_path.read_text(encoding="utf-8", errors="replace")
        prompt_count = count_question_prompts(questions_text)
        if prompt_count < 3:
            result.errors.append(
                "questions.md has too few prompts (found {}, expected >= 3)".format(
                    prompt_count
                )
            )

    if topic_number in MILESTONE_TOPICS and not (topic_dir / "metrics.md").exists():
        result.errors.append("milestone topic missing metrics.md")

    if topic_number is not None and topic_number >= BONUS_TOPIC_MIN:
        if not (topic_dir / "benchmark.py").exists():
            result.errors.append("bonus topic missing benchmark.py")

    return result


def find_topic_dirs(repo_root: Path, core_only: bool) -> List[Path]:
    topic_dirs = sorted(repo_root.glob("Module */Topic *"))
    if core_only:
        filtered = []
        for topic_dir in topic_dirs:
            number = parse_topic_number(topic_dir)
            if number is not None and number <= 34:
                filtered.append(topic_dir)
        return filtered
    return topic_dirs


def build_payload(results: List[TopicLintResult], core_only: bool) -> Dict[str, object]:
    total = len(results)
    failed = sum(1 for res in results if not res.passed)
    warnings = sum(len(res.warnings) for res in results)
    return {
        "core_only": core_only,
        "topics_checked": total,
        "topics_failed": failed,
        "warnings": warnings,
        "passed": failed == 0,
        "results": [
            {
                "topic_dir": res.topic_dir,
                "topic_number": res.topic_number,
                "passed": res.passed,
                "errors": res.errors,
                "warnings": res.warnings,
            }
            for res in results
        ],
    }


def print_summary(results: List[TopicLintResult]) -> None:
    for res in results:
        label = "PASS" if res.passed else "FAIL"
        suffix = ""
        if res.warnings:
            suffix = " ({} warning(s))".format(len(res.warnings))
        print("[{}] {}{}".format(label, res.topic_dir, suffix))
        for err in res.errors:
            print("  - ERROR: {}".format(err))
        for warning in res.warnings:
            print("  - WARN : {}".format(warning))

    total = len(results)
    failed = sum(1 for res in results if not res.passed)
    print("\nLint summary")
    print("------------")
    print("Topics checked: {}".format(total))
    print("Topics failed : {}".format(failed))
    print("Topics passed : {}".format(total - failed))


def main() -> int:
    parser = argparse.ArgumentParser(description="Lint topic content structure")
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Check only core topics (01-34), excluding optional bonus topics.",
    )
    parser.add_argument(
        "--json-out",
        default=".topic_content_lint_report.json",
        help="Path to JSON report output.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-topic output; print summary only.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    topic_dirs = find_topic_dirs(repo_root, core_only=args.core_only)
    if not topic_dirs:
        print("No topic directories found.", file=sys.stderr)
        return 2

    results = [lint_topic(topic_dir) for topic_dir in topic_dirs]
    payload = build_payload(results, core_only=args.core_only)

    out_path = repo_root / args.json_out
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.quiet:
        total = len(results)
        failed = sum(1 for res in results if not res.passed)
        print("Topics checked: {} | Failed: {}".format(total, failed))
    else:
        print_summary(results)

    print("JSON report: {}".format(out_path.relative_to(repo_root)))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
