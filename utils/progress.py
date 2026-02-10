#!/usr/bin/env python3
"""
Progress Tracker

Tracks learner progress through the DL-From-Scratch curriculum using local state.

Usage:
    python utils/progress.py
    python utils/progress.py --detailed
    python utils/progress.py --mark-topic 1
    python utils/progress.py --unmark-topic 1
    python utils/progress.py --reset
    python utils/progress.py --status-json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

# Module definitions: (name, topic_start, topic_end)
MODULES = [
    ("Module 00-Foundations", 1, 3),
    ("Module 01-Neural-Network-Core", 4, 10),
    ("Module 02-CNNs", 11, 17),
    ("Module 03-RNNs-Sequences", 18, 24),
    ("Module 04-Transformers-Production", 25, 34),
]

MILESTONES = {
    10: "MNIST 95%+ accuracy",
    17: "CIFAR-10 ResNet",
    24: "Bahdanau Attention",
    30: "Mini-GPT text generation",
    34: "Full curriculum complete!",
}

MIN_TOPIC = 1
MAX_TOPIC = 34


def get_root_dir() -> Path:
    """Get the repository root directory."""
    return Path(__file__).parent.parent


def get_progress_state_path() -> Path:
    """Get the local learner progress file path."""
    return get_root_dir() / "data" / "progress" / "progress_state.json"


def default_state() -> Dict:
    return {
        "version": 1,
        "completed_topics": [],
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _sanitize_topics(raw_topics: List[int]) -> List[int]:
    cleaned = []
    for topic in raw_topics:
        if isinstance(topic, int) and MIN_TOPIC <= topic <= MAX_TOPIC:
            cleaned.append(topic)
    return sorted(set(cleaned))


def load_state() -> Dict:
    """Load learner progress state from disk, creating defaults when needed."""
    state_path = get_progress_state_path()
    state_path.parent.mkdir(parents=True, exist_ok=True)

    if not state_path.exists():
        state = default_state()
        save_state(state)
        return state

    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        state = default_state()
        save_state(state)
        return state

    if not isinstance(state, dict):
        state = default_state()

    completed = state.get("completed_topics", [])
    if not isinstance(completed, list):
        completed = []

    state["completed_topics"] = _sanitize_topics(completed)
    state["version"] = 1
    state.setdefault("updated_at_utc", datetime.now(timezone.utc).isoformat())
    return state


def save_state(state: Dict) -> None:
    """Persist learner progress state to disk."""
    state_path = get_progress_state_path()
    state_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": 1,
        "completed_topics": _sanitize_topics(state.get("completed_topics", [])),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def get_progress_stats(completed: List[int]) -> Dict:
    """Calculate progress statistics from learner state."""
    completed = _sanitize_topics(completed)
    completed_set = set(completed)

    total_topics = MAX_TOPIC
    num_completed = len(completed)
    percentage = (num_completed / total_topics) * 100 if total_topics else 0.0

    current_topic = next(
        (
            topic
            for topic in range(MIN_TOPIC, MAX_TOPIC + 1)
            if topic not in completed_set
        ),
        MAX_TOPIC,
    )

    current_module = MODULES[0][0].split("-")[0]
    for module_name, start, end in MODULES:
        if start <= current_topic <= end:
            current_module = module_name.split("-")[0]
            break

    next_milestone = None
    next_milestone_desc = None
    for milestone_topic, desc in sorted(MILESTONES.items()):
        if milestone_topic not in completed_set:
            next_milestone = milestone_topic
            next_milestone_desc = desc
            break

    topics_to_milestone = 0
    if next_milestone is not None:
        topics_to_milestone = sum(
            1
            for topic in range(MIN_TOPIC, next_milestone + 1)
            if topic not in completed_set
        )

    return {
        "completed": num_completed,
        "total": total_topics,
        "percentage": percentage,
        "current_topic": current_topic,
        "current_module": current_module,
        "completed_topics": completed,
        "next_milestone": next_milestone,
        "next_milestone_desc": next_milestone_desc,
        "topics_to_milestone": topics_to_milestone,
        "state_file": str(get_progress_state_path()),
    }


def render_progress_bar(percentage: float, width: int = 40) -> str:
    """Render ASCII progress bar."""
    filled = int(width * percentage / 100)
    if filled >= width:
        return f"[{'█' * width}]"

    remaining = max(width - filled - 1, 0)
    return f"[{'█' * filled}▶{'░' * remaining}]"


def print_basic_progress(stats: Dict):
    """Print basic progress summary."""
    bar = render_progress_bar(stats["percentage"])

    print()
    print(bar)
    print(
        f"{stats['current_module']} Topic {stats['current_topic']:02d} ({stats['percentage']:.0f}%)"
    )
    print()
    print(f"Completed: {stats['completed']}/{stats['total']} topics")

    if stats["next_milestone"] is not None:
        print(
            f"Next milestone: Topic {stats['next_milestone']} - {stats['next_milestone_desc']}"
        )
        print(f"Topics remaining: {stats['topics_to_milestone']}")

    print(f"State file: {stats['state_file']}")


def print_detailed_progress(stats: Dict):
    """Print detailed progress by module."""
    print("\n" + "=" * 60)
    print("DL-From-Scratch Progress Report")
    print("=" * 60)

    completed_set = set(stats["completed_topics"])

    for module_name, start, end in MODULES:
        print(f"\n{module_name}")
        print("-" * len(module_name))

        module_completed = sum(1 for t in range(start, end + 1) if t in completed_set)
        module_total = end - start + 1
        module_pct = (module_completed / module_total) * 100

        print(f"Progress: {module_completed}/{module_total} ({module_pct:.0f}%)")

        for topic in range(start, end + 1):
            status = "✅" if topic in completed_set else "⬜"
            milestone_marker = " 🏆" if topic in MILESTONES else ""
            print(f"  {status} Topic {topic:02d}{milestone_marker}")

    print("\n" + "=" * 60)
    print_basic_progress(stats)


def validate_topics(topics: List[int], arg_name: str) -> List[int]:
    invalid = [topic for topic in topics if not (MIN_TOPIC <= topic <= MAX_TOPIC)]
    if invalid:
        invalid_str = ", ".join(str(topic) for topic in invalid)
        raise ValueError(
            f"{arg_name} values must be in range {MIN_TOPIC}-{MAX_TOPIC}. Invalid: {invalid_str}"
        )
    return sorted(set(topics))


def apply_actions(state: Dict, mark: List[int], unmark: List[int], reset: bool) -> Dict:
    if reset:
        state = default_state()

    completed = set(state.get("completed_topics", []))

    for topic in mark:
        completed.add(topic)

    for topic in unmark:
        completed.discard(topic)

    state["completed_topics"] = sorted(completed)
    return state


def main():
    parser = argparse.ArgumentParser(
        description="Track learner progress through DL-From-Scratch"
    )
    parser.add_argument(
        "--detailed",
        "-d",
        action="store_true",
        help="Show detailed progress by module",
    )
    parser.add_argument(
        "--mark-topic",
        type=int,
        nargs="+",
        default=[],
        help="Mark one or more topics complete (e.g. --mark-topic 1 2 3)",
    )
    parser.add_argument(
        "--unmark-topic",
        type=int,
        nargs="+",
        default=[],
        help="Unmark one or more completed topics",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset progress state to 0 completed topics",
    )
    parser.add_argument(
        "--status-json",
        action="store_true",
        help="Print progress stats as JSON",
    )

    args = parser.parse_args()

    try:
        mark_topics = validate_topics(args.mark_topic, "--mark-topic")
        unmark_topics = validate_topics(args.unmark_topic, "--unmark-topic")
    except ValueError as exc:
        parser.error(str(exc))

    state = load_state()
    has_actions = bool(mark_topics or unmark_topics or args.reset)

    if has_actions:
        state = apply_actions(state, mark_topics, unmark_topics, args.reset)
        save_state(state)
        state = load_state()

    stats = get_progress_stats(state["completed_topics"])

    if args.status_json:
        print(json.dumps(stats, indent=2))
        return

    if args.detailed:
        print_detailed_progress(stats)
    else:
        print_basic_progress(stats)


if __name__ == "__main__":
    main()
