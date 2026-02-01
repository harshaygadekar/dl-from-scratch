#!/usr/bin/env python3
"""
Progress Tracker

Shows visual progress through the DL-From-Scratch curriculum.

Usage:
    python utils/progress.py
    python utils/progress.py --detailed
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple


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


def get_root_dir() -> Path:
    """Get the repository root directory."""
    return Path(__file__).parent.parent


def scan_completed_topics() -> List[int]:
    """
    Scan for completed topics based on solution file existence.
    
    A topic is considered complete if it has at least the level01 solution.
    """
    root = get_root_dir()
    completed = []
    
    for module_name, start, end in MODULES:
        module_path = root / module_name
        if not module_path.exists():
            continue
        
        for topic_folder in module_path.iterdir():
            if not topic_folder.is_dir():
                continue
            
            # Extract topic number from folder name
            name = topic_folder.name
            if not name.startswith("Topic"):
                continue
            
            try:
                # Parse "Topic XX-Name" format
                parts = name.split("-")
                topic_num = int(parts[0].replace("Topic", "").strip())
            except (ValueError, IndexError):
                continue
            
            # Check for solution files
            solutions_dir = topic_folder / "solutions"
            if solutions_dir.exists():
                solution_files = list(solutions_dir.glob("*.py"))
                if len(solution_files) >= 1:
                    completed.append(topic_num)
    
    return sorted(completed)


def get_progress_stats(completed: List[int]) -> Dict:
    """Calculate progress statistics."""
    total_topics = 34
    num_completed = len(completed)
    percentage = (num_completed / total_topics) * 100
    
    # Find current topic (highest completed + 1)
    current_topic = max(completed) + 1 if completed else 1
    current_topic = min(current_topic, 34)
    
    # Find current module
    current_module = "Module 00"
    for module_name, start, end in MODULES:
        if start <= current_topic <= end:
            current_module = module_name.split("-")[0]
            break
    
    # Find next milestone
    next_milestone = None
    next_milestone_desc = None
    for milestone_topic, desc in sorted(MILESTONES.items()):
        if milestone_topic not in completed:
            next_milestone = milestone_topic
            next_milestone_desc = desc
            break
    
    return {
        "completed": num_completed,
        "total": total_topics,
        "percentage": percentage,
        "current_topic": current_topic,
        "current_module": current_module,
        "completed_topics": completed,
        "next_milestone": next_milestone,
        "next_milestone_desc": next_milestone_desc,
    }


def render_progress_bar(percentage: float, width: int = 40) -> str:
    """Render ASCII progress bar."""
    filled = int(width * percentage / 100)
    remaining = width - filled - 1
    
    if remaining < 0:
        remaining = 0
        filled = width
    
    bar = "█" * filled
    if remaining > 0:
        bar += "▶" + "░" * remaining
    
    return f"[{bar}]"


def print_basic_progress(stats: Dict):
    """Print basic progress summary."""
    bar = render_progress_bar(stats["percentage"])
    
    print()
    print(bar)
    print(f"{stats['current_module']} Topic {stats['current_topic']:02d} ({stats['percentage']:.0f}%)")
    print()
    print(f"Completed: {stats['completed']}/{stats['total']} topics")
    
    if stats["next_milestone"]:
        topics_to_milestone = stats["next_milestone"] - max(stats["completed_topics"], default=0)
        print(f"Next milestone: Topic {stats['next_milestone']} - {stats['next_milestone_desc']}")
        print(f"Topics remaining: {topics_to_milestone}")


def print_detailed_progress(stats: Dict):
    """Print detailed progress by module."""
    print("\n" + "="*60)
    print("DL-From-Scratch Progress Report")
    print("="*60)
    
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
    
    print("\n" + "="*60)
    print_basic_progress(stats)
    
    # Encouragement messages
    print()
    if stats["completed"] == stats["total"]:
        print("🎉 Congratulations! You've completed DL-From-Scratch!")
        print("   You now truly understand deep learning from the ground up.")
    elif stats["completed"] >= 30:
        print("🔥 Almost there! Just the advanced production topics left!")
    elif stats["completed"] >= 24:
        print("🌟 Incredible progress! You've mastered attention mechanisms!")
    elif stats["completed"] >= 17:
        print("📈 Great work! CNNs complete, moving to sequences!")
    elif stats["completed"] >= 10:
        print("💪 Excellent! You can now train a neural network from scratch!")
    elif stats["completed"] >= 3:
        print("🚀 Foundation complete! Ready for neural networks!")
    elif stats["completed"] >= 1:
        print("👍 You've started! Keep the momentum going!")
    else:
        print("📚 Welcome! Start with Topic 01 to begin your journey.")


def main():
    parser = argparse.ArgumentParser(
        description="Track your progress through DL-From-Scratch"
    )
    parser.add_argument(
        "--detailed", "-d", 
        action="store_true",
        help="Show detailed progress by module"
    )
    
    args = parser.parse_args()
    
    completed = scan_completed_topics()
    stats = get_progress_stats(completed)
    
    if args.detailed:
        print_detailed_progress(stats)
    else:
        print_basic_progress(stats)


if __name__ == "__main__":
    main()
