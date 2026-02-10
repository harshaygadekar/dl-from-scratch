# Contributing to DL-From-Scratch

Thank you for your interest in contributing! This guide explains how to add hints, fixes, and new content.

## Types of Contributions

### 🔧 Bug Fixes
Found an error in a solution or test? Open a PR with the fix.

### 💡 Hints
Want to add a helpful hint? Follow the hint format below.

### 📚 Clarifications
Improve intuition.md or math-refresh.md with clearer explanations.

### 🌐 Translations
Help translate content to other languages in the `translations/` directory.

## Contribution Rules

### ✅ Allowed
- Adding hints (never solutions)
- Fixing bugs in existing solutions
- Clarifying explanations
- Adding edge case tests
- Improving visualizations

### ❌ Not Allowed
- Adding full solutions to existing topics
- Removing the "from scratch" constraint
- Adding framework dependencies (PyTorch, TensorFlow)
- Changing the curriculum structure without discussion

## How to Contribute

### Step 1: Fork the Repository

```bash
git clone https://github.com/YOUR_USERNAME/dl-from-scratch.git
cd dl-from-scratch
```

### Step 2: Create a Branch

```bash
git checkout -b fix/topic-05-batchnorm-bug
# or
git checkout -b hint/topic-12-im2col
```

### Step 3: Make Changes

Follow the file templates exactly. Run tests before submitting.

### Step 4: Run Quality Gates

For topic/content changes, run:

```bash
# Content quality lint (all topics)
python3 scripts/lint_topic_content.py

# Or core-only lint
python3 scripts/lint_topic_content.py --core-only
```

For solution/test changes in a topic, also run:

```bash
# Replace XX with your topic number
python3 utils/test_runner.py --day XX
```

### Step 5: Submit PR

Use the PR template and describe your changes clearly.

---

## File Templates

### Hint File Format

```markdown
# Hint: [Brief Description]

## The Problem You're Facing
[One sentence describing when to use this hint]

## The Key Insight
[2-3 sentences with the core idea]

## Pseudocode
```python
# Step-by-step pseudocode
```

## Still Stuck?
→ Check hint-2 for more detailed algorithm
```

### Test File Format

```python
"""Tests for Topic XX: [Name]"""

import numpy as np
import pytest

def test_basic_case():
    """Verify output shape and basic correctness."""
    pass

def test_edge_case_batch_size_1():
    """Batch size of 1 should still work."""
    pass

def test_edge_case_empty():
    """Empty input should be handled gracefully."""
    pass
```

---

## Code Style

- Use Python 3.8+ features
- Follow PEP 8
- Include docstrings for all functions
- Add type hints where helpful
- Keep solutions readable over clever

---

## Topic Quality Checklist (Required for Topic PRs)

Use:

- `docs/authoring/topic-quality-checklist.md`

Minimum merge bar for topic-related PRs:

- Topic structure files are present (`README.md`, `questions.md`, hints, tests, solutions).
- README includes objective framing, implementation/problem framing, and success/verification criteria.
- `questions.md` includes at least 3 concrete interview prompts.
- Milestone topics (10, 17, 24, 30, 34) include `metrics.md`.
- Bonus topics (35-38) include `benchmark.py`.
- `python3 scripts/lint_topic_content.py` passes.

---

## Reporting Content Quality Issues

Use issue templates:

- `Bug Report` for implementation/test defects.
- `I'm Stuck` for learner debugging help.
- `Curriculum Quality Gap` for unclear/incomplete topic content.

---

## Questions?

Open an issue with the `question` label.

---

*Thank you for helping make deep learning education better!*
