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

### Step 4: Submit PR

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

## Questions?

Open an issue with the "question" label.

---

*Thank you for helping make deep learning education better!*
