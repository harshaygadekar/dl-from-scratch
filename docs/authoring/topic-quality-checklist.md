# Topic Quality Checklist

Use this checklist before opening a PR that changes or adds topic content.

## Required Topic Structure

- Topic folder includes:
  - `README.md`
  - `questions.md`
  - `intuition.md`
  - `math-refresh.md`
  - `hints/` with `hint-1*.md`, `hint-2*.md`, `hint-3*.md`
  - `solutions/level01_naive.py`
  - `solutions/level02_vectorized.py`
  - `solutions/level03_memory_efficient.py`
  - `solutions/level04_pytorch_reference.py`
  - `tests/test_basic.py`
  - `tests/test_edge.py`
  - `tests/test_stress.py`

## README Quality Bar

- README includes an objective section:
  - examples: "Learning Objectives", "Objective", or "Why This Topic"
- README includes implementation/problem framing:
  - examples: "Problem Statement", "The Problem", "Core APIs", "Deliverables", "Key Concepts to Master"
- README includes verification or completion criteria:
  - examples: "Success Criteria", "Verification Workflow", "Commands", "How to Use This Topic"

## Questions Quality Bar

- `questions.md` has at least 3 interview prompts.
- Prompts are topic-specific and not template filler.

## Hints Quality Bar

- All three progressive hints exist.
- Hint content is non-trivial (not placeholders).
- Hints guide without revealing full final implementation directly.

## Testing Quality Bar

- `test_basic.py` verifies baseline correctness.
- `test_edge.py` covers shape/edge behavior.
- `test_stress.py` exercises larger or adversarial inputs.

## Milestone and Bonus Rules

- Milestone topics (10, 17, 24, 30, 34) include `metrics.md`.
- Bonus topics (35-38) include `benchmark.py`.

## Automated Check

Run:

```bash
python3 scripts/lint_topic_content.py
```

Core-only lint:

```bash
python3 scripts/lint_topic_content.py --core-only
```
