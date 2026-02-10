# Topic 17: CIFAR10 From Scratch

> Goal: Build an end-to-end CIFAR-10 pipeline with augmentation, batching, and baseline CNN training loop.
> Time: 3-4 hours | Difficulty: Medium-Hard

---

## Learning Objectives

By the end of this topic, you will:
1. Implement core primitives for CIFAR10 From Scratch in pure NumPy.
2. Enforce strict tensor shape contracts and deterministic checks.
3. Diagnose and fix at least one high-risk failure mode before optimization.
4. Explain trade-offs between Level 1, Level 2, and Level 3 implementations.

---

## Why This Topic Matters

This topic is a dependency for later modules. If this implementation is unstable, downstream topics inherit silent bugs.
Focus area for this topic: data pipeline reproducibility and train/eval loop correctness.

---

## Problem Statement

Build a complete, testable implementation path for CIFAR10 From Scratch across solution tiers:
- Level 1: correctness-first baseline
- Level 2: vectorized acceleration
- Level 3: memory/performance-aware variant
- Level 4: reference verification only

Primary API focus for this topic:
- load_cifar10, CIFAR10DataLoader, random_crop, random_horizontal_flip

---

## Constraints

- Use NumPy and Python stdlib only for implementation.
- Preserve numerical stability for stress-test tensor sizes.
- Keep behavior consistent across Level 1, Level 2, and Level 3 within tolerance.
- Do not treat Level 4 as the implementation shortcut.

---

## High-Risk Failure Mode

- Augmentation modifies labels or channel order (NHWC/NCHW) causing unstable training metrics.

Mitigation strategy:
1. Lock shape contracts before optimization.
2. Add a deterministic seed-based regression case.
3. Compare Level 2 and Level 3 outputs against Level 1 on the same input.

---

## Shape Contract Example

- CIFAR batch (64,32,32,3) normalized then model-ready tensor conversion is consistent each epoch.

This contract should be explicitly asserted during development and in tests.

---

## Verification Workflow

1. Run basic correctness tests:
   'python3 -m pytest tests/test_basic.py -v'
2. Run boundary-condition tests:
   'python3 -m pytest tests/test_edge.py -v'
3. Run performance/stability tests:
   'python3 -m pytest tests/test_stress.py -v'
4. Validate full topic through the runner:
   'python3 utils/test_runner.py --day 17'

---

## Success Criteria

1. All topic tests pass with finite outputs.
2. Deterministic checks reproduce the same outputs under fixed seed.
3. Documented failure mode is reproducibly prevented by assertions/tests.
4. Level 2 or Level 3 shows concrete implementation improvement over Level 1.

---

## Deliverables

- Updated implementation in solutions/
- Passing tests in tests/test_basic.py, tests/test_edge.py, tests/test_stress.py
- Notes in questions.md and progressive troubleshooting via hints/

---

## Next Topic

Continue to Topic 18 after passing all tests at this topic.
