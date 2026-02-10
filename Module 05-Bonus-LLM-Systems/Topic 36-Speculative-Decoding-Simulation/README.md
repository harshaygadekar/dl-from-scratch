# Topic 36: Speculative Decoding Simulation

> Goal: Build a realistic NumPy prototype for Speculative Decoding Simulation.
> Track: Optional bonus systems topic.

## Why This Topic

This topic reflects practical LLM systems work that appears in modern inference/training stacks.

## Core APIs

- greedy_next(logits)\n- propose_draft(draft_logits, k)\n- speculative_accept_reject(target_logits, draft_tokens)

## Constraints

- NumPy-only implementation for core logic.
- Deterministic behavior under fixed seeds.
- Include one benchmark comparing baseline and improved path.

## Primary Risk

Acceptance logic uses mismatched token positions and overestimates speedup.

## Success Criteria

1. tests/test_basic.py passes.
2. tests/test_edge.py passes.
3. tests/test_stress.py passes.
4. benchmark.py reports a baseline vs improved comparison.

## Commands

- 'python3 -m pytest tests/test_basic.py -v'
- 'python3 -m pytest tests/test_edge.py -v'
- 'python3 -m pytest tests/test_stress.py -v'
- 'python3 benchmark.py'
