# Topic 38: SSM and Mamba Selective Scan

> Goal: Build a realistic NumPy prototype for SSM and Mamba Selective Scan.
> Track: Optional bonus systems topic.

## Why This Topic

This topic reflects practical LLM systems work that appears in modern inference/training stacks.

## Core APIs

- selective_scan_naive(a, b, x)\n- selective_scan_vectorized(a, b, x)\n- chunked_selective_scan(a, b, x, chunk)

## Constraints

- NumPy-only implementation for core logic.
- Deterministic behavior under fixed seeds.
- Include one benchmark comparing baseline and improved path.

## Primary Risk

State update order is wrong, yielding unstable recurrence outputs.

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
