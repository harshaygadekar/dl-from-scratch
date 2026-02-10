# Topic 37: Mixture of Experts Routing

> Goal: Build a realistic NumPy prototype for Mixture of Experts Routing.
> Track: Optional bonus systems topic.

## Why This Topic

This topic reflects practical LLM systems work that appears in modern inference/training stacks.

## Core APIs

- top1_route(router_logits)\n- top2_route(router_logits)\n- load_balance_penalty(assignments, num_experts)

## Constraints

- NumPy-only implementation for core logic.
- Deterministic behavior under fixed seeds.
- Include one benchmark comparing baseline and improved path.

## Primary Risk

Routing collapses to a few experts and destroys load balance.

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
