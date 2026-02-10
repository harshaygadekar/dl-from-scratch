# Topic 35: LoRA and QLoRA Fundamentals

> Goal: Build a realistic NumPy prototype for LoRA and QLoRA Fundamentals.
> Track: Optional bonus systems topic.

## Why This Topic

This topic reflects practical LLM systems work that appears in modern inference/training stacks.

## Core APIs

- lora_delta(base_w, a, b, alpha)\n- apply_lora(base_w, a, b, alpha)\n- quantize_4bit_linear(x, scale)

## Constraints

- NumPy-only implementation for core logic.
- Deterministic behavior under fixed seeds.
- Include one benchmark comparing baseline and improved path.

## Primary Risk

Low-rank update is applied with wrong scaling, producing unstable effective weights.

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
