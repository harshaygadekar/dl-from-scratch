# Topic 28: Transformer Encoder Block

> Goal: build a working NumPy scaffold for Transformer Encoder Block.

## Scope
- Implement core primitives with clear shape contracts.
- Add one correctness check for key math behavior.
- Keep level04 as optional verification only.

## Files
- `questions.md`, `intuition.md`, `math-refresh.md`
- `hints/` (progressive guidance)
- `solutions/` (level01 to level04)
- `tests/` (basic/edge/stress)

## Success Criteria
1. `test_basic.py` passes with stable outputs.
2. `test_edge.py` validates at least one tricky mask/shape edge case.
3. `test_stress.py` runs medium-size tensors without crashing.
