# Hint 2: RoPE Intuition

## Goal In This Hint

Inject position information so token order is recoverable by attention-only architectures.

## Core Idea

RoPE rotates query/key pairs by position-dependent angles to encode relative phase.
Before coding, keep a shape trace next to your implementation and update it whenever you reshape, transpose, split, or merge axes.

## Implementation Plan

1. Integrate the component into the full block/pipeline while preserving axis order.
2. Handle optional logic explicitly (masks, padding, teacher forcing, residual branch, cache path).
3. Validate against a simple reference implementation on random seeds and edge sizes.
4. Log one intermediate tensor statistic (mean/std/min/max) to catch silent failures early.

## Common Mistakes

- Mixing axis conventions (`NCHW` vs `NHWC`, `B,T,H,D` vs `B,H,T,D`) and debugging values before fixing shape contracts.
- Applying residual, normalization, masking, or scaling in the wrong order for the intended algorithm.
- Skipping finite checks (`NaN`/`Inf`) during development and finding instability only after many training steps.
- Broadcasting can succeed with wrong semantics; verify intended dimensions, not just runnable code.

## Quick Self-Check

- Can you state the expected shape and dtype at every major line?
- Do tiny deterministic tests and random-input tests both pass?
- Do masked/optional branches produce identical results to the reference in supported cases?

## Next

Continue with [Hint 3 - Position Scaling](hint-3-position-scaling.md) after you can pass the quick checks below.
