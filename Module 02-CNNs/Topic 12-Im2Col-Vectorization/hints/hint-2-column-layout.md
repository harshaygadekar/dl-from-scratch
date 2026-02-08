# Hint 2: Column Layout

## Goal In This Hint

Turn convolution into matrix multiplication by extracting deterministic sliding windows.

## Core Idea

Choose a column layout that preserves (channel, kernel_row, kernel_col) ordering consistently.
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

Continue with [Hint 3 - Matmul Integration](hint-3-matmul-integration.md) after you can pass the quick checks below.
