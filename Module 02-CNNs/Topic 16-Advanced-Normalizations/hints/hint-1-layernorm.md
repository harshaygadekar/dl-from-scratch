# Hint 1: LayerNorm

## Goal In This Hint

Implement normalization layers that work well across different batch sizes and tasks.

## Core Idea

LayerNorm normalizes across feature dimensions within each sample, independent of batch size.
Before coding, keep a shape trace next to your implementation and update it whenever you reshape, transpose, split, or merge axes.

## Implementation Plan

1. Write the exact tensor shapes for inputs, outputs, and key intermediates.
2. Implement a smallest-case forward path first (`batch=1`, minimal sequence/spatial length).
3. Add one deterministic numeric example and one strict shape assertion before generalizing.
4. Only then expand to full batch and real dimensions.

## Common Mistakes

- Mixing axis conventions (`NCHW` vs `NHWC`, `B,T,H,D` vs `B,H,T,D`) and debugging values before fixing shape contracts.
- Applying residual, normalization, masking, or scaling in the wrong order for the intended algorithm.
- Skipping finite checks (`NaN`/`Inf`) during development and finding instability only after many training steps.
- Starting with full-size tensors hides indexing mistakes that are obvious on tiny examples.

## Quick Self-Check

- Can you state the expected shape and dtype at every major line?
- Do tiny deterministic tests and random-input tests both pass?
- Can you explain every axis transition line by line without guessing?

## Next

Continue with [Hint 2 - GroupNorm](hint-2-groupnorm.md) after you can pass the quick checks below.
