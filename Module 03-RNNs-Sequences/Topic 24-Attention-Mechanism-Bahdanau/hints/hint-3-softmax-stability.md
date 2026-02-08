# Hint 3: Softmax Stability

## Goal In This Hint

Use alignment scores to dynamically focus decoder updates on relevant encoder states.

## Core Idea

Use max-shifted logits before exp to avoid overflow and NaN probabilities.
Before coding, keep a shape trace next to your implementation and update it whenever you reshape, transpose, split, or merge axes.

## Implementation Plan

1. Stress test numerical stability with extreme logits/activations and long sequences when applicable.
2. Add guardrails (`epsilon`, clipping, max-shifted softmax, finite checks) at known unstable points.
3. Profile runtime/memory on realistic sizes and confirm optimization does not change outputs.
4. Write one regression test for the hardest bug you hit while implementing this topic.

## Common Mistakes

- Mixing axis conventions (`NCHW` vs `NHWC`, `B,T,H,D` vs `B,H,T,D`) and debugging values before fixing shape contracts.
- Applying residual, normalization, masking, or scaling in the wrong order for the intended algorithm.
- Skipping finite checks (`NaN`/`Inf`) during development and finding instability only after many training steps.
- Optimization before correctness usually bakes in subtle errors that are expensive to unwind.

## Quick Self-Check

- Can you state the expected shape and dtype at every major line?
- Do tiny deterministic tests and random-input tests both pass?
- If you disable optimizations, do outputs still match within tolerance?

## Next

After this hint, run the topic tests and inspect at least one case end-to-end (input -> intermediate -> output) before moving on.
