# Hint 3: Softmax Stability

## Goal In This Hint

Optimize runtime and memory only after matching baseline numerics, then freeze a regression check.

## Topic-Specific Mini Example

- q/k/v (B=2,T=5,D=8) -> weights (2,5,5), output (2,5,8).
- Verify the related API path: scaled_dot_product_attention, causal_mask, chunked_attention_scores

## Concrete Failure Mode

- An optimization changes semantics and drifts from Level 1 outputs beyond tolerance.

## Debugging Checklist

- Set a deterministic seed and record one known-good tensor output.
- Assert shape and dtype after every transformation step.
- Check finite values (no NaN/Inf) after normalization, masking, or exponentials.
- Compare Level 3 vs Level 1 with fixed seed and assert max error within tolerance.

## Exit Criteria

- Basic case passes with exact or near-exact expectation.
- One edge case is validated with explicit assertion.
- You can explain why this hint prevents the listed failure mode.

## Next

After this hint, run all topic tests and preserve one failing case as a permanent regression test.
