# Hint 3: Gradient Clipping

## Goal In This Hint

Optimize runtime and memory only after matching baseline numerics, then freeze a regression check.

## Topic-Specific Mini Example

- Input x (T=5,B=2,D=3), h0 (2,H=4) -> output h (5,2,4).
- Verify the related API path: rnn_step, rnn_forward, clip_grad_norm

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
