# Hint 3: Correctness Check

## Goal In This Hint

Optimize runtime and memory only after matching baseline numerics, then freeze a regression check.

## Topic-Specific Mini Example

- Cache K/V (B,H,Tmax,D), append (B,H,1,D), current len increments by one each step.
- Verify the related API path: init_kv_cache, append_kv, current_kv, cached_attention_step

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
