# Hint 1: Cache Layout

## Goal In This Hint

Establish strict shape and dtype contracts for the first correct implementation path.

## Topic-Specific Mini Example

- Cache K/V (B,H,Tmax,D), append (B,H,1,D), current len increments by one each step.
- Verify the related API path: init_kv_cache, append_kv, current_kv, cached_attention_step

## Concrete Failure Mode

- Appending without capacity checks overwrites wrong time index and corrupts decoding context.

## Debugging Checklist

- Set a deterministic seed and record one known-good tensor output.
- Assert shape and dtype after every transformation step.
- Check finite values (no NaN/Inf) after normalization, masking, or exponentials.
- Verify every reshape and transpose with an assert immediately after it.

## Exit Criteria

- Basic case passes with exact or near-exact expectation.
- One edge case is validated with explicit assertion.
- You can explain why this hint prevents the listed failure mode.

## Next

Continue with Hint 2 using file 'hint-2-append-vs-concat.md' after passing the checks below.
