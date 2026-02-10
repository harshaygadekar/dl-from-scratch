# Hint 1: Head Splitting

## Goal In This Hint

Establish strict shape and dtype contracts for the first correct implementation path.

## Topic-Specific Mini Example

- Input (2,6,16), heads=4 -> split (2,4,6,4) -> combine back (2,6,16).
- Verify the related API path: split_heads, combine_heads, multi_head_attention, multi_head_attention_vectorized

## Concrete Failure Mode

- Incorrect transpose in combine_heads returns permuted token order with same final shape.

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

Continue with Hint 2 using file 'hint-2-head-concat.md' after passing the checks below.
