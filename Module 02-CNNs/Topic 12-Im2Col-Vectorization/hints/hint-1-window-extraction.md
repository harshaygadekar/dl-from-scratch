# Hint 1: Window Extraction

## Goal In This Hint

Establish strict shape and dtype contracts for the first correct implementation path.

## Topic-Specific Mini Example

- Input (2,3,8,8), kernels (4,3,3,3), stride=1, pad=1 -> output (2,4,8,8).
- Verify the related API path: im2col_naive, conv2d_im2col_naive

## Concrete Failure Mode

- Column flatten order mismatch (C,K,K vs K,K,C) silently scrambles filters while shapes still look correct.

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

Continue with Hint 2 using file 'hint-2-column-layout.md' after passing the checks below.
