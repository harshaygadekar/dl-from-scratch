# Hint 1: Max Pool Mask

## Goal In This Hint

Establish strict shape and dtype contracts for the first correct implementation path.

## Topic-Specific Mini Example

- Input (2,3,8,8), kernel=2, stride=2 -> pooled output (2,3,4,4).
- Verify the related API path: max_pool2d_forward, avg_pool2d_forward, max_pool2d_vectorized

## Concrete Failure Mode

- Reusing a max mask across windows causes wrong gradient routing and mismatched pooled outputs.

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

Continue with Hint 2 using file 'hint-2-avg-pool.md' after passing the checks below.
