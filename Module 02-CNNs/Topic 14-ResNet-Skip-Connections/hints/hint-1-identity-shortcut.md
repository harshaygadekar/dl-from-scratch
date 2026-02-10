# Hint 1: Identity Shortcut

## Goal In This Hint

Establish strict shape and dtype contracts for the first correct implementation path.

## Topic-Specific Mini Example

- Input (N,C,H,W)=(4,64,16,16) with projection path -> output (4,128,16,16).
- Verify the related API path: residual_block_forward, projection_shortcut

## Concrete Failure Mode

- Projection shortcut omitted when channel count changes, producing silent broadcasting or wrong dimensions.

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

Continue with Hint 2 using file 'hint-2-projection-shortcut.md' after passing the checks below.
