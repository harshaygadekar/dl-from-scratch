# Hint 1: Layernorm

## Goal In This Hint

Establish strict shape and dtype contracts for the first correct implementation path.

## Topic-Specific Mini Example

- Input (8,64,8,8), GroupNorm groups=8 -> output same shape with finite normalized activations.
- Verify the related API path: layer_norm_nchw, group_norm_nchw, instance_norm_nchw

## Concrete Failure Mode

- Computing GroupNorm statistics over batch axis makes outputs batch-size dependent.

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

Continue with Hint 2 using file 'hint-2-groupnorm.md' after passing the checks below.
