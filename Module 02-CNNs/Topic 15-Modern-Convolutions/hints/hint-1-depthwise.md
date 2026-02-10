# Hint 1: Depthwise

## Goal In This Hint

Establish strict shape and dtype contracts for the first correct implementation path.

## Topic-Specific Mini Example

- Depthwise (B,32,16,16) + pointwise 32->64 should end at (B,64,16,16).
- Verify the related API path: depthwise_conv2d, pointwise_conv2d, dilated_conv2d

## Concrete Failure Mode

- Depthwise convolution incorrectly mixes channels, turning grouped conv into standard conv.

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

Continue with Hint 2 using file 'hint-2-pointwise.md' after passing the checks below.
