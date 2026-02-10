# Hint 1: Data Pipeline

## Goal In This Hint

Establish strict shape and dtype contracts for the first correct implementation path.

## Topic-Specific Mini Example

- CIFAR batch (64,32,32,3) normalized then model-ready tensor conversion is consistent each epoch.
- Verify the related API path: load_cifar10, CIFAR10DataLoader, random_crop, random_horizontal_flip

## Concrete Failure Mode

- Augmentation modifies labels or channel order (NHWC/NCHW) causing unstable training metrics.

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

Continue with Hint 2 using file 'hint-2-model-baseline.md' after passing the checks below.
