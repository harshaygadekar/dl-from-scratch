# Hint 2: Gradient Aggregation

## Goal In This Hint

Stabilize algorithm branches (masking, padding, stride, gating, or context flow) for non-trivial cases.

## Topic-Specific Mini Example

- Split batch size 17 over 4 workers -> chunks sum back to 17 with deterministic ordering.
- Verify the related API path: split_batch, average_gradients, sync_sgd_step, allreduce_volume_bytes

## Concrete Failure Mode

- Branch-specific logic is applied on the wrong axis or timestep, producing plausible but incorrect outputs.

## Debugging Checklist

- Set a deterministic seed and record one known-good tensor output.
- Assert shape and dtype after every transformation step.
- Check finite values (no NaN/Inf) after normalization, masking, or exponentials.
- Run at least one batch_size=1 and one irregular-size case before scaling up.

## Exit Criteria

- Basic case passes with exact or near-exact expectation.
- One edge case is validated with explicit assertion.
- You can explain why this hint prevents the listed failure mode.

## Next

Continue with Hint 3 using file 'hint-3-sync-costs.md' after passing the checks below.
