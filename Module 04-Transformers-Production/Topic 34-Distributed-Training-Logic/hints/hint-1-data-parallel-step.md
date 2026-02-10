# Hint 1: Data Parallel Step

## Goal In This Hint

Establish strict shape and dtype contracts for the first correct implementation path.

## Topic-Specific Mini Example

- Split batch size 17 over 4 workers -> chunks sum back to 17 with deterministic ordering.
- Verify the related API path: split_batch, average_gradients, sync_sgd_step, allreduce_volume_bytes

## Concrete Failure Mode

- Averaging gradients with inconsistent key ordering across workers updates wrong parameters.

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

Continue with Hint 2 using file 'hint-2-gradient-aggregation.md' after passing the checks below.
