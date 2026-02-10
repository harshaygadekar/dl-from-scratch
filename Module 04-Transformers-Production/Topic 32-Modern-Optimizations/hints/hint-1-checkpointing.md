# Hint 1: Checkpointing

## Goal In This Hint

Establish strict shape and dtype contracts for the first correct implementation path.

## Topic-Specific Mini Example

- blockwise_qk_scores on q/k (2,256,64) returns full score tensor (2,256,256).
- Verify the related API path: split_segments, checkpoint_plan, blockwise_qk_scores, activation_memory_bytes

## Concrete Failure Mode

- Block loop misses tail segment, dropping tokens from attention score matrix.

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

Continue with Hint 2 using file 'hint-2-attention-tiling.md' after passing the checks below.
