# Hint 1: Forward Backward Streams

## Goal In This Hint

Establish strict shape and dtype contracts for the first correct implementation path.

## Topic-Specific Mini Example

- Input lengths [7,5,3] with max_len=7 -> masked outputs for padded steps remain neutral.
- Verify the related API path: bidirectional_rnn_forward, apply_sequence_mask

## Concrete Failure Mode

- Mask applied after recurrence leaks padded tokens into hidden states.

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

Continue with Hint 2 using file 'hint-2-padding-mask.md' after passing the checks below.
