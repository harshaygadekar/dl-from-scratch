# Hint 1: Unroll

## Goal In This Hint

Establish strict shape and dtype contracts for the first correct implementation path.

## Topic-Specific Mini Example

- Input x (T=5,B=2,D=3), h0 (2,H=4) -> output h (5,2,4).
- Verify the related API path: rnn_step, rnn_forward, clip_grad_norm

## Concrete Failure Mode

- Swapping (T,B,D) with (B,T,D) unroll order gives valid shapes but incorrect sequence behavior.

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

Continue with Hint 2 using file 'hint-2-bptt.md' after passing the checks below.
