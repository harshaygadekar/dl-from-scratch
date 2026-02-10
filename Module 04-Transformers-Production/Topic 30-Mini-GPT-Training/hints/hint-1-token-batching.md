# Hint 1: Token Batching

## Goal In This Hint

Establish strict shape and dtype contracts for the first correct implementation path.

## Topic-Specific Mini Example

- input tokens (B,T) -> logits (B,T,V), targets (B,T) from one-step shift.
- Verify the related API path: make_batches, cross_entropy_from_logits, bigram_forward, generate_bigram

## Concrete Failure Mode

- Targets are not shifted by one token, producing deceptively low but meaningless loss.

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

Continue with Hint 2 using file 'hint-2-next-token-loss.md' after passing the checks below.
