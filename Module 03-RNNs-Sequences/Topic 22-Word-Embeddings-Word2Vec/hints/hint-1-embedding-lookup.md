# Hint 1: Embedding Lookup

## Goal In This Hint

Establish strict shape and dtype contracts for the first correct implementation path.

## Topic-Specific Mini Example

- Target batch (B,), context positives (B,), negatives (B,K) -> scalar average loss.
- Verify the related API path: embedding_lookup, negative_sampling_loss, subsample_tokens

## Concrete Failure Mode

- Sampling positives inside negative pool collapses learning signal and gradients.

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

Continue with Hint 2 using file 'hint-2-negative-sampling.md' after passing the checks below.
