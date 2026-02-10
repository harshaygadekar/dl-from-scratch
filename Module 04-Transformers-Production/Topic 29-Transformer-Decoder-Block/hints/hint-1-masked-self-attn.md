# Hint 1: Masked Self Attn

## Goal In This Hint

Establish strict shape and dtype contracts for the first correct implementation path.

## Topic-Specific Mini Example

- prefix (B,64,D), enc_out (B,64,D) -> autoregressive step output (B,D).
- Verify the related API path: decoder_block_forward, decoder_autoregressive_step, truncate_prefix

## Concrete Failure Mode

- Cross-attention key/value mistakenly sourced from decoder prefix instead of encoder outputs.

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

Continue with Hint 2 using file 'hint-2-cross-attn.md' after passing the checks below.
