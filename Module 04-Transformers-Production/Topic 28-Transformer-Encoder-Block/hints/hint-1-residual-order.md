# Hint 1: Residual Order

## Goal In This Hint

Establish strict shape and dtype contracts for the first correct implementation path.

## Topic-Specific Mini Example

- x (2,6,16) through block parameters -> y (2,6,16) with finite outputs.
- Verify the related API path: encoder_block_forward, layer_norm, chunked_ffn

## Concrete Failure Mode

- Applying layer norm before residual add changes block behavior and breaks reference expectations.

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

Continue with Hint 2 using file 'hint-2-layernorm-placement.md' after passing the checks below.
