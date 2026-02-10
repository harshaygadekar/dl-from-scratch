# Hint 1: Encoder Context

## Goal In This Hint

Establish strict shape and dtype contracts for the first correct implementation path.

## Topic-Specific Mini Example

- Encoder states (B,T_enc,H) + decoder inputs (B,T_dec,D) -> logits (B,T_dec,V).
- Verify the related API path: encoder_forward, decoder_step, seq2seq_forward

## Concrete Failure Mode

- Teacher-forcing shift bug uses current token as target and hides exposure bias issues.

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

Continue with Hint 2 using file 'hint-2-teacher-forcing.md' after passing the checks below.
