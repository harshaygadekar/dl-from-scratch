# Hint 1: Alignment Score

## Goal In This Hint

Establish strict shape and dtype contracts for the first correct implementation path.

## Topic-Specific Mini Example

- Scores (B,T_enc), weights softmax over T_enc, context (B,H_enc).
- Verify the related API path: bahdanau_scores, attention_context, attention_decode_step

## Concrete Failure Mode

- Softmax over wrong axis makes weights sum across batch instead of encoder time.

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

Continue with Hint 2 using file 'hint-2-context-vector.md' after passing the checks below.
