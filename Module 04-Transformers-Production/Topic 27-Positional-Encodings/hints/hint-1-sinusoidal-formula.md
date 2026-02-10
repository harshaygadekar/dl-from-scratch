# Hint 1: Sinusoidal Formula

## Goal In This Hint

Establish strict shape and dtype contracts for the first correct implementation path.

## Topic-Specific Mini Example

- PE(10,16) -> (10,16); add to x (B,10,16) preserves shape and dtype.
- Verify the related API path: sinusoidal_positional_encoding, add_positional_encoding, apply_rope

## Concrete Failure Mode

- RoPE applied to odd embedding dimension silently truncates channels or crashes late in training.

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

Continue with Hint 2 using file 'hint-2-rope-intuition.md' after passing the checks below.
