# Hint 1: Update Gate

## Goal In This Hint

Establish strict shape and dtype contracts for the first correct implementation path.

## Topic-Specific Mini Example

- h_t = z_t*h_prev + (1-z_t)*h_tilde with all tensors shaped (B,H).
- Verify the related API path: gru_step, gru_forward

## Concrete Failure Mode

- Using update gate with reversed interpolation flips learning dynamics and stalls convergence.

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

Continue with Hint 2 using file 'hint-2-reset-gate.md' after passing the checks below.
