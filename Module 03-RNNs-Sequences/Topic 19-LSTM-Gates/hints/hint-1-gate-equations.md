# Hint 1: Gate Equations

## Goal In This Hint

Establish strict shape and dtype contracts for the first correct implementation path.

## Topic-Specific Mini Example

- For B=4,H=16 each gate tensor should be (4,16) and c_t/h_t maintain same hidden size.
- Verify the related API path: lstm_step, lstm_forward

## Concrete Failure Mode

- Forgetting to carry c_t causes model to behave like shallow RNN despite passing shape tests.

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

Continue with Hint 2 using file 'hint-2-cell-state.md' after passing the checks below.
