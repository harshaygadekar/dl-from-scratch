# Hint 1: Qkv Shapes

## Goal In This Hint

Establish strict shape and dtype contracts for the first correct implementation path.

## Topic-Specific Mini Example

- q/k/v (B=2,T=5,D=8) -> weights (2,5,5), output (2,5,8).
- Verify the related API path: scaled_dot_product_attention, causal_mask, chunked_attention_scores

## Concrete Failure Mode

- Mask broadcast uses wrong rank, allowing future-token leakage while tests on shape still pass.

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

Continue with Hint 2 using file 'hint-2-causal-mask.md' after passing the checks below.
