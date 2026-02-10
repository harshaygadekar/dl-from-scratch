# Hint 2: Fake Quant

## Goal In This Hint

Stabilize algorithm branches (masking, padding, stride, gating, or context flow) for non-trivial cases.

## Topic-Specific Mini Example

- x (16,16) -> int8 q same shape; dequantized tensor remains finite with bounded error.
- Verify the related API path: symmetric_quant_params, quantize_int8, dequantize_int8, quantize_per_channel

## Concrete Failure Mode

- Branch-specific logic is applied on the wrong axis or timestep, producing plausible but incorrect outputs.

## Debugging Checklist

- Set a deterministic seed and record one known-good tensor output.
- Assert shape and dtype after every transformation step.
- Check finite values (no NaN/Inf) after normalization, masking, or exponentials.
- Run at least one batch_size=1 and one irregular-size case before scaling up.

## Exit Criteria

- Basic case passes with exact or near-exact expectation.
- One edge case is validated with explicit assertion.
- You can explain why this hint prevents the listed failure mode.

## Next

Continue with Hint 3 using file 'hint-3-error-analysis.md' after passing the checks below.
