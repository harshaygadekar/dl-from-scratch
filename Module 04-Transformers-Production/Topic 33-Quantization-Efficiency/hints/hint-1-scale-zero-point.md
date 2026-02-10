# Hint 1: Scale Zero Point

## Goal In This Hint

Establish strict shape and dtype contracts for the first correct implementation path.

## Topic-Specific Mini Example

- x (16,16) -> int8 q same shape; dequantized tensor remains finite with bounded error.
- Verify the related API path: symmetric_quant_params, quantize_int8, dequantize_int8, quantize_per_channel

## Concrete Failure Mode

- Scale computed as zero for near-zero tensors causes divide-by-zero and NaN values.

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

Continue with Hint 2 using file 'hint-2-fake-quant.md' after passing the checks below.
