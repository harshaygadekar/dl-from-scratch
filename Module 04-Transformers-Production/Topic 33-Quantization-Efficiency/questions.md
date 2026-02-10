# Topic 33 Questions

1. For Quantization Efficiency, what exact input/output shape contract must hold at each major step?
2. In 'symmetric_quant_params, quantize_int8, dequantize_int8, quantize_per_channel', which operation is most error-prone and why?
3. What deterministic test (seed + input) would expose this failure mode: "Scale computed as zero for near-zero tensors causes divide-by-zero and NaN values."?
4. Which Level 2 or Level 3 optimization gives the best gain without changing numerical behavior?
5. If outputs diverge between Level 1 and Level 2, what is your first debug probe and expected value?
