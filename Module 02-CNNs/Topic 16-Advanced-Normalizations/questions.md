# Topic 16 Questions

1. For Advanced Normalizations, what exact input/output shape contract must hold at each major step?
2. In 'layer_norm_nchw, group_norm_nchw, instance_norm_nchw', which operation is most error-prone and why?
3. What deterministic test (seed + input) would expose this failure mode: "Computing GroupNorm statistics over batch axis makes outputs batch-size dependent."?
4. Which Level 2 or Level 3 optimization gives the best gain without changing numerical behavior?
5. If outputs diverge between Level 1 and Level 2, what is your first debug probe and expected value?
