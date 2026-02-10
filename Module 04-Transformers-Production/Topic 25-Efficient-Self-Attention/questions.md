# Topic 25 Questions

1. For Efficient Self Attention, what exact input/output shape contract must hold at each major step?
2. In 'scaled_dot_product_attention, causal_mask, chunked_attention_scores', which operation is most error-prone and why?
3. What deterministic test (seed + input) would expose this failure mode: "Mask broadcast uses wrong rank, allowing future-token leakage while tests on shape still pass."?
4. Which Level 2 or Level 3 optimization gives the best gain without changing numerical behavior?
5. If outputs diverge between Level 1 and Level 2, what is your first debug probe and expected value?
