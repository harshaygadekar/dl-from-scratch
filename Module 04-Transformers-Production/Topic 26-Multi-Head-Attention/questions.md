# Topic 26 Questions

1. For Multi Head Attention, what exact input/output shape contract must hold at each major step?
2. In 'split_heads, combine_heads, multi_head_attention, multi_head_attention_vectorized', which operation is most error-prone and why?
3. What deterministic test (seed + input) would expose this failure mode: "Incorrect transpose in combine_heads returns permuted token order with same final shape."?
4. Which Level 2 or Level 3 optimization gives the best gain without changing numerical behavior?
5. If outputs diverge between Level 1 and Level 2, what is your first debug probe and expected value?
