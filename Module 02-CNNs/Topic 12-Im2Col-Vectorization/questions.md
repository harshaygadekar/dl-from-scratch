# Topic 12 Questions

1. For Im2Col Vectorization, what exact input/output shape contract must hold at each major step?
2. In 'im2col_naive, conv2d_im2col_naive', which operation is most error-prone and why?
3. What deterministic test (seed + input) would expose this failure mode: "Column flatten order mismatch (C,K,K vs K,K,C) silently scrambles filters while shapes still look correct."?
4. Which Level 2 or Level 3 optimization gives the best gain without changing numerical behavior?
5. If outputs diverge between Level 1 and Level 2, what is your first debug probe and expected value?
