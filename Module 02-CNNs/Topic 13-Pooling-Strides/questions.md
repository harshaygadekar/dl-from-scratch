# Topic 13 Questions

1. For Pooling Strides, what exact input/output shape contract must hold at each major step?
2. In 'max_pool2d_forward, avg_pool2d_forward, max_pool2d_vectorized', which operation is most error-prone and why?
3. What deterministic test (seed + input) would expose this failure mode: "Reusing a max mask across windows causes wrong gradient routing and mismatched pooled outputs."?
4. Which Level 2 or Level 3 optimization gives the best gain without changing numerical behavior?
5. If outputs diverge between Level 1 and Level 2, what is your first debug probe and expected value?
