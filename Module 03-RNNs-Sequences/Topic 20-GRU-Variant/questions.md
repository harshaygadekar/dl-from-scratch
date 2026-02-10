# Topic 20 Questions

1. For GRU Variant, what exact input/output shape contract must hold at each major step?
2. In 'gru_step, gru_forward', which operation is most error-prone and why?
3. What deterministic test (seed + input) would expose this failure mode: "Using update gate with reversed interpolation flips learning dynamics and stalls convergence."?
4. Which Level 2 or Level 3 optimization gives the best gain without changing numerical behavior?
5. If outputs diverge between Level 1 and Level 2, what is your first debug probe and expected value?
