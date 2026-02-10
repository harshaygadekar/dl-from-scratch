# Topic 32 Questions

1. For Modern Optimizations, what exact input/output shape contract must hold at each major step?
2. In 'split_segments, checkpoint_plan, blockwise_qk_scores, activation_memory_bytes', which operation is most error-prone and why?
3. What deterministic test (seed + input) would expose this failure mode: "Block loop misses tail segment, dropping tokens from attention score matrix."?
4. Which Level 2 or Level 3 optimization gives the best gain without changing numerical behavior?
5. If outputs diverge between Level 1 and Level 2, what is your first debug probe and expected value?
