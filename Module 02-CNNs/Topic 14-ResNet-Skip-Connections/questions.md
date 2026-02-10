# Topic 14 Questions

1. For ResNet Skip Connections, what exact input/output shape contract must hold at each major step?
2. In 'residual_block_forward, projection_shortcut', which operation is most error-prone and why?
3. What deterministic test (seed + input) would expose this failure mode: "Projection shortcut omitted when channel count changes, producing silent broadcasting or wrong dimensions."?
4. Which Level 2 or Level 3 optimization gives the best gain without changing numerical behavior?
5. If outputs diverge between Level 1 and Level 2, what is your first debug probe and expected value?
