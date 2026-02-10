# Topic 15 Questions

1. For Modern Convolutions, what exact input/output shape contract must hold at each major step?
2. In 'depthwise_conv2d, pointwise_conv2d, dilated_conv2d', which operation is most error-prone and why?
3. What deterministic test (seed + input) would expose this failure mode: "Depthwise convolution incorrectly mixes channels, turning grouped conv into standard conv."?
4. Which Level 2 or Level 3 optimization gives the best gain without changing numerical behavior?
5. If outputs diverge between Level 1 and Level 2, what is your first debug probe and expected value?
