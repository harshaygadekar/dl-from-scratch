# Topic 17 Questions

1. For CIFAR10 From Scratch, what exact input/output shape contract must hold at each major step?
2. In 'load_cifar10, CIFAR10DataLoader, random_crop, random_horizontal_flip', which operation is most error-prone and why?
3. What deterministic test (seed + input) would expose this failure mode: "Augmentation modifies labels or channel order (NHWC/NCHW) causing unstable training metrics."?
4. Which Level 2 or Level 3 optimization gives the best gain without changing numerical behavior?
5. If outputs diverge between Level 1 and Level 2, what is your first debug probe and expected value?
