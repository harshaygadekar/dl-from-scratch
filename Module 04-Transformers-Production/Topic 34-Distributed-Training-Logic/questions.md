# Topic 34 Questions

1. For Distributed Training Logic, what exact input/output shape contract must hold at each major step?
2. In 'split_batch, average_gradients, sync_sgd_step, allreduce_volume_bytes', which operation is most error-prone and why?
3. What deterministic test (seed + input) would expose this failure mode: "Averaging gradients with inconsistent key ordering across workers updates wrong parameters."?
4. Which Level 2 or Level 3 optimization gives the best gain without changing numerical behavior?
5. If outputs diverge between Level 1 and Level 2, what is your first debug probe and expected value?
