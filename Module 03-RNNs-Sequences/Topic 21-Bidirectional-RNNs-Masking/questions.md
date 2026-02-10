# Topic 21 Questions

1. For Bidirectional RNNs Masking, what exact input/output shape contract must hold at each major step?
2. In 'bidirectional_rnn_forward, apply_sequence_mask', which operation is most error-prone and why?
3. What deterministic test (seed + input) would expose this failure mode: "Mask applied after recurrence leaks padded tokens into hidden states."?
4. Which Level 2 or Level 3 optimization gives the best gain without changing numerical behavior?
5. If outputs diverge between Level 1 and Level 2, what is your first debug probe and expected value?
