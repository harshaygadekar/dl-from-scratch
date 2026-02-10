# Topic 30 Questions

1. For Mini GPT Training, what exact input/output shape contract must hold at each major step?
2. In 'make_batches, cross_entropy_from_logits, bigram_forward, generate_bigram', which operation is most error-prone and why?
3. What deterministic test (seed + input) would expose this failure mode: "Targets are not shifted by one token, producing deceptively low but meaningless loss."?
4. Which Level 2 or Level 3 optimization gives the best gain without changing numerical behavior?
5. If outputs diverge between Level 1 and Level 2, what is your first debug probe and expected value?
