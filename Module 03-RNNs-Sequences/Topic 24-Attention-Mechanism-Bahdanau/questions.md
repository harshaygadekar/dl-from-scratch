# Topic 24 Questions

1. For Attention Mechanism Bahdanau, what exact input/output shape contract must hold at each major step?
2. In 'bahdanau_scores, attention_context, attention_decode_step', which operation is most error-prone and why?
3. What deterministic test (seed + input) would expose this failure mode: "Softmax over wrong axis makes weights sum across batch instead of encoder time."?
4. Which Level 2 or Level 3 optimization gives the best gain without changing numerical behavior?
5. If outputs diverge between Level 1 and Level 2, what is your first debug probe and expected value?
