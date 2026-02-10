# Topic 28 Questions

1. For Transformer Encoder Block, what exact input/output shape contract must hold at each major step?
2. In 'encoder_block_forward, layer_norm, chunked_ffn', which operation is most error-prone and why?
3. What deterministic test (seed + input) would expose this failure mode: "Applying layer norm before residual add changes block behavior and breaks reference expectations."?
4. Which Level 2 or Level 3 optimization gives the best gain without changing numerical behavior?
5. If outputs diverge between Level 1 and Level 2, what is your first debug probe and expected value?
