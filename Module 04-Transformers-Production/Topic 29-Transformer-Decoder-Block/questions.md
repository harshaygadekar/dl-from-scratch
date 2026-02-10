# Topic 29 Questions

1. For Transformer Decoder Block, what exact input/output shape contract must hold at each major step?
2. In 'decoder_block_forward, decoder_autoregressive_step, truncate_prefix', which operation is most error-prone and why?
3. What deterministic test (seed + input) would expose this failure mode: "Cross-attention key/value mistakenly sourced from decoder prefix instead of encoder outputs."?
4. Which Level 2 or Level 3 optimization gives the best gain without changing numerical behavior?
5. If outputs diverge between Level 1 and Level 2, what is your first debug probe and expected value?
