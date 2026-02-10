# Topic 27 Questions

1. For Positional Encodings, what exact input/output shape contract must hold at each major step?
2. In 'sinusoidal_positional_encoding, add_positional_encoding, apply_rope', which operation is most error-prone and why?
3. What deterministic test (seed + input) would expose this failure mode: "RoPE applied to odd embedding dimension silently truncates channels or crashes late in training."?
4. Which Level 2 or Level 3 optimization gives the best gain without changing numerical behavior?
5. If outputs diverge between Level 1 and Level 2, what is your first debug probe and expected value?
