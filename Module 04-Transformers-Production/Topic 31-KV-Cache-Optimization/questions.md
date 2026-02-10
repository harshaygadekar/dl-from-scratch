# Topic 31 Questions

1. For KV Cache Optimization, what exact input/output shape contract must hold at each major step?
2. In 'init_kv_cache, append_kv, current_kv, cached_attention_step', which operation is most error-prone and why?
3. What deterministic test (seed + input) would expose this failure mode: "Appending without capacity checks overwrites wrong time index and corrupts decoding context."?
4. Which Level 2 or Level 3 optimization gives the best gain without changing numerical behavior?
5. If outputs diverge between Level 1 and Level 2, what is your first debug probe and expected value?
