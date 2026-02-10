# Topic 18 Questions

1. For Vanilla RNN, what exact input/output shape contract must hold at each major step?
2. In 'rnn_step, rnn_forward, clip_grad_norm', which operation is most error-prone and why?
3. What deterministic test (seed + input) would expose this failure mode: "Swapping (T,B,D) with (B,T,D) unroll order gives valid shapes but incorrect sequence behavior."?
4. Which Level 2 or Level 3 optimization gives the best gain without changing numerical behavior?
5. If outputs diverge between Level 1 and Level 2, what is your first debug probe and expected value?
