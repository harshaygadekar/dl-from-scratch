# Topic 19 Questions

1. For LSTM Gates, what exact input/output shape contract must hold at each major step?
2. In 'lstm_step, lstm_forward', which operation is most error-prone and why?
3. What deterministic test (seed + input) would expose this failure mode: "Forgetting to carry c_t causes model to behave like shallow RNN despite passing shape tests."?
4. Which Level 2 or Level 3 optimization gives the best gain without changing numerical behavior?
5. If outputs diverge between Level 1 and Level 2, what is your first debug probe and expected value?
