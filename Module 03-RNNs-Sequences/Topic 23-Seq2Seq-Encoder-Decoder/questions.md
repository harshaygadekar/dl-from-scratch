# Topic 23 Questions

1. For Seq2Seq Encoder Decoder, what exact input/output shape contract must hold at each major step?
2. In 'encoder_forward, decoder_step, seq2seq_forward', which operation is most error-prone and why?
3. What deterministic test (seed + input) would expose this failure mode: "Teacher-forcing shift bug uses current token as target and hides exposure bias issues."?
4. Which Level 2 or Level 3 optimization gives the best gain without changing numerical behavior?
5. If outputs diverge between Level 1 and Level 2, what is your first debug probe and expected value?
